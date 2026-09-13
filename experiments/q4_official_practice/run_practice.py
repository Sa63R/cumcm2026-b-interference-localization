"""Run exactly one session after visually confirming Q4 PRACTICE in the UI."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid
from runtime import METHODS, SimulatorClient, prepare_policy, run_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='v4')
    parser.add_argument('--robot-id', required=True, help='Team ID shown by the simulator, not a password')
    parser.add_argument('--practice-confirmed', action='store_true',
                        help='Operator has just verified Q4 PRACTICE in the simulator UI')
    parser.add_argument('--case-label', default='not_recorded')
    parser.add_argument('--seed', type=int, default=90210000, help='Planner seed only; does not set official case')
    parser.add_argument('--output', type=Path, default=Path('practice_results'))
    args = parser.parse_args()
    if not args.practice_confirmed:
        parser.error('No connection made. First visually verify Q4 PRACTICE, then pass --practice-confirmed.')
    run_dir = args.output / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.method + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    state, planner, geometry = prepare_policy(args.method, args.seed)
    report = dict(method=args.method, case_label=args.case_label, geometry=geometry,
                  status='not_entered', official_ui_feedback='not_recorded',
                  mode='operator_confirmed_q4_practice',
                  mode_note='API cannot query mode; flag is an operator assertion, not server verification.',
                  python=sys.version, platform=platform.platform(), planner_seed=args.seed)
    tic = time.perf_counter()
    try:
        with SimulatorClient(args.robot_id, log_path=run_dir / 'requests.jsonl') as client:
            try:
                report['enter_response'] = client.enter()
                report['status'] = 'entered'
                def progress(state, client):
                    print(f"actions={state.actions} cleared={client.state.cleared_count} "
                          f"virtual_s={client.state.virtual_time_s:.2f} "
                          f"remaining_real_s={client.remaining_real_time_s:.1f}", flush=True)
                report['policy'] = run_policy(client, state, planner, progress)
                report['exit_response'] = client.exit()
                report['status'] = 'policy_completed_and_exited'
            except BaseException:
                report['status'] = 'interrupted_or_failed'
                report['exception'] = traceback.format_exc()
                # Never send a different request after an uncertain outcome.
                if client.state.session == 'active' and client.pending_request is None:
                    try:
                        report['error_exit_response'] = client.exit()
                    except Exception:
                        report['error_exit_exception'] = traceback.format_exc()
                raise
            finally:
                report['client_state'] = client.state.snapshot()
                report['pending_request'] = client.pending_request
                count = client.state.cleared_count
                report['seconds_per_accepted_clear'] = client.state.virtual_time_s / count if count else None
    finally:
        report['wall_seconds'] = time.perf_counter() - tic
        (run_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Result saved: {run_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()

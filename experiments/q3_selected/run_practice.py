"""Run one visually confirmed Q3 practice session in the official simulator."""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid
from q3_runtime import HERE, METHODS, VERSION, SimulatorClient, run_policy, v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--robot-id', required=True)
    ap.add_argument('--case-label', default='not_recorded')
    ap.add_argument('--practice-confirmed', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('results/q3/selected_practice'))
    args = ap.parse_args(argv)
    if not args.practice_confirmed:
        ap.error('Visually confirm Q3 PRACTICE before passing --practice-confirmed')
    folder = args.output / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.method + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    import numpy
    report = dict(method=args.method, version=VERSION, case_label=args.case_label,
                  status='not_entered', mode='operator_confirmed_q3_practice',
                  python=sys.version, platform=platform.platform(), numpy=numpy.__version__,
                  numba_available=importlib.util.find_spec('numba') is not None,
                  mode_note='Mode and case label are visually verified; API cannot query them.')
    code_files = sorted(HERE.glob('*.py')) + sorted((HERE/'vendor').glob('*.py'))
    report['source_sha256'] = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}
    report['parameters'] = ({'initial_channels': 20} if args.method == 'v3_origin20' else
                            {'initial_channels': 0, 'threshold': 0.4, 'max_radius': 80.0, 'cap': 2})
    start = time.perf_counter()
    try:
        # Optional JIT initialization happens before the simulator's /enter.
        preparation = time.perf_counter()
        v.warmup()
        report['preparation_wall_seconds'] = time.perf_counter() - preparation
        with SimulatorClient(args.robot_id, log_path=folder/'requests.jsonl') as client:
            try:
                report['enter_response'] = client.enter()
                report['status'] = 'entered'
                def progress(c):
                    if c.state.accepted_actions % 20 == 0:
                        print(f'actions={c.state.accepted_actions} cleared={c.state.cleared_count} virtual_s={c.state.virtual_time_s:.3f}', flush=True)
                report['policy'] = run_policy(client, args.method, progress)
                report['exit_response'] = client.exit()
                report['status'] = 'policy_completed_and_exited'
            except BaseException:
                report['status'] = 'interrupted_or_failed'
                report['exception'] = traceback.format_exc()
                if client.state.session == 'active' and client.pending_request is None:
                    try:
                        report['error_exit_response'] = client.exit()
                    except Exception:
                        report['error_exit_exception'] = traceback.format_exc()
                raise
            finally:
                report['client_state'] = client.state.snapshot()
                report['pending_request'] = client.pending_request
                n = client.state.cleared_count
                report['seconds_per_accepted_clear'] = client.state.virtual_time_s/n if n else None
    finally:
        report['wall_seconds'] = time.perf_counter()-start
        (folder/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('RESULT_SAVED=' + str(folder.resolve()), flush=True)
    return folder


if __name__ == '__main__':
    main()

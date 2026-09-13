"""Run Q3 origin-20 scanning in the current official simulator session.

Select question 3 and practice/formal in the simulator UI. No mode argument is
needed. --check-only verifies files and imports without starting any test.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
METHOD = 'v3_origin20'


def prepare():
    manifest = json.loads((ROOT / 'q3_origin/source_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen Q3 file changed: {name}')
    runtime_path = ROOT / 'q3_origin/code/experiments/q3_official_practice'
    libraries = [ROOT / 'q3_origin/lib', ROOT.parent / 'Q3Practice/lib']
    sys.path[:0] = [str(path) for path in libraries if path.is_dir()] + [str(runtime_path)]
    import runtime
    if runtime.v.CONFIGS[METHOD] != {'initial_channels': 20}:
        raise RuntimeError('Unexpected Q3 origin-scan configuration')
    return runtime, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-id')
    parser.add_argument('--case-label')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true',
                        help='Verify files and dependencies without running a test.')
    args = parser.parse_args(argv)
    if args.check_only:
        runtime, manifest = prepare()
        print(json.dumps(dict(status='preflight_passed', method=METHOD,
                              initial_channels=20, verified_files=len(manifest['files']),
                              numpy=runtime.np.__version__, numpy_file=runtime.np.__file__,
                              numba_available=importlib.util.find_spec('numba') is not None,
                              simulator_contacted=False, test_started=False), indent=2))
        return
    missing = [name for name in ('robot_id', 'case_label', 'output')
               if not getattr(args, name)]
    if missing:
        parser.error('No connection made: supply ' + ', '.join('--' + name.replace('_', '-')
                     for name in missing) + '.')
    runtime, manifest = prepare()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(method=METHOD, question=3, case_label=args.case_label,
                  status='not_entered', mode='unspecified',
                  mode_note='The official simulator controls the mode; no mode argument is required.',
                  started_utc=datetime.now(timezone.utc).isoformat(), source_manifest=manifest,
                  python=sys.version, platform=platform.platform(), numpy=runtime.np.__version__,
                  numba_available=importlib.util.find_spec('numba') is not None)
    start = time.perf_counter()
    print(f'Q3 origin-20 | case={args.case_label} | robot={args.robot_id}', flush=True)
    try:
        with runtime.SimulatorClient(args.robot_id, log_path=args.output / 'requests.jsonl') as client:
            try:
                report['enter_response'] = client.enter()
                report['status'] = 'entered'
                def progress(c):
                    if c.state.accepted_actions % 20 == 0:
                        print(f'actions={c.state.accepted_actions} cleared={c.state.cleared_count} '
                              f'virtual_s={c.state.virtual_time_s:.3f}', flush=True)
                report['policy'] = runtime.run_policy(client, METHOD, progress)
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
                report['seconds_per_accepted_clear'] = client.state.virtual_time_s / n if n else None
    finally:
        report['wall_seconds'] = time.perf_counter() - start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('RESULT_SAVED=' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()

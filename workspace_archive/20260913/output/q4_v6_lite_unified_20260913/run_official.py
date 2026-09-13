"""Run one Q4 practice or formal test using the published, unmodified V6 Lite.

Select the actual mode in the official simulator first. --mode records that
operator selection; the business API cannot query or change simulator mode.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'source_v6_lite')]
import q4_baseline as b
from q4_v6_lite import solve_v6_lite, default_config
from q4_coverage import certify
from q4_v3_solver import validate_layout
from simulator_client import SimulatorClient


def prepare():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen file changed: {name}')
    b.DELTA = math.radians(1.005)
    b.COS_D, b.TAN_D = math.cos(b.DELTA), math.tan(b.DELTA)
    assert math.sqrt(21/16 - b.COS_D + math.sin(b.DELTA)/2) < b.KAPPA
    cfg = default_config()
    n1, n2, r1, r2 = cfg.base.ring_sites
    sites = [(0., 0.)] + [b.mul(r, b.unit(2*math.pi*k/n)) for n, r in [(n1, r1), (n2, r2)] for k in range(n)]
    validate_layout(tuple(sites))
    certificate = certify(sites)
    assert certificate['ok']
    return sites, certificate, manifest


class OfficialDevice:
    def __init__(self, client, progress=None):
        self.__client = client
        self._position = (client.state.position.x, client.state.position.y)
        self.measured = {c: set() for c in range(1, 21)}
        self.discovered = set()
        self.cleared = set()
        self.last_success_virtual_s = None
        self.progress = progress

    @property
    def position(self):
        return self._position

    @property
    def channel(self):
        return self.__client.state.current_channel

    def move(self, point):
        if len(point) != 2 or not all(math.isfinite(v) and abs(v) <= 2_000_000 for v in point):
            raise ValueError('Invalid destination')
        self._position = tuple(map(float, point))

    def tick(self):
        if self.progress and self.__client.state.accepted_actions % 25 == 0:
            self.progress(self.__client)

    def detect(self, channel):
        response = self.__client.measure(self.position, channel)
        self.measured[channel].add(self.position)
        kind = response['measure_result']
        if kind in ('direction', 'near'):
            self.discovered.add(channel)
        self.tick()
        if kind == 'direction':
            return b.Observation('bearing', math.radians(response['svd_deg']))
        if kind == 'near':
            return b.Observation('strong')
        if kind == 'no_signal':
            return b.Observation('none')
        raise RuntimeError('Unexpected measurement response')

    def clear(self, channel):
        response = self.__client.clear(self.position, channel)
        kind = response['clear_result']
        if kind not in ('success', 'no_target_in_range'):
            raise RuntimeError('Unexpected clear response')
        if kind == 'success':
            assert channel not in self.cleared
            self.cleared.add(channel)
            self.last_success_virtual_s = response['virtual_time_s']
        self.tick()
        return kind == 'success'


def run_session(client, report, sites, certificate, progress=None):
    device = None
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    try:
        report['enter_response'] = client.enter()
        report['status'] = 'entered'
        device = OfficialDevice(client, progress)
        policy_start_cpu, policy_start_wall = time.process_time(), time.perf_counter()
        try:
            report['policy'] = solve_v6_lite(device)
        finally:
            report['policy_cpu_seconds'] = time.process_time() - policy_start_cpu
            report['policy_wall_seconds'] = time.perf_counter() - policy_start_wall
        assert device.discovered <= device.cleared
        assert len(device.cleared) == client.state.cleared_count
        if len(device.cleared) == 16:
            assert report['policy']['stop_certificate'] == 'source_upper_bound'
            report['completion_audit'] = dict(kind='source_upper_bound', successful_clears=16)
        else:
            assert report['policy']['stop_certificate'] == 'coverage_complete'
            assert sorted(report['policy']['visited_stations']) == list(range(21))
            for channel in set(range(1, 21)) - device.cleared:
                assert set(sites) <= device.measured[channel], f'Incomplete coverage for {channel}'
            report['completion_audit'] = dict(kind='coverage_complete', certificate=certificate,
                                            all_undiscovered_channels_measured_at_21_sites=True)
        report['exit_response'] = client.exit()
        report['status'] = 'policy_completed_and_exited'
    except BaseException:
        report['status'] = 'interrupted_or_failed'
        report['exception'] = traceback.format_exc()
        # Unknown requests remain unresolved; never replace them with a new exit.
        if client.state.session == 'active' and client.pending_request is None:
            try:
                report['error_exit_response'] = client.exit()
            except Exception:
                report['error_exit_exception'] = traceback.format_exc()
        raise
    finally:
        report['cpu_seconds'] = time.process_time() - started_cpu
        report['wall_seconds'] = time.perf_counter() - started_wall
        report['client_state'] = client.state.snapshot()
        report['pending_request'] = client.pending_request
        report['successful_clear_channels'] = sorted(device.cleared) if device else []
        report['last_success_virtual_s'] = device.last_success_virtual_s if device else None
        count = client.state.cleared_count
        report['seconds_per_accepted_clear'] = client.state.virtual_time_s / count if count else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-id')
    parser.add_argument('--case-label')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--mode', choices=['practice', 'formal'],
                       help='Mode already selected in the official simulator.')
    modes.add_argument('--practice-confirmed', dest='mode', action='store_const',
                       const='practice', help='Compatibility alias for --mode practice.')
    modes.add_argument('--formal-confirmed', dest='mode', action='store_const',
                       const='formal', help='Alias for --mode formal.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true',
                        help='Check files and coverage without contacting any simulator.')
    args = parser.parse_args(argv)
    if args.check_only:
        _, certificate, manifest = prepare()
        print(json.dumps(dict(status='preflight_passed', simulator_contacted=False,
                              method='v6_lite', source_commit=manifest['strategy_commit'],
                              verified_files=len(manifest['files']),
                              coverage_ok=certificate['ok']), indent=2))
        return
    missing = [name for name in ('mode', 'robot_id', 'case_label', 'output')
               if not getattr(args, name)]
    if missing:
        parser.error('No connection made: supply ' + ', '.join('--' + name.replace('_', '-')
                     for name in missing) + '. Select practice or formal in the simulator first.')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(method='v6_lite', source_commit='ed795304a37a5cce8541c046a0ec0a0e4e88ba6d',
                  mode=f'operator_confirmed_q4_{args.mode}', case_label=args.case_label,
                  mode_note=f'{args.mode.capitalize()} mode was selected by the operator; '
                            'the business API cannot query or change it.',
                  started_utc=datetime.now(timezone.utc).isoformat(), status='preparing',
                  python=sys.version, platform=platform.platform(), bearing_bound_deg=1.005)
    try:
        started = time.perf_counter()
        sites, certificate, manifest = prepare()
        report['preparation_wall_seconds'] = time.perf_counter() - started
        report['source_manifest'] = manifest
        def progress(client):
            print(f'actions={client.state.accepted_actions} cleared={client.state.cleared_count} '
                  f'virtual_s={client.state.virtual_time_s:.3f} remaining_real_s={client.remaining_real_time_s:.1f}', flush=True)
        print(f'V6 Lite | mode={args.mode} | case={args.case_label} | robot={args.robot_id}', flush=True)
        with SimulatorClient(args.robot_id, log_path=args.output / 'requests.jsonl') as client:
            run_session(client, report, sites, certificate, progress)
    except BaseException:
        report.setdefault('exception', traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Saved result:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()

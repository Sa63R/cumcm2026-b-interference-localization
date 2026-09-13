"""Frozen paired V4/V5/V6 evaluation on the reconstructed public protocol.

Run from any directory. No official service, authentication or hidden-state
policy input. Complete raw sessions are retained for deterministic replay.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'source_v6'), str(ROOT/'simulator')]
from jammers_local.core import Scenario, Session, LocalClient, Source


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def verify_sources():
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    for category, directory in [('v6_files', 'source_v6'), ('simulator_files', 'simulator')]:
        for name, expected in manifest[category].items():
            actual = hashlib.sha256((ROOT/directory/name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Source changed: {directory}/{name}')
    return hashlib.sha256((ROOT/'source_manifest.json').read_bytes()).hexdigest()


def freeze():
    path = ROOT/'plan.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite frozen plan')
    cases = []
    for i in range(300):
        scene = Scenario.generate(4, f'q4-v6-reconstructed-20260912-confirmation-{i:04d}')
        cases.append(dict(key=f'main-{i:04d}', group='practice_generator', scenario=scene.as_dict()))
    for group in ['all_directional', 'boundary_outward_r1000']:
        for i in range(35):
            n = 10+i%7
            trial = 0
            while True:
                label = f'q4-v6-reconstructed-20260912-{group}-{i:04d}-{trial}'
                scene = Scenario.generate(4, label)
                if len(scene.sources) == n:
                    break
                trial += 1
            rng = random.Random(label)
            sources = []
            for source in scene.sources:
                theta = rng.random()*math.tau
                if group == 'all_directional':
                    sources.append(replace(source, kind='directional', direction_udeg=int(math.degrees(theta)*1e6)))
                else:
                    radius = rng.uniform(1760., 1769.)
                    sources.append(Source(source.channel, round(radius*math.cos(theta)*1e6),
                        round(radius*math.sin(theta)*1e6), 1_000_000_000, 'directional', int(math.degrees(theta)*1e6)))
            scene = replace(scene, sources=tuple(sources), seed_hex=None, label=label+'-stress-transform')
            cases.append(dict(key=f'{group}-{i:04d}', group=group, scenario=scene.as_dict()))
    dump(path, dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_manifest_sha256=verify_sources(), methods=['v4', 'v5', 'v6'],
        primary_group='practice_generator', primary_case_count=300,
        stress_case_count=70, bearing_bound_deg=1.005,
        simulator='local static reconstruction, not official runtime',
        metric='mean of complete virtual task time divided by source count, paired by case',
        runtime='warmed imports/models/layout; policy and local protocol CPU/wall; audit/export excluded',
        limitations='Stress distributions excluded from primary mean. No tuning based on results. Official runtime equivalence not established.',
        cases=cases))


def initialize(bound):
    global b, solve_v4, solve_v5, solve_v6, V6_CONFIG, SITES, CERTIFICATE
    import q4_baseline as b
    from q4_v4_solver import solve_v4, V4Config
    from q4_v5 import solve_v5
    from q4_v6 import solve_v6, default_config
    from q4_transit_critic import Critic
    from q4_v3_solver import validate_layout
    from q4_coverage import certify
    b.DELTA = math.radians(bound)
    b.COS_D, b.TAN_D = math.cos(b.DELTA), math.tan(b.DELTA)
    contraction = math.sqrt(21/16-b.COS_D+math.sin(b.DELTA)/2)
    assert contraction < b.KAPPA
    V6_CONFIG = replace(default_config(), route_model=str(ROOT/'source_v6/route_critic_extra.json'),
                        transit_model=str(ROOT/'source_v6/transit_critic_big_extra.json'))
    Critic(V6_CONFIG.route_model)
    Critic(V6_CONFIG.transit_model)
    n1, n2, r1, r2 = V4Config().ring_sites
    SITES = [(0.,0.)]+[b.mul(r,b.unit(2*math.pi*k/n)) for n,r in [(n1,r1),(n2,r2)] for k in range(n)]
    validate_layout(tuple(SITES))
    CERTIFICATE = certify(SITES)
    assert CERTIFICATE['ok']


class Device:
    """Only accepted measure/clear feedback; move is combined with next action."""
    __slots__ = ('__client', '_position', '_channel')
    def __init__(self, client):
        self.__client = client
        self._position, self._channel = (0.,0.), 1
    @property
    def position(self): return self._position
    @property
    def channel(self): return self._channel
    def move(self, point):
        if len(point) != 2 or not all(math.isfinite(v) and abs(v)<=2_000_000 for v in point):
            raise ValueError('Invalid destination')
        self._position = tuple(map(float, point))
    def detect(self, channel):
        response = self.__client.measure(self.position, channel)
        self._channel = channel
        kind = response['measure_result']
        if kind == 'direction': return b.Observation('bearing', math.radians(response['svd_deg']))
        if kind == 'near': return b.Observation('strong')
        if kind == 'no_signal': return b.Observation('none')
        raise RuntimeError(f'Unexpected measure response: {kind}')
    def clear(self, channel):
        response = self.__client.clear(self.position, channel)
        if response['clear_result'] not in ('success', 'no_target_in_range'):
            raise RuntimeError('Unexpected clear response')
        return response['clear_result'] == 'success'


def audit(session, report):
    engine = session.engine
    assert len(engine.cleared) == len(engine.sources), 'Not all sources cleared'
    assert engine.time_us == sum(engine.costs_us.values()), 'Microsecond accounting mismatch'
    assert engine.time_us < engine.max_virtual_us, 'Virtual time limit exceeded'
    clears = [r for r in session.history if r['response'].get('clear_result') == 'success']
    assert len(clears) == len(engine.cleared), 'Clear feedback count mismatch'
    assert engine.costs_us['detection'] == engine.measurements*5_000_000
    assert engine.costs_us['switching'] == engine.switches*1_000_000
    assert engine.costs_us['optical'] == engine.clear_attempts*3_000_000
    assert engine.costs_us['removal'] == len(clears)*2_000_000
    if len(clears) == 16:
        assert report['stop_certificate'] == 'source_upper_bound'
        return dict(kind='source_upper_bound', accepted_clears=16)
    assert report['stop_certificate'] == 'coverage_complete'
    assert sorted(report['visited_stations']) == list(range(21))
    # Independently check every still-unseen channel was measured at every
    # certified site. A low posterior probability is never a stopping reason.
    observed = {c:set() for c in range(1,21)}
    for record in session.history:
        if record['path'] == '/measure':
            request = record['request']; point = request['position']
            observed[request['channel']].add((point['x'],point['y']))
    for channel in set(range(1,21))-engine.cleared:
        assert set(SITES) <= observed[channel], f'Incomplete absence scan for channel {channel}'
    return dict(kind='coverage_complete', continuous_certificate=CERTIFICATE,
                all_undiscovered_channels_measured_at_21_sites=True)


def time_limit(signum, frame):
    raise TimeoutError('Harness wall limit of 180 seconds per policy')


def run_case(task):
    case, methods, destination = task
    scene = Scenario.from_dict(case['scenario'])
    rows = []
    for method in methods:
        session = Session(scene)
        client = LocalClient(session.dispatch)
        report = certificate = error = None
        cpu, wall = time.process_time(), time.perf_counter()
        signal.signal(signal.SIGALRM, time_limit)
        signal.setitimer(signal.ITIMER_REAL, 180.)
        try:
            client.enter()
            device = Device(client)
            if method == 'v4': report = solve_v4(device)
            elif method == 'v5': report = solve_v5(device)
            elif method == 'v6': report = solve_v6(device, V6_CONFIG)
            else: raise ValueError(method)
            client.exit()
        except Exception:
            error = traceback.format_exc()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        cpu, wall = time.process_time()-cpu, time.perf_counter()-wall
        if error is None:
            try: certificate = audit(session, report)
            except Exception: error = traceback.format_exc()
        snapshot = session.engine.snapshot(truth=True)
        snapshot.pop('scenario')
        successful = [r['response']['virtual_time_s'] for r in session.history if r['response'].get('clear_result') == 'success']
        canonical = [{k:v for k,v in r.items() if k != 'index'} for r in session.history]
        for record in canonical:
            record['response'] = {k:v for k,v in record['response'].items() if k not in ('real_timestamp_ms','remaining_real_duration_s')}
        row = dict(case_key=case['key'], group=case['group'], case_id=scene.case_id, method=method,
            **snapshot, error=error, cpu_seconds=cpu, wall_seconds=wall,
            seconds_per_source=snapshot['virtual_time_s']/len(scene.sources),
            directional_count=sum(s.kind=='directional' for s in scene.sources),
            tail_after_last_clear_s=snapshot['virtual_time_s']-successful[-1] if successful else None,
            action_sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest(),
            planner_seconds=report.get('planner_seconds',0) if report else None,
            transit_stops=report.get('transit_stops',0) if report else None,
            route_accepts=report.get('route_accepts',0) if report else None)
        dump(Path(destination)/'sessions'/f'{case["key"]}-{method}.json',
             dict(**session.export(), row=row, algorithm_report=report, completion_audit=certificate))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--phase', choices=['smoke','main','bound_sensitivity'], default='main')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.freeze:
        freeze(); return
    plan = json.loads((ROOT/'plan.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['benchmark_sha256']
    assert verify_sources() == plan['source_manifest_sha256']
    bound = 1.0 if args.phase == 'bound_sensitivity' else plan['bearing_bound_deg']
    if args.phase == 'smoke':
        cases = [dict(key=f'smoke-{i}',group='smoke',scenario=Scenario.generate(4,f'q4-v6-reconstructed-smoke-{i}').as_dict()) for i in range(3)]
    elif args.phase == 'bound_sensitivity': cases = plan['cases'][:20]
    else: cases = plan['cases']
    destination = ROOT/args.phase
    destination.mkdir(exist_ok=True)
    records = destination/'records.jsonl'
    tasks = []
    for i, case in enumerate(cases):
        methods = plan['methods'][i%3:]+plan['methods'][:i%3]
        tasks.append((case, methods, str(destination)))
    started = time.perf_counter()
    count = 0
    with records.open('x') as stream:
        with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(bound,)) as executor:
            futures = [executor.submit(run_case, task) for task in tasks]
            for future in cf.as_completed(futures):
                rows = future.result()
                for row in rows:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush(); count += 1
                errors = [r['error'] for r in rows if r['error']]
                print(f'{count}/{len(cases)} {rows[0]["case_key"]}: '+', '.join(f'{r["method"]}={r["seconds_per_source"]:.2f}' for r in rows)+f' s/source; errors={len(errors)}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    assert verify_sources() == plan['source_manifest_sha256']
    dump(destination/'execution.json',dict(cases=count,runs=count*3,workers=args.workers,
        elapsed_seconds=time.perf_counter()-started,bearing_bound_deg=bound,sources_verified=True))


if __name__ == '__main__': main()

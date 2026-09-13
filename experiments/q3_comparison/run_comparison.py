#!/usr/bin/env python3
"""Run downloaded Q3 policies on paired local cases; never contact a server."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor' / 'B_Q3_optimized_v3'
sys.path.insert(0, str(VENDOR))
import numpy as np
import q3_base as b
import q3_optimized as o
import q3_v3 as v
from stress_v3 import special_sources
sys.path.insert(0, str(HERE / 'vendor' / 'avg200_diagnostics'))
from optical_experiment import ExperimentalOpticalAgent

DEG = math.pi / 180
MODES = ['phased', 'joint', 'v2', 'v3', 'v3_origin20', 'optical',
         'scenario', 'future_cover', 'scenario_future']


def configure(profile):
    # The 0.005-degree rounding allowance belongs in the feasible set, while
    # the physical error remains at most 1 degree. Toy _error uses global b.A.
    b.A = DEG if profile == 'original' else 1.005 * DEG


class AnnexSimulator(b.ToySimulator):
    """Toy world with Annex 1/2 clear semantics and two-decimal bearings.

    Source distribution and spatial error field remain SELF-BUILT assumptions.
    This class is not the official simulator or an HTTP adapter.
    """
    def _error(self, c):
        return super()._error(c) * DEG / b.A

    def detect(self, channel):
        ans = super().detect(channel)
        if ans.kind == 'bearing':
            ans = b.Observation('bearing', (round(math.degrees(ans.bearing), 2) % 360) * DEG)
            if self.record:
                self.log[-1]['bearing_radians'] = ans.bearing
        return ans

    def clear(self, channel):
        if channel not in b.CHANNELS:
            raise ValueError('Invalid channel')
        self.optical += 1
        ok = channel in self._live and b.norm(self._sources[channel].xy - self.pos) <= 20
        if ok:
            self._live.remove(channel)
            self.clears += 1
        else:
            self.failed += 1
        self._log('clear', channel=channel, success=ok)
        return ok


def make_agent(sim, mode):
    public = v.PublicBackend(sim)
    if mode in ('scenario', 'future_cover', 'scenario_future'):
        from new_methods import PlanningAgent
        return PlanningAgent(public, scenario=mode != 'future_cover', future_cover=mode != 'scenario')
    if mode in ('phased', 'joint'):
        return b.Agent(public, mode=mode)
    if mode == 'optical':
        return ExperimentalOpticalAgent(public)
    return v.make_agent(sim, mode)


def initialize(profile):
    configure(profile)
    v.warmup()
    from new_methods import warmup
    warmup()


def work(job):
    profile, kind, seed, mode, noise, repeat, trace = job
    configure(profile)
    sources = special_sources(seed - 93000) if kind == 'special' else b.make_case(seed, kind == 'boundary')
    cls = b.ToySimulator if profile == 'original' else AnnexSimulator
    sim = cls(sources, seed, noise, bool(trace))
    start = time.perf_counter()
    error = ''
    agent = None
    try:
        agent = make_agent(sim, mode)
        agent.run()
        if sim._live or sim.clears != len(sources):
            raise AssertionError('Uncleared sources')
        if mode != 'optical' and sim.failed:
            raise AssertionError('Certified clear failed')
        if agent.unseen or agent.tracks or len(agent.cleared | agent.absent) != 20:
            raise AssertionError('Incomplete terminal certificate')
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter() - start
    row = dict(profile=profile, kind=kind, seed=seed, mode=mode, noise=noise,
               repeat=repeat, true_sources=len(sources), cleared=sim.clears,
               success=not error and not sim._live,
               virtual_seconds=sim.virtual_time, seconds_per_source=sim.virtual_time / len(sources),
               movement_metres=sim.distance, detects=sim.detects, switches=sim.switches,
               optical_attempts=sim.optical, failed_clears=sim.failed,
               local_runtime_seconds=elapsed,
               scenario_decisions=getattr(agent, 'scenario_decisions', 0),
               changed_first_actions=getattr(agent, 'information_stops', 0),
               accepted_future_plans=getattr(agent, 'plan_changes', 0), error=error)
    if trace:
        Path(trace).write_text(json.dumps({'metadata': row, 'actions': sim.log}, ensure_ascii=False, indent=2))
    return row


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=['original', 'annex'], default='annex')
    ap.add_argument('--suite', choices=['random', 'runtime', 'stress'], default='random')
    ap.add_argument('--start', type=int, default=5000)
    ap.add_argument('--cases', type=int, default=400)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    initialize(args.profile)
    warmup_seconds = time.perf_counter() - t0
    jobs = []
    if args.suite == 'stress':
        # The same 350 cases as the downloaded v3 stress test.
        for noise in ['hash', 'smooth', 'plus', 'minus', 'alternating']:
            for kind, seeds in [('boundary', range(91000, 91050)), ('special', range(93000, 93020))]:
                jobs.extend((args.profile, kind, s, m, noise, 0, '') for s in seeds for m in args.modes)
    else:
        for repeat in range(args.repeats if args.suite == 'runtime' else 1):
            for seed in range(args.start, args.start + args.cases):
                # Rotate order across cases; reverse on the second timing pass.
                shift = (seed - args.start) % len(args.modes)
                modes = args.modes[shift:] + args.modes[:shift]
                if repeat % 2:
                    modes = modes[::-1]
                for mode in modes:
                    trace = ''
                    if seed == args.start and args.suite == 'random':
                        trace = str(args.out.with_name(f'{args.out.stem}_{mode}_first_trace.json'))
                    jobs.append((args.profile, 'random', seed, mode, 'hash', repeat, trace))
    rows = []
    run_start = time.perf_counter()
    print(f'{args.profile} {args.suite}: {len(jobs)} runs; warmup {warmup_seconds:.3f}s', flush=True)
    journal = args.out.with_suffix('.jsonl')
    with journal.open('w') as stream:
        def accept(row):
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if row['error']:
                print('FAIL', row['seed'], row['mode'], row['noise'], row['error'], flush=True)
            if len(rows) % 100 == 0 or len(rows) == len(jobs):
                print(f'completed {len(rows)}/{len(jobs)} in {time.perf_counter() - run_start:.1f}s', flush=True)
        if args.suite == 'runtime' or args.workers == 1:
            for job in jobs:
                accept(work(job))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.profile,)) as pool:
                futures = [pool.submit(work, job) for job in jobs]
                for future in as_completed(futures):
                    accept(future.result())
    rows.sort(key=lambda r: (r['kind'], r['seed'], r['noise'], r['repeat'], r['mode']))
    dump_csv(args.out, rows)
    import numba
    metadata = dict(command=sys.argv, python=sys.version, numpy=np.__version__, numba=numba.__version__,
                    platform=platform.platform(), machine=platform.machine(), logical_cpus=os.cpu_count(),
                    workers=1 if args.suite == 'runtime' else args.workers,
                    warmup_seconds=warmup_seconds, total_wall_seconds=time.perf_counter() - t0,
                    scope='SELF-BUILT simulation; includes agent construction; excludes environment creation and HTTP; warmup excluded from each row',
                    profile=args.profile, suite=args.suite, rows=len(rows), failures=sum(not r['success'] for r in rows),
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    args.out.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    for mode in args.modes:
        rr = [r for r in rows if r['mode'] == mode]
        print(mode, json.dumps({k: float(np.mean([x[k] for x in rr])) for k in
              ['seconds_per_source', 'virtual_seconds', 'movement_metres', 'detects', 'switches', 'local_runtime_seconds', 'failed_clears']}), flush=True)
    if metadata['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Fixed incumbent-route screening and known worst-case replay, LOCAL ONLY."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import bootstrap
from bootstrap import SCENARIOS,make_case,RoundedSimulator,PublicDevice
from online import State
import q4_coverage
from incumbent_route import build_incumbent
from compute_fast import ExactComputeCache
from coverage_compute_fast import CoverageComputeCache
from bench import summarize

METHODS=('v4','incumbent','flex_incumbent','cells_flex_incumbent')


def run(method,seed,scenario):
    name,fraction,placement,error=scenario
    sim=RoundedSimulator(make_case(seed,fraction,placement),seed,error,trace=True)
    device=PublicDevice(sim)
    state=State() if method=='v4' else build_incumbent(method)
    timeline=[];failure=None;certificate=None;start=time.perf_counter();cpu=time.process_time()
    try:
        with ExactComputeCache() as compute,CoverageComputeCache() as coverage:
            while state.prepare():
                actions=state.ordered_actions(device)
                normalized=[('site',a[1]) if a[0]=='shift' else a for a in actions]
                legal=[('site',i) for i in sorted(state.remaining)]+[('target',c) for c in sorted(state.pending)]
                if state.visited:
                    assert len(normalized)==len(legal) and set(normalized)==set(legal)
                    if hasattr(state,'incumbent_order'):
                        assert len(state.incumbent_order)==len(legal) and set(state.incumbent_order)==set(legal)
                        assert not any(a[0]=='shift' for a in state.incumbent_order)
                else:assert actions==[('site',0)]
                action=actions[0];state.execute(device,action)
                timeline.append(dict(action_number=len(timeline)+1,action=action,
                    seconds=sim.virtual_seconds,known=len(state.pending)+len(state.cleared),cleared=len(state.cleared)))
            assert sim.clear_count==len(sim._targets)==len(state.cleared)
            if sim.clear_count<16:
                certificate=q4_coverage.certify(state.scanpoints);assert certificate['ok']
            expected=sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count
            assert abs(expected-sim.virtual_seconds)<1e-6
    except Exception:failure=traceback.format_exc()
    report=state.report()
    return dict(method=method,seed=seed,scenario=name,success=failure is None,exception=failure,
        **sim.summary(),seconds_per_source=sim.virtual_seconds/len(sim._targets),
        wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        tail_search_seconds=sim.virtual_seconds-sim.last_clear if sim.last_clear is not None else None,
        coverage_verified=(certificate or {}).get('ok'),report=report,timeline=timeline,
        trace_sha256=hashlib.sha256(json.dumps(sim.trace,separators=(',',':')).encode()).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worst-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]/'output/q4_speedup/incumbent_dev40'
    root.mkdir(parents=True,exist_ok=True)
    worst=[run(m,663020263,('mixed25',.25,'uniform','hash')) for m in METHODS]
    (root/'known_worst_replay.json').write_text(json.dumps(dict(kind='Known development diagnostic, excluded from fresh evaluation',rows=worst),indent=2))
    for row in worst:
        print('WORST',row['method'],row['virtual_seconds'],row['report']['visited_stations'],row['success'],flush=True)
        if not row['success']:print(row['exception'],flush=True)
    if args.worst_only:return
    protocol=dict(kind='LOCAL ONLY fixed development screening; not heldout',seed_base=441920260,
                  cases_per_group=4,groups=SCENARIOS,methods=METHODS,
                  cache='ExactComputeCache + CoverageComputeCache, all strategies preimported',
                  constants=dict(physical_deg=1.,reported_deg=.01,geometric_deg=1.005),
                  candidates_sha256=hashlib.sha256((Path(__file__).parent/'incumbent_route.py').read_bytes()).hexdigest())
    (root/'manifest.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with (root/'runs.jsonl').open('w') as logfile:
        for gi,scenario in enumerate(SCENARIOS):
            for index in range(4):
                seed=441920260+100000*gi+index;k=(gi+index)%len(METHODS)
                for method in METHODS[k:]+METHODS[:k]:
                    row=run(method,seed,scenario);rows.append(row)
                    logfile.write(json.dumps(row)+'\n');logfile.flush()
                    if not row['success']:print(row['exception'],flush=True)
            print(scenario[0],'completed',len(rows),flush=True)
    result=dict(protocol=protocol,overall=summarize(rows),
                groups={s[0]:summarize([r for r in rows if r['scenario']==s[0]]) for s in SCENARIOS})
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['overall'],indent=2),flush=True)

if __name__=='__main__':main()

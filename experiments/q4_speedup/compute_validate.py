"""One frozen 100-case exactness/performance comparison on a shared machine."""
from __future__ import annotations
import gzip,hashlib,json,math,random,statistics,time
from pathlib import Path
import bootstrap
from bootstrap import b,SCENARIOS,make_case,RoundedSimulator,PublicDevice
from online import State,finish
from q4_coverage import certify
import q4_route_cached
import q4_v3_local
import online
from compute_fast import ExactComputeCache,_circle_exact,_clip_halfplane_exact,_route_exact


def unit_checks():
    rng=random.Random(72283201)
    route_cases=clip_cases=circle_cases=0
    for n in range(1,32):
        for k in range(8):
            pts=[(rng.randint(-2000,2000),rng.randint(-2000,2000)) for _ in range(n)]
            remaining=[i for i in range(n) if rng.random()<.85]
            if not remaining:continue
            pos=(rng.randint(-2000,2000),rng.randint(-2000,2000))
            assert q4_route_cached.multi_route(pos,remaining,pts,8)==_route_exact(pos,remaining,pts,8,b.route_order)
            route_cases+=1
    for k in range(10000):
        poly=[(rng.uniform(-1900,1900),rng.uniform(-1900,1900)) for _ in range(rng.randrange(0,20))]
        normal=b.unit(rng.random()*2*math.pi);bound=rng.uniform(-2000,2000)
        assert b.clip_halfplane(poly,normal,bound)==_clip_halfplane_exact(poly,normal,bound)
        clip_cases+=1
    for k in range(1200):
        scale=10**rng.uniform(-5,5)
        poly=[(rng.uniform(-scale,scale),rng.uniform(-scale,scale)) for _ in range(rng.randrange(1,16))]
        assert b.enclosing_circle(poly)==_circle_exact(poly)
        circle_cases+=1
    original=(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_v3_local.optical_cover)
    # A mutable optical result cannot corrupt a cached result, bounded eviction
    # works, clear actually empties memory, and exceptions restore all aliases.
    try:
        with ExactComputeCache(polygon_limit=2,gain_limit=2,route_limit=2,optical_limit=2) as cache:
            poly=[(0.,0.),(50.,0.),(50.,1.),(0.,1.)]
            first=q4_v3_local.optical_cover(poly);frozen=first[:];first.reverse();first.append((999.,999.))
            assert q4_v3_local.optical_cover(poly)==frozen
            for k in range(5):b.enclosing_circle([(0.,0.),(float(k+1),0.),(0.,1.)])
            assert cache.stats()['circle']['currsize']==2
            cache.clear();assert all(x['currsize']==0 for x in cache.stats().values())
            try:
                with ExactComputeCache():pass
            except RuntimeError:pass
            else:raise AssertionError('Nested contexts must fail')
            raise LookupError('restoration test')
    except LookupError:pass
    assert original==(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_v3_local.optical_cover)
    return dict(route_cases=route_cases,clip_cases=clip_cases,circle_cases=circle_cases,
                mutable_cache_test=True,bounded_eviction=True,clear_test=True,
                exception_restoration=True,nested_context_rejection=True)


def main():
    root=Path(__file__).resolve().parents[3]/'output/q4_speedup/schedule_dev/compute100';root.mkdir(parents=True,exist_ok=True)
    checks=unit_checks();State()
    manifest=dict(kind='Exact execution equivalence and CPU-only optimization; not virtual task improvement',
                  shared_machine=True,timing='Rotating paired serial runs; process_time and perf_counter; one comparison per case; trace recording enabled; cache enter/exit included; simulator setup and coverage audit excluded',
                  seed_base=230400000,cases_per_group=10,groups=SCENARIOS,
                  physical_error_deg=1.,display_round_deg=.01,geometric_bound_deg=1.005,
                  cache_limits=dict(polygon=2048,gain=512,route=128,optical=512),
                  acceleration_sha256=hashlib.sha256((Path(__file__).parent/'compute_fast.py').read_bytes()).hexdigest())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];cache_stats=[];trace_count=0
    identities=(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_v3_local.optical_cover)
    with (root/'cases.jsonl').open('w') as log,gzip.open(root/'identical_action_traces.jsonl.gz','wt') as trace_log:
        for gi,scenario in enumerate(SCENARIOS):
            name,fraction,placement,error=scenario
            for i in range(10):
                seed=manifest['seed_base']+gi*100000+i;pair={}
                for accelerated in ([False,True] if (gi+i)%2==0 else [True,False]):
                    sim=RoundedSimulator(make_case(seed,fraction,placement),seed,error,trace=True)
                    begin_cpu=time.process_time();begin_wall=time.perf_counter()
                    if accelerated:
                        with ExactComputeCache() as cache:report=finish(State(),PublicDevice(sim))
                    else:report=finish(State(),PublicDevice(sim))
                    wall=time.perf_counter()-begin_wall;cpu=time.process_time()-begin_cpu
                    stats=cache.stats() if accelerated else None
                    assert identities==(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_v3_local.optical_cover)
                    pair[accelerated]=dict(summary=sim.summary(),report=report,trace=sim.trace,cpu=cpu,wall=wall,cache=stats)
                base,fast=pair[False],pair[True]
                assert base['summary']==fast['summary'] and base['report']==fast['report'] and base['trace']==fast['trace'],seed
                assert base['summary']['targets']==base['summary']['cleared']
                certificate=certify(base['report']['actual_scanpoints'])['ok'] if base['summary']['cleared']<16 else True
                assert certificate
                trace_count+=len(base['trace'])
                trace_text=json.dumps(base['trace'],separators=(',',':'))
                row=dict(seed=seed,scenario=name,actions_equal=True,summary_equal=True,report_equal=True,
                         action_count=len(base['trace']),trace_sha256=hashlib.sha256(trace_text.encode()).hexdigest(),
                         cleared=base['summary']['cleared'],virtual_seconds=base['summary']['virtual_seconds'],
                         baseline_cpu=base['cpu'],accelerated_cpu=fast['cpu'],baseline_wall=base['wall'],accelerated_wall=fast['wall'],cache=fast['cache'],coverage_verified=certificate)
                rows.append(row);cache_stats.append(fast['cache']);log.write(json.dumps(row)+'\n');log.flush()
                trace_log.write(json.dumps(dict(seed=seed,trace=base['trace']))+'\n')
            print(name,'completed',len(rows),flush=True)
    base_cpu=sum(r['baseline_cpu'] for r in rows);fast_cpu=sum(r['accelerated_cpu'] for r in rows)
    base_wall=sum(r['baseline_wall'] for r in rows);fast_wall=sum(r['accelerated_wall'] for r in rows)
    differences=[r['baseline_cpu']-r['accelerated_cpu'] for r in rows]
    standard_error=statistics.stdev(differences)/math.sqrt(len(rows));mean=statistics.mean(differences)
    result=dict(manifest=manifest,unit_checks=checks,cases=len(rows),all_actions_identical=True,
                all_reports_identical=True,all_summary_identical=True,compared_actions=trace_count,
                cleared_sources=sum(r['cleared'] for r in rows),all_clear_and_certified=True,
                baseline_cpu_s=base_cpu,accelerated_cpu_s=fast_cpu,cpu_reduction=1-fast_cpu/base_cpu,cpu_speedup=base_cpu/fast_cpu,
                baseline_wall_s=base_wall,accelerated_wall_s=fast_wall,wall_reduction=1-fast_wall/base_wall,
                mean_cpu_saved_s=mean,paired_normal95_cpu_saved_s=[mean-1.96*standard_error,mean+1.96*standard_error],
                cached={name:dict(hits=sum(c[name]['hits'] for c in cache_stats),misses=sum(c[name]['misses'] for c in cache_stats),
                                  peak_case_entries=max(c[name]['currsize'] for c in cache_stats),limit=cache_stats[0][name]['maxsize']) for name in cache_stats[0]},
                note='CPU result is one paired comparison on a nonexclusive machine. It does not imply the official HTTP run is faster by the same percentage. No virtual task-time reduction: every action and feedback is exactly equal.')
    (root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

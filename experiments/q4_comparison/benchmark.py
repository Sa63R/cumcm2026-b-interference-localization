"""Serial paired LOCAL_ONLY comparison. Results are incremental JSONL."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, signal, statistics, sys, time, traceback
from dataclasses import asdict
from pathlib import Path
import common
import q4_baseline as b
from q4_v4_solver import solve_v4
from q4_coverage import certify
from online import solve_resumable, State
from planner import Planner, PlanningConfig
from run_v3_validation import SCENARIOS

METHODS=['v4','analytic','rollout','shared','dynamic']
class PublicDevice:
    def __init__(self,d):self.__d=d
    @property
    def position(self):return self.__d.position
    @property
    def channel(self):return self.__d.channel
    def move(self,p):return self.__d.move(p)
    def detect(self,c):return self.__d.detect(c)
    def clear(self,c):return self.__d.clear(c)
class TimedSimulator(b.LocalSimulator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.last_success_time=None
    def clear(self,c):
        ok=super().clear(c)
        if ok:self.last_success_time=self.virtual_seconds
        return ok

def q95(xs):
    xs=sorted(xs);pos=.95*(len(xs)-1);a=int(pos)
    return xs[a]+(xs[min(a+1,len(xs)-1)]-xs[a])*(pos-a)
def summarize(rows):
    out={};base={r['seed']:r for r in rows if r['method']=='v4'}
    for m in METHODS:
        rs=[r for r in rows if r['method']==m]
        if not rs:continue
        ok=[r for r in rs if r['success']]
        out[m]={'cases':len(rs),'successes':len(ok),'failures':len(rs)-len(ok)}
        if not ok:continue
        metrics=['seconds_per_source','virtual_seconds','wall_seconds','distance_m','detections','switches','failed_clears','tail_search_seconds']
        for key in metrics: out[m]['mean_'+key]=statistics.mean(r[key] for r in ok)
        out[m]['p95_seconds_per_source']=q95([r['seconds_per_source'] for r in ok])
        out[m]['p95_wall_seconds']=q95([r['wall_seconds'] for r in ok])
        out[m]['max_wall_seconds']=max(r['wall_seconds'] for r in ok)
        if m!='v4':
            paired=[r for r in ok if r['seed'] in base and base[r['seed']]['success']]
            if paired:
                saved=[base[r['seed']]['seconds_per_source']-r['seconds_per_source'] for r in paired]
                relative=[1-r['virtual_seconds']/base[r['seed']]['virtual_seconds'] for r in paired]
                avg=statistics.mean(saved);se=statistics.stdev(saved)/math.sqrt(len(saved)) if len(saved)>1 else None
                out[m]['paired']={'n':len(paired),'mean_saved_seconds_per_source':avg,
                      'normal95_saved_seconds_per_source':[avg-1.96*se,avg+1.96*se] if se is not None else None,
                      'reduction_of_paired_mean':avg/statistics.mean(base[r['seed']]['seconds_per_source'] for r in paired),
                      'faster':sum(x>1e-6 for x in saved),'slower':sum(x<-1e-6 for x in saved),'tied':sum(abs(x)<=1e-6 for x in saved),
                      'worst_relative_regression':-min(relative)}
        if m in ['rollout','shared','dynamic']:
            out[m]['planning']={k:sum(r.get('planning',{}).get(k,0) for r in rs) for k in ['attempts','accepted','timeouts','posterior_fallbacks','invalid_rollouts','rollout_runs','shared_candidates','shift_candidates','coverage_checks','coverage_passed']}
            out[m]['selected_shared']=sum(r.get('extra_actions',0) for r in rs)
            out[m]['selected_shift']=sum(r.get('shifted_sites',0) for r in rs)
    return out

class CaseBudgetExceeded(Exception): pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=int,default=2)
    p.add_argument('--seed-base',type=int,default=107000000);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--scenarios',type=int,default=4);p.add_argument('--decisions',type=int,default=4)
    p.add_argument('--decision-budget',type=float,default=3);p.add_argument('--case-timeout',type=float,default=1200)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cfg=PlanningConfig(coarse_scenarios=args.scenarios,verify_scenarios=args.scenarios,max_decisions=args.decisions,seconds_per_decision=args.decision_budget)
    manifest=dict(warning='LOCAL_ONLY, not official simulator results',python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),seed_base=args.seed_base,cases_per_group=args.cases,groups=SCENARIOS,planning=asdict(cfg),method_order='rotating serial',prior='uniform area/radius/orientation; count 10..16; both types conditioned; independent bounded location-specific errors',hashes={str(f.relative_to(common.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in common.ROOT.rglob('*.py')})
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    def alarm(*_):raise CaseBudgetExceeded('Per-case wall time exceeded')
    signal.signal(signal.SIGALRM,alarm)
    rows=[];start_all=time.perf_counter()
    # Count one-time certificate startup separately. Every method uses it.
    t=time.perf_counter();State();manifest['coverage_startup_seconds']=time.perf_counter()-t
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    with (args.out/'runs.jsonl').open('w') as log:
        for gi,(scenario,fraction,placement,error) in enumerate(SCENARIOS):
            for i in range(args.cases):
                seed=args.seed_base+gi*100000+i
                rotation=(i+gi)%len(args.methods)
                methods=args.methods[rotation:]+args.methods[:rotation]
                for method in methods:
                    sim=TimedSimulator(b.make_case(seed,fraction,placement),seed,error,trace=i==0)
                    start=time.perf_counter();report={};exception=None
                    try:
                        signal.setitimer(signal.ITIMER_REAL,args.case_timeout)
                        if method=='v4':report=solve_v4(PublicDevice(sim))
                        else:
                            planner=None
                            if method!='analytic':
                                conf=copy.deepcopy(cfg);conf.shared=method in ['shared','dynamic'];conf.dynamic=method=='dynamic'
                                planner=Planner(seed+720000000,conf)
                            report=solve_resumable(PublicDevice(sim),analytic=True,planner=planner)
                    except Exception as e:exception=repr(e);traceback.print_exc()
                    finally:signal.setitimer(signal.ITIMER_REAL,0)
                    wall=time.perf_counter()-start
                    coverage_verified=None
                    if report.get('shifted_sites',0)>0 and report.get('stop_certificate')=='coverage_complete':
                        coverage_verified=certify(report['actual_scanpoints'])['ok']
                        if not coverage_verified: exception='Final actual coverage certificate failed'
                    row=dict(seed=seed,scenario=scenario,method=method,**sim.summary(),wall_seconds=wall,
                             seconds_per_source=sim.virtual_seconds/len(sim._targets),
                             tail_search_seconds=sim.virtual_seconds-sim.last_success_time if sim.clear_count==len(sim._targets) else None,
                             success=exception is None and sim.clear_count==len(sim._targets),exception=exception,coverage_verified=coverage_verified,
                             visited_stations=len(report.get('visited_stations',[])),extra_actions=report.get('extra_actions',0),shifted_sites=report.get('shifted_sites',0),planning=report.get('planning',{}),stop_certificate=report.get('stop_certificate'))
                    expected=sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count
                    if abs(expected-sim.virtual_seconds)>1e-6:raise AssertionError('Time accounting mismatch')
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    if i==0:(args.out/f'trace_{method}_{seed}.json').write_text(json.dumps(dict(row=row,report=report,actions=sim.trace)))
                    print(f'{len(rows):4d} {scenario:20s} {seed} {method:8s} {row["seconds_per_source"]:8.2f} s/source {wall:6.3f} wall success={row["success"]}',flush=True)
                (args.out/'summary.json').write_text(json.dumps(dict(overall=summarize(rows),groups={name:summarize([r for r in rows if r['scenario']==name]) for name,*_ in SCENARIOS},elapsed=time.perf_counter()-start_all),indent=2))
    print(json.dumps(summarize(rows),indent=2),flush=True)
if __name__=='__main__':main()

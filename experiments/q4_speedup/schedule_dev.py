"""New development cases only, all policies see the same observations API."""
import argparse,copy,json,math,statistics,time,traceback
from pathlib import Path
from schedule_candidate import CandidateState,b
from online import State,finish
from benchmark import PublicDevice
from run_v3_validation import SCENARIOS


class RoundedSimulator(b.LocalSimulator):
    def _error(self,c):
        return super()._error(c)/1.005
    def detect(self,c):
        obs=super().detect(c)
        if obs.status=='bearing':return b.Observation('bearing',math.radians(round(math.degrees(obs.theta),2)))
        return obs


def summarize(rows):
    methods=sorted(set(r['method'] for r in rows)); bases={r['seed']:r for r in rows if r['method']=='baseline'}
    out={}
    for m in methods:
        rr=[r for r in rows if r['method']==m]
        ok=[r for r in rr if r['success']]
        diffs=[bases[r['seed']]['seconds_per_source']-r['seconds_per_source'] for r in ok]
        ci=1.96*statistics.stdev(diffs)/math.sqrt(len(diffs)) if len(diffs)>1 else 0
        mean=statistics.mean(diffs) if diffs else 0
        out[m]=dict(n=len(rr),ok=len(ok),mean_seconds_per_source=statistics.mean(r['seconds_per_source'] for r in ok) if ok else None,
                    saved_seconds_per_source=mean,normal95_saved=[mean-ci,mean+ci],
                    percent_reduction=100*mean/statistics.mean(bases[r['seed']]['seconds_per_source'] for r in ok) if ok else 0,
                    faster=sum(d>1e-6 for d in diffs),slower=sum(d<-1e-6 for d in diffs),
                    wall_seconds=statistics.mean(r['wall_seconds'] for r in rr))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=int,default=4);p.add_argument('--seed-base',type=int,default=214700000)
    p.add_argument('--methods',nargs='+',default=['baseline','entry','entry_half','entry_quarter','entry_threequarter','service_route','continue','nearest','entry_nearest','stations_first','targets_first'])
    p.add_argument('--name',default='screen24'); args=p.parse_args()
    root=Path(__file__).resolve().parents[3]/'output/q4_speedup/schedule_dev'/args.name;root.mkdir(parents=True,exist_ok=True)
    b.DELTA=math.radians(1.005);b.COS_D=math.cos(b.DELTA);b.TAN_D=math.tan(b.DELTA)
    State();rows=[]
    (root/'manifest.json').write_text(json.dumps(dict(warning='LOCAL SYNTHETIC DEVELOPMENT ONLY; not held-out or official',args=vars(args),geometry_degrees=1.005,physical_error_degrees=1,quantization_degrees=.01,groups=SCENARIOS),indent=2))
    with (root/'runs.jsonl').open('w') as log:
        for gi,(name,frac,placement,error) in enumerate(SCENARIOS):
            for i in range(args.cases):
                seed=args.seed_base+gi*100000+i
                for method in args.methods:
                    targets=b.make_case(seed,frac,placement)
                    sim=RoundedSimulator(targets,seed,error)
                    state=CandidateState(strategy=method);start=time.perf_counter();exception=None;report={}
                    try:report=finish(state,PublicDevice(sim))
                    except Exception as e:exception=repr(e);traceback.print_exc()
                    wall=time.perf_counter()-start
                    row=dict(seed=seed,scenario=name,method=method,success=not exception and sim.clear_count==len(targets),exception=exception,
                             seconds_per_source=sim.virtual_seconds/len(targets),wall_seconds=wall,**sim.summary(),report=report)
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                print(name,seed,json.dumps(summarize(rows)),flush=True)
                (root/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
if __name__=='__main__':main()

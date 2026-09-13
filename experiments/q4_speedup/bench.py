"""Paired development or frozen holdout; synthetic, with official bearing precision."""
import bootstrap
import argparse,copy,hashlib,json,math,platform,statistics,time,traceback
from pathlib import Path
from online import State
from q4_coverage import certify
from bootstrap import SCENARIOS,make_case,RoundedSimulator,PublicDevice

def build(method):
    if method=='v4':return State()
    if method=='analytic':return State(analytic=True)
    from strategy import build_candidate
    return build_candidate(method)

def run_case(method,seed,scenario):
    name,fraction,placement,error=scenario
    sim=RoundedSimulator(make_case(seed,fraction,placement),seed,error)
    device=PublicDevice(sim)
    start=time.perf_counter();state=build(method);exc=None;certificate=None
    try:
        while state.prepare():
            ordered=state.ordered_actions(device)
            planner=getattr(state,'planner',None)
            action=ordered[0] if planner is None else planner.choose(state,device,ordered)
            state.execute(device,action)
        assert len(state.cleared)==len(sim._targets)==sim.clear_count
        if sim.clear_count<16:
            certificate=certify(state.scanpoints)
            assert certificate['ok'],certificate
        expected=sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count
        assert abs(expected-sim.virtual_seconds)<1e-6
    except Exception:
        exc=traceback.format_exc()
    return dict(method=method,seed=seed,scenario=name,success=exc is None,exception=exc,
                **sim.summary(),wall_seconds=time.perf_counter()-start,
                seconds_per_source=sim.virtual_seconds/len(sim._targets),
                tail_search_seconds=sim.virtual_seconds-sim.last_clear if sim.last_clear is not None else None,
                coverage_verified=(certificate or {}).get('ok'),report=state.report(),
                planning=getattr(getattr(state,'planner',None),'stats',None))

def summarize(rows):
    base={r['seed']:r for r in rows if r['method']=='v4'}
    out={}
    for method in dict.fromkeys(r['method'] for r in rows):
        rs=[r for r in rows if r['method']==method]
        z=dict(cases=len(rs),successes=sum(r['success'] for r in rs))
        for key in ['seconds_per_source','virtual_seconds','wall_seconds','distance_m','detections','failed_clears','tail_search_seconds']:
            xs=[r[key] for r in rs if r[key] is not None]
            z['mean_'+key]=statistics.mean(xs)
        paired=[r for r in rs if r['seed'] in base]
        if paired:
            ds=[base[r['seed']]['seconds_per_source']-r['seconds_per_source'] for r in paired]
            rel=[1-r['virtual_seconds']/base[r['seed']]['virtual_seconds'] for r in paired]
            se=statistics.stdev(ds)/math.sqrt(len(ds)) if len(ds)>1 else 0.
            z.update(saved_seconds_per_source=statistics.mean(ds),
                     reduction_of_mean=statistics.mean(ds)/statistics.mean(base[r['seed']]['seconds_per_source'] for r in paired),
                     exploratory_normal95=[statistics.mean(ds)-1.96*se,statistics.mean(ds)+1.96*se],
                     faster=sum(d>1e-6 for d in ds),slower=sum(d< -1e-6 for d in ds),
                     worst_relative_regression=-min(rel))
        out[method]=z
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--cases',type=int,default=5)
    p.add_argument('--seed-base',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--groups',nargs='*')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    scenarios=[s for s in SCENARIOS if not args.groups or s[0] in args.groups]
    manifest=dict(kind='LOCAL_ONLY paired synthetic; physical error +/-1deg, reported 0.01deg, geometric bound1.005deg',
                  python=platform.python_version(),methods=args.methods,seed_base=args.seed_base,
                  cases_per_group=args.cases,groups=scenarios,method_order='rotating serial',
                  source_hashes={str(f.relative_to(bootstrap.HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in bootstrap.HERE.glob('*.py')})
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    rows=[]
    with (args.out/'runs.jsonl').open('w') as log:
        for gi,s in enumerate(SCENARIOS):
            if s not in scenarios:continue
            for i in range(args.cases):
                seed=args.seed_base+100000*gi+i
                k=(gi+i)%len(args.methods)
                for method in args.methods[k:]+args.methods[:k]:
                    row=run_case(method,seed,s);rows.append(row)
                    log.write(json.dumps(row)+'\n');log.flush()
                    print(f"{len(rows)} {s[0]} {seed} {method} {row['seconds_per_source']:.2f} success={row['success']}",flush=True)
                    if not row['success']:print(row['exception'],flush=True)
            result=dict(overall=summarize(rows),groups={s[0]:summarize([r for r in rows if r['scenario']==s[0]]) for s in scenarios if any(r['scenario']==s[0] for r in rows)})
            (args.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(summarize(rows),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

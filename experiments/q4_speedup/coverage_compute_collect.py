"""Collect exact certificate arguments from unmodified cells_flex actions."""
import bootstrap
import cProfile,json,pstats,time
from pathlib import Path
from bootstrap import SCENARIOS,make_case,RoundedSimulator,PublicDevice
from strategy import build_candidate
from online import State,finish
import coverage_candidate
import q4_coverage

ROOT=Path(__file__).resolve().parents[3]/'output/q4_speedup/schedule_dev/coverage_compute100'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    State();original=coverage_candidate.certify;recorded=[];case_reports=[]
    def record(sites,max_depth=22,keep_leaves=False):
        answer=original(sites,max_depth=max_depth,keep_leaves=keep_leaves)
        recorded.append(dict(sites=sites,max_depth=max_depth,keep_leaves=keep_leaves,
                             original={k:v for k,v in answer.items() if k!='leaves'}))
        return answer
    coverage_candidate.certify=record
    try:
        for gi in [0,4,5]:
            name,fraction,place,error=SCENARIOS[gi];seed=244900000+gi*100000
            sim=RoundedSimulator(make_case(seed,fraction,place),seed,error)
            state=build_candidate('cells_flex');report=finish(state,PublicDevice(sim))
            assert sim.clear_count==len(sim._targets)
            case_reports.append(dict(seed=seed,scenario=name,cleared=sim.clear_count,virtual_seconds=sim.virtual_seconds,
                                     recorded_checks=report['coverage_checks']))
    finally:coverage_candidate.certify=original
    # Evenly select actual trial certificates, retaining expensive successes
    # and failures rather than only the first cheap rejections.
    selected=[dict(recorded[round(i*(len(recorded)-1)/89)],source='actual_cells_flex_trial') for i in range(90)]
    sites=State().sites
    extras=[(sites,22),(sites,0),(sites,2),(sites,8),([],22),([(0.,0.)],22),
            ([(0.,0.),(1000.,0.),(-1000.,0.)],22),(sites+sites[:4],22),
            ([(-y,x) for x,y in sites],22),([(x*.999,y*.999) for x,y in sites],18)]
    selected += [dict(sites=ss,max_depth=depth,keep_leaves=False,source='edge_case') for ss,depth in extras]
    for i,item in enumerate(selected):item['keep_leaves']=i%4==0
    (ROOT/'inputs.json').write_text(json.dumps(dict(cases=case_reports,collected=len(recorded),inputs=selected),indent=2))
    # Profile a small representative set; final paired performance comparison
    # will be separate and unprofiled, with each input evaluated once per side.
    interesting=sorted(recorded,key=lambda r:r['original']['wall'],reverse=True)[:8]
    profiler=cProfile.Profile();profiler.enable()
    for item in interesting:original(item['sites'],max_depth=item['max_depth'])
    profiler.disable()
    profiler.dump_stats(str(ROOT/'baseline_profile.prof'))
    with (ROOT/'baseline_profile.txt').open('w') as f:pstats.Stats(profiler,stream=f).strip_dirs().sort_stats('cumtime').print_stats(35)
    print(json.dumps(dict(collected=len(recorded),cases=case_reports),indent=2));print((ROOT/'baseline_profile.txt').read_text())
if __name__=='__main__':main()

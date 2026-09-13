"""LOCAL ONLY paired tuning, never official simulator evidence."""
from __future__ import annotations
import copy, itertools, json, math, random, statistics, time
from pathlib import Path
from local_candidate import CandidateState, b
from online import State, finish
from benchmark import PublicDevice

b.DELTA=math.radians(1.005)
b.COS_D=math.cos(b.DELTA)
b.TAN_D=math.tan(b.DELTA)

class RoundedSimulator(b.LocalSimulator):
    def _error(self, c):
        return super()._error(c) / 1.005
    def detect(self,c):
        obs=super().detect(c)
        if obs.status=='bearing':
            return b.Observation('bearing',math.radians(round(math.degrees(obs.theta),2)))
        return obs

GROUPS=[(.25,'uniform','hash'),(.75,'uniform','hash'),(1.,'uniform','hash'),(1.,'outward_boundary','hash'),(.5,'uniform','plus'),(.5,'uniform','minus')]
def cases(n=5,base=130920260):
    for gi,(frac,placement,error) in enumerate(GROUPS):
        for i in range(n):
            seed=base+gi*10000+i
            targets=b.make_case(seed,frac,placement)
            if frac==1:
                rng=random.Random(seed+731)
                for t in targets:
                    t.direction=b.unit(math.atan2(t.position[1],t.position[0]) if placement=='outward_boundary' else rng.random()*2*math.pi)
            yield dict(seed=seed,group=gi,targets=targets,error=error)

def run(parameters,cases_list,state_class=CandidateState,extra=None):
    rows=[]
    for case in cases_list:
        sim=RoundedSimulator(copy.deepcopy(case['targets']),case['seed'],case['error'])
        state=state_class(local_parameters=parameters,**(extra or {})) if state_class is CandidateState else state_class()
        start=time.perf_counter()
        report=finish(state,PublicDevice(sim))
        assert sim.clear_count==len(case['targets'])
        assert not any('fallback' in x.get('certificate','') for x in report['localizations'])
        rows.append(dict(seed=case['seed'],group=case['group'],**sim.summary(),wall_seconds=time.perf_counter()-start))
    return rows

def summary(base,rows):
    saved=[(a['virtual_seconds']-z['virtual_seconds'])/a['targets'] for a,z in zip(base,rows)]
    return dict(n=len(rows),mean_seconds_per_source=statistics.mean(r['virtual_seconds']/r['targets'] for r in rows),saved_seconds_per_source=statistics.mean(saved),faster=sum(v>1e-7 for v in saved),slower=sum(v<-1e-7 for v in saved),misses=sum(r['failed_clears'] for r in rows),mean_wall=statistics.mean(r['wall_seconds'] for r in rows),group_saved=[statistics.mean(x for x,r in zip(saved,rows) if r['group']==g) for g in range(6)])

def main():
    cases_list=list(cases());base=run({},cases_list,state_class=State)
    candidates=[{}]
    candidates += [dict(optical_cover_limit=n) for n in [0,3,6,8,16,20,30,40]]
    candidates += [dict(axial=a,lateral=h) for a in [.4,.5,.6,.7,.8] for h in [.025,.05,.10]]
    candidates += [dict(range_fraction=r,adaptive_lateral=h) for r in [.15,.35,.5,.7] for h in [.025,.05,.1]]
    records=[]
    for cfg in candidates:
        rows=base if not cfg else run(cfg,cases_list)
        rec=dict(parameters=cfg,summary=summary(base,rows),rows=rows)
        records.append(rec)
        print(json.dumps(dict(parameters=cfg,**rec['summary'])),flush=True)
    out=Path(__file__).resolve().parent/'local_dev_tuning.json'
    out.write_text(json.dumps(dict(warning='LOCAL ONLY development tuning, not heldout',seed_base=130920260,base=base,records=records),ensure_ascii=False,indent=2))
    print('BEST',json.dumps(sorted(records,key=lambda x:x['summary']['mean_seconds_per_source'])[:3],default=str)[:1500])
if __name__=='__main__':main()

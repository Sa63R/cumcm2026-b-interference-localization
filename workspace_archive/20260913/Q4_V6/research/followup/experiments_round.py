from __future__ import annotations
import math,time,json,random,sys,statistics,dataclasses,concurrent.futures as cf
import q4_baseline as b
from q4_state import Engine,Action
from q4_v4_solver import V4Config
from q4_route_cached import multi_route
from validate_v5 import SCENARIOS,DeviceOnly,AuditSimulator,make_targets

class RotationEngine(Engine):
    def __init__(self,device,cfg,angles=16,steps=1,objective='route'):
        super().__init__(device,cfg);self.angles=angles;self.steps=steps;self.rotated=False;self.objective=objective
    def rotate(self):
        s=self.state
        if self.rotated or not s.visited:return
        self.rotated=True
        old=s.sites[:];targets=[tr.center() for tr in s.pending.values()]
        if not targets:return
        best=float('inf');chosen=old
        for k in range(self.angles):
            ang=math.pi/2*k/self.angles;c=math.cos(ang);z=math.sin(ang)
            sites=[(c*x-z*y,z*x+c*y) for x,y in old]
            pts=sites[1:]+targets
            route=multi_route(self.device.position,list(range(len(pts))),pts,4)
            score=sum(b.dist(a,v) for a,v in zip([self.device.position]+[pts[j] for j in route[:-1]],[pts[j] for j in route]))
            if score<best:chosen=sites;best=score
        s.sites=chosen
    def ordered_actions(self):
        self.rotate();return super().ordered_actions()

def run(task):
    g,i,mode,seedbase=task
    name,fraction,placement,error=SCENARIOS[g];seed=seedbase+100000*g+i
    sim=AuditSimulator(make_targets(seed,fraction,placement),seed,error)
    t=time.perf_counter()
    try:
        cfg=V4Config()
        if mode=='v4':engine=Engine(DeviceOnly(sim),cfg)
        elif mode.startswith('rotate'):engine=RotationEngine(DeviceOnly(sim),cfg,int(mode[6:]))
        else:raise ValueError(mode)
        r=engine.run_base()
        assert len(sim._targets)==sim.clear_count
        return dict(mode=mode,seed=seed,scenario=name,**sim.summary(),wall=time.perf_counter()-t)
    except Exception:
        import traceback
        return dict(mode=mode,seed=seed,error=traceback.format_exc())

if __name__=='__main__':
    modes=sys.argv[1:] or ['v4','rotate8','rotate24','rotate64']
    tasks=[(g,i,m,147100000) for g in range(6) for i in range(6) for m in modes]
    with cf.ProcessPoolExecutor(max_workers=6) as ex,open('rotation_development.jsonl','w') as f:
        for r in ex.map(run,tasks):f.write(json.dumps(r)+'\n');f.flush()
    rows=[json.loads(x) for x in open('rotation_development.jsonl')]
    for m in modes:
        rr=[r for r in rows if r['mode']==m];print(m,len(rr),sum('error' in r for r in rr),statistics.mean(r['seconds_per_cleared'] for r in rr if 'error' not in r),flush=True)

"""Paired frozen-configuration evaluation. Local synthetic cases, not official.

Only this harness may inspect hidden source metadata to check completion. The
solver receives an object exposing just the documented movement/feedback API.
"""
from __future__ import annotations
import argparse,concurrent.futures as cf,dataclasses,hashlib,json,math,pathlib,random,time
import q4_baseline as b
from q4_v4_solver import solve_v4
from q4_v5 import solve_v5,default_config,solve_v5_global,global_config
from q4_local_rollout import solve_local_rollout

SCENARIOS=[('mixed25',.25,'uniform','hash'),('mixed50',.5,'uniform','hash'),
 ('mixed75',.75,'uniform','hash'),('boundary',.75,'outward_boundary','hash'),
 ('plus',.5,'uniform','plus'),('minus',.5,'uniform','minus'),
 ('cluster',.5,'cluster','hash'),('all_radius_1000',.5,'r1000','hash'),
 ('smooth_error',.5,'uniform','smooth')]

class DeviceOnly:
    __slots__=('__sim',)
    def __init__(self,sim):object.__setattr__(self,'_DeviceOnly__sim',sim)
    @property
    def position(self):return object.__getattribute__(self,'_DeviceOnly__sim').position
    @property
    def channel(self):return object.__getattribute__(self,'_DeviceOnly__sim').channel
    def move(self,p):return object.__getattribute__(self,'_DeviceOnly__sim').move(p)
    def detect(self,c):return object.__getattribute__(self,'_DeviceOnly__sim').detect(c)
    def clear(self,c):return object.__getattribute__(self,'_DeviceOnly__sim').clear(c)
    def __getattribute__(self,name):
        if name in ('position','channel','move','detect','clear'):return object.__getattribute__(self,name)
        raise AttributeError('Only documented device interface is available')

class AuditSimulator(b.LocalSimulator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.last_clear_time=None
    def clear(self,c):
        yes=super().clear(c)
        if yes:self.last_clear_time=self.virtual_seconds
        return yes
    def _error(self,c):
        if self._error_mode=='smooth':
            x,y=self.position
            return b.DELTA*(.55*math.sin(x/450+c*.31)+.45*math.cos(y/530-c*.13))
        return super()._error(c)


def make_targets(seed,fraction,placement):
    targets=b.make_case(seed,fraction,placement if placement in ['uniform','outward_boundary'] else 'uniform')
    if placement=='r1000':
        for t in targets:t.radius=1000.
    if placement=='cluster':
        rng=random.Random(seed+987623)
        centers=[b.mul(rng.uniform(500,1300),b.unit(rng.random()*2*math.pi)) for _ in range(3)]
        for t in targets:
            z=rng.choice(centers);p=(z[0]+rng.gauss(0,130),z[1]+rng.gauss(0,130))
            if b.norm(p)>1799:p=b.mul(1799/b.norm(p),p)
            t.position=p
    return targets


def run(task):
    gi,i,version,seedbase,trace_dir=task
    name,fraction,placement,error=SCENARIOS[gi];seed=seedbase+gi*100000+i
    sim=AuditSimulator(make_targets(seed,fraction,placement),seed,error,trace=bool(trace_dir and i==0 and gi in [1,3,6]))
    start=time.perf_counter()
    try:
        if version=='v4':r=solve_v4(DeviceOnly(sim))
        elif version=='global':r=solve_v5_global(DeviceOnly(sim))
        else:
            cfg=default_config()
            if version=='rb_only':cfg=dataclasses.replace(cfg,max_decisions=0)
            elif version=='local_only':cfg=dataclasses.replace(cfg,rb_sharing=False)
            elif version!='v5':raise ValueError(version)
            r=solve_v5(DeviceOnly(sim),cfg)
        assert sim.clear_count==len(sim._targets), 'Incomplete clear'
        assert r['stop_certificate']!='in_progress'
        assert sim.virtual_seconds<360000
        assert abs(sim.virtual_seconds-(sim.distance_m/5+5*sim.detect_count+sim.switch_count+3*sim.optical_count+2*sim.clear_count))<1e-5
        row=dict(seed=seed,scenario=name,version=version,**sim.summary(),
            wall_seconds=time.perf_counter()-start,tail_after_last_clear=sim.virtual_seconds-sim.last_clear_time,
            planner_seconds=r.get('planner_seconds',0.),planner_calls=r.get('planner_calls',0),
            planner_accepts=r.get('planner_accepts',0),free_probes=r.get('free_probes',0),
            failures=r.get('planner_failures',{}),rb_share_checks=r.get('rb_share_checks',0),
            rb_share_failures=r.get('rb_share_failures',0),certificate=r['stop_certificate'])
        if sim.trace:
            path=pathlib.Path(trace_dir);path.mkdir(parents=True,exist_ok=True)
            (path/f'LOCAL_ONLY_{version}_{seed}.json').write_text(json.dumps(dict(
                warning='SELF-BUILT LOCAL SIMULATION, NOT OFFICIAL',row=row,report=r,actions=sim.trace),ensure_ascii=False,indent=2))
    except Exception:
        import traceback
        row=dict(seed=seed,scenario=name,version=version,error=traceback.format_exc(),wall_seconds=time.perf_counter()-start)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=int,default=20);p.add_argument('--ood-cases',type=int,default=10)
    p.add_argument('--seedbase',type=int,default=106000000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--versions',nargs='+',default=['v4','rb_only','local_only','v5'])
    p.add_argument('--out',default='results/holdout.jsonl');p.add_argument('--traces',default='')
    p.add_argument('--no-ood',action='store_true');a=p.parse_args()
    if a.cases<0 or a.ood_cases<0 or a.workers<1:p.error('Invalid counts')
    tasks=[(g,i,v,a.seedbase,a.traces) for g in range(6 if a.no_ood else 9)
           for i in range(a.cases if g<6 else a.ood_cases) for v in a.versions]
    path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with path.open('w') as out,cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(run,tasks):
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
            print(f"{time.perf_counter()-start:.1f}s",row,flush=True)

if __name__=='__main__':main()

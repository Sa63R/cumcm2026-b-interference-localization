"""Run one synthetic case without any network access; NOT an official test."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from validate_v5 import AuditSimulator,DeviceOnly,make_targets
from q4_v4_solver import solve_v4
from q4_v5 import solve_v5,solve_v5_global


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['fast','global','v4'],default='fast')
    p.add_argument('--seed',type=int,default=109100001)
    p.add_argument('--fraction',type=float,default=.5)
    p.add_argument('--placement',choices=['uniform','outward_boundary','cluster','r1000'],default='uniform')
    p.add_argument('--error',choices=['hash','plus','minus','smooth'],default='hash')
    p.add_argument('--out',default='demo_result.json');a=p.parse_args()
    if not 0<=a.fraction<=1:p.error('fraction must be in [0,1]')
    sim=AuditSimulator(make_targets(a.seed,a.fraction,a.placement),a.seed,a.error,trace=True)
    start=time.perf_counter()
    func={'fast':solve_v5,'global':solve_v5_global,'v4':solve_v4}[a.mode]
    report=func(DeviceOnly(sim));elapsed=time.perf_counter()-start
    result=dict(warning='LOCAL SYNTHETIC SIMULATION, NOT OFFICIAL',mode=a.mode,seed=a.seed,
        summary=sim.summary(),wall_seconds=elapsed,report=report,actions=sim.trace)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(warning=result['warning'],**result['summary'],wall_seconds=elapsed,
         certificate=report['stop_certificate']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

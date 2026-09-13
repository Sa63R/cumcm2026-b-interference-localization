"""Five frozen Q3 policies, paired cases from local-jammers-simulator."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'vendor/simulator'),str(HERE/'vendor/policies')]
import numpy as np
import q3_base as b
import q3_v3 as v
import new_methods
from jammers_local.core import Scenario,Session,LocalClient
from jammers_local.__main__ import replay

METHODS=('v3_origin20','future_cover','optical','v2','v3')
LABELS={'v3_origin20':'v3 原点扫描','future_cover':'位置联合设计','optical':'v3 提前光学','v2':'v2','v3':'v3'}
OPTICAL=None

def initialize():
    global OPTICAL
    b.A=math.radians(1.005)
    source=HERE/'vendor/policies/optical_experiment.py'
    tree=ast.parse(source.read_text(),filename=str(source))
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='main']
    namespace={'np':np,'b':b,'v':v}
    exec(compile(tree,str(source),'exec'),namespace)
    OPTICAL=namespace['ExperimentalOpticalAgent']
    v.warmup();new_methods.warmup()

class Device:
    """Only public responses and chosen positions reach the unchanged policies."""
    def __init__(self,client):
        self.__client=client
        self._pos=np.zeros(2)
        self._channel=1
        self.request_wall_s=0.
    @property
    def pos(self):return self._pos.copy()
    @property
    def channel(self):return self._channel
    def move(self,p):
        q=np.asarray(p,dtype=float)
        if q.shape!=(2,) or not np.all(np.isfinite(q)) or np.any(np.abs(q)>2_000_000):raise ValueError('Invalid position')
        self._pos=q.copy()
    def _request(self,fn,channel):
        start=time.perf_counter()
        try:return fn(tuple(map(float,self._pos)),int(channel))
        finally:self.request_wall_s+=time.perf_counter()-start
    def detect(self,channel):
        response=self._request(self.__client.measure,channel)
        self._channel=int(channel)
        kind=response['measure_result']
        if kind=='direction':return b.Observation('bearing',math.radians(response['svd_deg']))
        if kind=='near':return b.Observation('strong')
        if kind=='no_signal':return b.Observation('none')
        raise ValueError(kind)
    def clear(self,channel):
        response=self._request(self.__client.clear,channel)
        if response['clear_result'] not in ('success','no_target_in_range'):raise ValueError(response)
        return response['clear_result']=='success'

def make_agent(device,method):
    if method=='future_cover':return new_methods.PlanningAgent(v.PublicBackend(device),scenario=False,future_cover=True)
    if method=='optical':return OPTICAL(v.PublicBackend(device))
    return v.make_agent(device,method)

def run_one(scene,method):
    session=Session(scene)
    client=LocalClient(session.dispatch)
    device=Device(client)
    error='';certificate=False;agent=None
    start=time.perf_counter()
    try:
        client.enter()
        agent=make_agent(device,method)
        agent.run()
        certificate=not agent.unseen and not agent.tracks and len(agent.cleared|agent.absent)==20 and agent.cleared==client.cleared
        if not certificate:raise AssertionError('Incomplete channel certificate')
    except Exception:error=traceback.format_exc()
    finally:
        if client.active:
            try:client.exit()
            except Exception:error+='\nExit failed: '+traceback.format_exc()
    wall=time.perf_counter()-start
    # Truth is read by the evaluator only after the policy has finished.
    data=session.export();state=data['state']
    audit=replay(data)
    if not audit['matched']:error+='\nDeterministic replay mismatch'
    normal_exit=state['stop_reason']=='user_exit'
    all_clear=state['all_cleared']
    if not all_clear:error+='\nSome sources remain uncleared'
    if method!='optical' and state['failed_clear_count']:error+='\nCertified clear failed'
    row=dict(method=method,case_id=scene.case_id,seed=scene.label,source_total=state['source_total'],
             cleared=state['cleared_count'],all_cleared=all_clear,certificate=certificate,normal_exit=normal_exit,
             success=not error and certificate and all_clear and normal_exit,error=error,
             virtual_s=state['virtual_time_s'],virtual_us=state['virtual_time_us'],
             seconds_per_source=state['virtual_time_s']/state['cleared_count'] if state['cleared_count'] else None,
             wall_s=wall,request_wall_s=device.request_wall_s,nonrequest_wall_s=wall-device.request_wall_s,
             measurements=state['measurement_count'],switches=state['switch_count'],
             failed_clears=state['failed_clear_count'],clear_attempts=state['clear_attempt_count'],
             movement_m=state['distance_m'],time_breakdown_s=state['time_breakdown_s'],
             replay_matched=audit['matched'],replay_actions=audit['compared_actions'],
             future_plan_changes=getattr(agent,'plan_changes',0))
    data['evaluation']=row.copy()
    return row,data

def work(job):
    index,seed,repeat,out=job
    scene=Scenario.generate(3,seed)
    shift=index%len(METHODS)
    order=METHODS[shift:]+METHODS[:shift]
    if repeat%2:order=order[::-1]
    rows=[]
    for method in order:
        row,data=run_one(scene,method)
        row.update(index=index,repeat=repeat)
        path=Path(out)/'runs'/f'{index:04d}_{method}_r{repeat}.json.gz'
        raw=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        packed=gzip.compress(raw,compresslevel=1,mtime=0)
        with path.open('xb') as stream:stream.write(packed)
        row.update(evidence=str(path.relative_to(out)),evidence_sha256=hashlib.sha256(packed).hexdigest())
        rows.append(row)
    return rows

def check_inputs():
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((HERE/item['snapshot']).read_bytes()).hexdigest()==item['sha256']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=int,default=2000)
    ap.add_argument('--seed-prefix',default='q3-top5-holdout-20260912-')
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit('Output must be new or empty')
    (out/'runs').mkdir(parents=True)
    check_inputs();start=time.perf_counter();initialize();warmup=time.perf_counter()-start
    jobs=[(i,args.seed_prefix+str(i),r,str(out)) for r in range(args.repeats) for i in range(args.cases)]
    total=len(jobs)*len(METHODS);rows=[]
    def save_status(state):
        data=dict(status=state,completed=len(rows),target=total,failures=sum(not r['success'] for r in rows),
                  elapsed_s=time.perf_counter()-start,last_index=rows[-1]['index'] if rows else None)
        temp=out/'status.tmp';temp.write_text(json.dumps(data));temp.replace(out/'status.json')
    save_status('running')
    print(f'{args.cases} paired cases x {len(METHODS)} methods x {args.repeats} repeats = {total} runs; warmup {warmup:.2f}s',flush=True)
    with (out/'results.jsonl').open('x') as journal:
        def accept(batch):
            for row in batch:
                rows.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n')
                if not row['success']:print('FAILED',row['seed'],row['method'],row['error'],flush=True)
            journal.flush()
            if len(rows)%100==0 or len(rows)==total:
                save_status('running');print(f'{len(rows)}/{total}; {time.perf_counter()-start:.1f}s; failures={sum(not r["success"] for r in rows)}',flush=True)
        if args.workers==1:
            for job in jobs:accept(work(job))
        else:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
                futures=[pool.submit(work,job) for job in jobs]
                for future in as_completed(futures):accept(future.result())
    check_inputs()
    import numba
    meta=dict(command=sys.argv,methods=METHODS,cases=args.cases,repeats=args.repeats,workers=args.workers,
              seed_prefix=args.seed_prefix,python=sys.version,numpy=np.__version__,numba=numba.__version__,
              platform=platform.platform(),wall_total_s=time.perf_counter()-start,warmup_s=warmup,
              source_manifest_sha256=hashlib.sha256((HERE/'source_manifest.json').read_bytes()).hexdigest(),
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              timing_scope='enter, construct policy, policy and in-process protocol, exit; excludes map creation, warmup, export and replay',
              provenance='local static reconstruction; all five policies receive identical maps and noise fields; no official service')
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    save_status('completed')
    if any(not r['success'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()

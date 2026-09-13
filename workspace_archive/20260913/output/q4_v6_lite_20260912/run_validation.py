"""Frozen paired component tests, using only the reconstructed public protocol."""
import argparse
import concurrent.futures as cf
from dataclasses import asdict,replace
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import traceback
import candidates as c
from candidates import bench
from jammers_local.core import Scenario,Source,Session,LocalClient

ROOT=c.ROOT


def verify_sources():
    bench.verify_sources()
    manifest=json.loads((ROOT/'source_hashes.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen source changed: {name}')
    return manifest


def freeze():
    if (ROOT/'plan.json').exists():raise RuntimeError('Existing plan preserved')
    cases=[]
    for i in range(1000):
        scene=Scenario.generate(4,f'q4-v6-lite-20260912-holdout-main-{i:04d}')
        cases.append(dict(key=f'main-{i:04d}',group='new_practice',scenario=scene.as_dict()))
    for group in ['all_directional','boundary_outward_r1000']:
        for i in range(70):
            n=10+i%7;trial=0
            while True:
                seed=f'q4-v6-lite-20260912-holdout-{group}-{i:04d}-{trial}'
                scene=Scenario.generate(4,seed)
                if len(scene.sources)==n:break
                trial+=1
            rng=random.Random(seed);sources=[]
            for s in scene.sources:
                theta=rng.random()*math.tau
                if group=='all_directional':
                    sources.append(replace(s,kind='directional',direction_udeg=int(math.degrees(theta)*1e6)))
                else:
                    radius=rng.uniform(1760.,1769.)
                    sources.append(Source(s.channel,round(radius*math.cos(theta)*1e6),round(radius*math.sin(theta)*1e6),
                                          1_000_000_000,'directional',int(math.degrees(theta)*1e6)))
            scene=replace(scene,sources=tuple(sources),seed_hex=None,label=seed+'-stress-transform')
            cases.append(dict(key=f'{group}-{i:04d}',group=group,scenario=scene.as_dict()))
    previous_ids=set()
    for filename in ['previous_v6_plan.json','previous_ablation_plan.json']:
        old=json.loads((ROOT/filename).read_text())['cases']
        previous_ids.update(Scenario.from_dict(case['scenario']).case_id for case in old)
    new_ids={Scenario.from_dict(case['scenario']).case_id for case in cases}
    assert len(new_ids)==len(cases) and not previous_ids.intersection(new_ids)
    bench.dump(ROOT/'plan.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        base_commit='d83999ea0ec74d35bc33e86072d3fbd0092e6162',source_hashes=verify_sources(),
        methods=c.VARIANTS,bearing_bound_deg=1.005,
        new_main_cases=1000,new_stress_cases=140,known_diagnostic_cases=0,
        primary_metric='case-equal mean of complete virtual task seconds/source',
        primary_contrasts=['lite minus full on new random cases'],
        inference='50000 paired bootstrap replicates; one prespecified primary contrast, 95% percentile CI. Stress groups, subgroups and tails descriptive or exploratory.',
        candidate_selection='Exactly one simplified candidate selected from the earlier ablations, before generating these new maps. No adaptive variants or tuning.',
        no_tuning=True,safety='Unchanged source count bound, 21-site coverage, feasible geometry and successful-clear feedback',
        runtime='warm imports/models/layout; sequential 50-case rerun separately; 120s planner and 180s per-run wall budgets unchanged',
        exclusions='No failed cases or outliers dropped; known diagnostics and stress excluded from main mean',cases=cases))


def initialize():
    bench.verify_sources();bench.initialize(1.005)
    from q4_transit_critic import Critic
    Critic(c.default_config().transit_model)


def run(task):
    case,method,destination=task
    scene=Scenario.from_dict(case['scenario']);session=Session(scene)
    client=LocalClient(session.dispatch);report=certificate=error=None
    wall=time.perf_counter();cpu=time.process_time()
    signal.signal(signal.SIGALRM,bench.time_limit);signal.setitimer(signal.ITIMER_REAL,180.)
    try:
        client.enter();report=c.solve(bench.Device(client),method);client.exit()
    except Exception:error=traceback.format_exc()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    if error is None:
        try:certificate=bench.audit(session,report)
        except Exception:error=traceback.format_exc()
    snapshot=session.engine.snapshot(truth=True);snapshot.pop('scenario')
    successful=[r['response']['virtual_time_s'] for r in session.history if r['response'].get('clear_result')=='success']
    # First/last real discovery events come only from accepted public feedback.
    discovered=set();last_discovery=None
    canonical=[]
    for r in session.history:
        response={k:v for k,v in r['response'].items() if k not in ('real_timestamp_ms','remaining_real_duration_s')}
        canonical.append(dict(path=r['path'],request=r['request'],response=response))
        if r['path']=='/measure' and response.get('measure_result') in ('direction','near'):
            channel=r['request']['channel']
            if channel not in discovered:discovered.add(channel);last_discovery=response['virtual_time_s']
    row=dict(case_key=case['key'],group=case['group'],method=method,case_id=scene.case_id,
        **snapshot,error=error,cpu_seconds=cpu,wall_seconds=wall,
        seconds_per_source=snapshot['virtual_time_s']/len(scene.sources),
        directional_count=sum(s.kind=='directional' for s in scene.sources),
        last_discovery_s=last_discovery,tail_after_last_clear_s=snapshot['virtual_time_s']-successful[-1] if successful else None,
        action_sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest())
    for name in ['planner_calls','planner_accepts','planner_seconds','planner_budget_reached','rb_share_checks',
                 'rb_share_failures','transit_stops','transit_detections','route_calls','route_accepts',
                 'guarded_steps','shared_detections','global_actions']:
        row[name]=report.get(name,0) if report else None
    row['planner_budget_reached']=bool(report and (report.get('planner_budget_reached',False) or report.get('planner_seconds',0)>=120.))
    row['visited_station_count']=len(report['visited_stations']) if report else None
    payload=dict(**session.export(),row=row,algorithm_report=report,completion_audit=certificate)
    filename=Path(destination)/'sessions'/f'{case["key"]}-{method}.json.gz'
    filename.parent.mkdir(parents=True,exist_ok=True)
    with gzip.GzipFile(filename=str(filename),mode='wb',compresslevel=6,mtime=0) as stream:
        stream.write(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--phase',choices=['smoke','main','serial'],default='main')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.freeze:freeze();return
    plan=json.loads((ROOT/'plan.json').read_text())
    assert verify_sources()==plan['source_hashes']
    assert c.VARIANTS==plan['methods']
    if args.phase=='smoke':
        cases=[dict(key='smoke',group='smoke',scenario=Scenario.generate(4,'q4-v6-lite-standalone-smoke-only').as_dict())]
    elif args.phase=='serial':cases=plan['cases'][:50]
    else:cases=plan['cases']
    destination=ROOT/args.phase;destination.mkdir(exist_ok=True)
    methods=list(c.VARIANTS);tasks=[]
    for i,case in enumerate(cases):
        order=methods[i%len(methods):]+methods[:i%len(methods)]
        tasks.extend((case,method,str(destination)) for method in order)
    started=time.perf_counter();count=errors=0
    with (destination/'records.jsonl').open('x') as stream:
        with cf.ProcessPoolExecutor(max_workers=1 if args.phase=='serial' else args.workers,initializer=initialize) as executor:
            for future in cf.as_completed([executor.submit(run,task) for task in tasks]):
                row=future.result();count+=1;errors+=int(bool(row['error']))
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if row['error']:print('ERROR',row['case_key'],row['method'],row['error'],flush=True)
                if count%60==0 or count==len(tasks):
                    print(f'{count}/{len(tasks)} runs; errors={errors}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    assert verify_sources()==plan['source_hashes']
    bench.dump(destination/'execution.json',dict(runs=count,errors=errors,cases=len(cases),
        workers=1 if args.phase=='serial' else args.workers,elapsed_seconds=time.perf_counter()-started,sources_verified=True))


if __name__=='__main__':main()

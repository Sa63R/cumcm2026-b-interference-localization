"""Re-solve only old621 truncated point-task prefixes; no policy/case execution."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.diagnose_q4_joint_visibility import observation_record
from experiments.audit_q4_joint_visibility_prefix import wire_prefix, observations
from planning.chain_route import ChainSource, ChainRouteResult
from planning.chain_route_exact import refine_truncated_route


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'research/q4_exact_frontier/prefix-diagnostic.json'
    if out.exists():raise ValueError('Preserve previous diagnostic')
    started=time.perf_counter()
    stages={'development':list(range(621001,621025)),'development-stress':list(range(621031,621045))}
    identity,events=[],[]
    total=0
    for stage,seeds in stages.items():
        directory=ROOT/'results/q4_joint_continuation'/stage
        manifest=json.loads((directory/'manifest.json').read_bytes())
        if manifest['seeds']!=seeds:raise ValueError('Only predeclared old621 development allowed')
        expected=manifest['specs']['compact_joint_continuation']
        identity.append(dict(stage=stage,manifest_sha256=sha(directory/'manifest.json'),
                             source_archive_sha256=sha(directory/'source.zip')))
        for seed in seeds:
            path=directory/'records'/f'compact_joint_continuation-{seed}.json.gz'
            record=observation_record(path)
            if record['spec']!=expected:raise ValueError('Original spec differs')
            h,before=wire_prefix(record)
            for e in record['summary']['strategy_parameters']['chain_route_log']:
                total+=1
                if e['result']['exact']:continue
                n=e['after_actual_action_count'];current=before[n][0]
                cleared={a['channel'] for a in h[:n] if a['action']=='clear' and a['result']=='success'}
                positions=[]
                for channel,logged in zip(e['source_channels'],e['source_positions']):
                    region,_,_,near,is_clear=observations(h,n,channel)
                    target=near if near is not None else tuple(region.enclosing_disk().center)
                    if is_clear or tuple(logged)!=target:raise ValueError('Task point is not prefix-derived canonical target')
                    positions.append(target)
                sources=[ChainSource(p,5.) for p in positions]
                covers=e['remaining_covers'];background=6.*max(0,20-len(cleared)-len(sources))
                old_data=dict(e['result']);old_data['order']=tuple(tuple(a) for a in old_data['order'])
                old=ChainRouteResult(**old_data)
                # Independently bind old upper bound to the same transition costs.
                cost=0.;last=current
                for kind,index in old.order:
                    p=positions[index] if kind=='source' else tuple(covers[index])
                    cost+=math.dist(last,p)/5.+(5. if kind=='source' else background)
                    last=p
                if abs(cost-old.cost_s)>1e-7:raise ValueError('Old logged upper bound cost mismatch')
                new,log=refine_truncated_route(old,covers,sources,current,background_scan_s=background,scan_source_s=0.)
                events.append(dict(stage=stage,seed=seed,record_sha256=sha(path),prefix=n,
                    current_position=current,source_channels=e['source_channels'],source_positions=positions,
                    remaining_covers=covers,background_scan_s=background,scan_source_s=0.,
                    old=asdict(old),selected=asdict(new),**log))
    summary=dict(all_route_events=total,truncated_events=len(events),dp_calls=sum(e['called'] for e in events),
        improved_upper_bounds=sum(e['applied'] for e in events),changed_first_actions=sum(e['first_action_changed'] for e in events),
        maximum_proxy_improvement_s=max((e['proxy_saved_s'] for e in events),default=0.),
        total_dp_runtime_s=sum(e['dp_result']['runtime_s'] for e in events if e['called']),
        maximum_dp_runtime_s=max((e['dp_result']['runtime_s'] for e in events if e['called']),default=0.),
        wall_s=time.perf_counter()-started,actual_new_T_or_T_LB=None,
        boundary='Proxy cost improvements only, same observed frozen task points; correlated repeated prefixes must not be summed as task-time savings')
    source_sha={p:sha(ROOT/p) for p in ('src/planning/chain_route_exact.py','src/planning/chain_route.py',
        'experiments/diagnose_q4_exact_frontier.py','experiments/diagnose_q4_joint_visibility.py',
        'experiments/audit_q4_joint_visibility_prefix.py')}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(dict(summary=summary,source_sha256=source_sha,inputs=identity,events=events),stream,indent=2,allow_nan=False)
    print(json.dumps(summary))


if __name__=='__main__':main()

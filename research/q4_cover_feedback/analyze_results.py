"""Read-only R39 diagnostics from all 119 completed rows and their first audits.

Reads row, summary and accepted wire fields only; never reads evaluation/truth.
No simulator, case generator or production strategy is imported. Output is new
and exclusive. Proxy deltas are diagnostics, never counterfactual time savings.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ('movement_s','detection_s','switching_s','optical_s','removal_s')
LABEL = 'compact_cover_feedback'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    at = fraction*(len(values)-1)
    lower = math.floor(at)
    return values[lower]*(1-(at-lower))+values[math.ceil(at)]*(at-lower)


def distribution(values):
    values = list(values)
    require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                for v in values), 'Nonfinite diagnostic value')
    if not values:
        return dict(count=0,min=None,mean=None,median=None,p95=None,max=None)
    return dict(count=len(values),min=min(values),mean=statistics.mean(values),
        median=percentile(values,.5),p95=percentile(values,.95),max=max(values))


def observed_mean(rows, key):
    values=[r.get(key) for r in rows]
    return statistics.mean(values) if all(isinstance(v,(int,float)) and not isinstance(v,bool)
        and math.isfinite(v) for v in values) else None


def metrics(rows):
    times=[r['penalized_time_s'] for r in rows]
    ratios=[r['penalized_time_s']/r['source_total'] for r in rows]
    for r in rows:
        require(type(r['successful']) is bool and type(r['source_total']) is int and 10<=r['source_total']<=16,
                'Invalid outcome or source count')
        require(r['penalized_time_s']==(r['virtual_time_s'] if r['successful'] else 360000),
                'Failure penalty changed or failure omitted')
    lower=observed_mean(rows,'common_lower_bound_s')
    actual=observed_mean(rows,'virtual_time_s')
    actual_tn=[r['virtual_time_s']/r['source_total'] for r in rows if r['virtual_time_s'] is not None]
    return dict(runs=len(rows),successful=sum(r['successful'] for r in rows),
        all_clear=all(r['successful'] for r in rows),mean_time_s=statistics.mean(times),
        mean_time_per_source_s=statistics.mean(ratios),p95_time_per_source_s=percentile(ratios,.95),
        pooled_time_per_source_s=sum(times)/sum(r['source_total'] for r in rows),
        mean_actual_time_s=actual,
        mean_actual_time_per_source_s=statistics.mean(actual_tn) if len(actual_tn)==len(rows) else None,
        mean_lower_bound_s=lower,mean_time_over_mean_lower_bound=statistics.mean(times)/lower if lower and lower>0 else None,
        mean_components_s={k:observed_mean(rows,k) for k in COMPONENTS},
        mean_program_runtime_s=observed_mean(rows,'program_runtime_s'))


def point(value):
    return [value['x'],value['y']] if isinstance(value,dict) else list(value)


def wire_costs(record):
    actions=record['summary']['action_history']
    wire=[a for a in record['history'] if a['action'] in ('/measure','/clear')
          and a['response'].get('accepted') is True]
    require(len(actions)==len(wire),'Accepted wire/summary lengths differ')
    current,channel,clock=[0.,0.],1,0.
    totals=Counter();phases=defaultdict(Counter);feedback=Counter();cost_rows=[]
    for a,w in zip(actions,wire):
        p=point(w['position']);response=w['response']
        require(w['action']=='/'+a['action'] and p==list(a['position']) and w['channel']==a['channel'],
                'Wire action/position/channel differs')
        require(a['result']==response[a['action']+'_result'] and a['virtual_time_s']==response['virtual_time_s'],
                'Wire result/clock differs')
        if a['action']=='measure' and a['result']=='direction':
            require(a['bearing_deg']==response['svd_deg'],'Actual bearing differs')
        costs={k:0. for k in COMPONENTS}
        costs['movement_s']=round(math.dist(current,p)/5*1_000_000)/1_000_000
        if a['action']=='measure':
            costs.update(detection_s=5.,switching_s=float(channel!=a['channel']))
            channel=a['channel'];feedback[a['result']]+=1
        else:
            costs.update(optical_s=3.,removal_s=2.*(a['result']=='success'))
        delta=sum(costs.values())
        require(abs(response['virtual_time_s']-clock-delta)<=2e-6,'Wire physical time increment differs')
        clock=response['virtual_time_s'];current=p
        totals.update(costs);phases[a['phase']].update(costs);phases[a['phase']]['actions']+=1
        cost_rows.append(costs)
    row=record['row']
    require(abs(clock-row['virtual_time_s'])<=2e-6,'Final wire/row time differs')
    for k in COMPONENTS:
        require(abs(totals[k]-row[k])<=max(2e-6,2e-9*max(1.,abs(row[k]))),'Row cost differs: '+k)
    require(sum(a['action']=='measure' for a in actions)==row['measurement_count'],'Measurement count differs')
    return actions,cost_rows,dict(totals),{k:dict(v) for k,v in sorted(phases.items())},dict(feedback)


def mechanism(record, audit_item):
    actions,costs,totals,phases,feedback=wire_costs(record)
    params=record['summary']['strategy_parameters'];events=params['cover_feedback_log'];routes=params['chain_route_log']
    status=Counter();eligible=Counter();fallback=Counter();ds=[];d0s=[];scored=[];completed=[];partials=[]
    veto_count=0;owned=set();previous_end=0;prediction_count=0;expanded=0;model_wall=[]
    for i,e in enumerate(events):
        require(e['id']==i and type(e['veto_selected']) is bool and type(e['executed_cover']) is bool,
                'Malformed ownership event')
        start,end=e['after_actual_action_count'],e['end_actual_action_count']
        require(type(start) is int and type(end) is int and previous_end<=start<=end<=len(actions),'Invalid event range')
        previous_end=end
        require(e['actual_vetoes_before']==veto_count,'Actual veto counter discontinuity')
        require(e['prediction_calls_before']==prediction_count and e['prediction_expanded_before']==expanded,
                'Prediction counter discontinuity')
        eligible[e['eligibility_reason'] or 'eligible']+=1
        prediction=e['prediction']
        if prediction is not None:
            prediction_count+=1;expanded+=prediction['expanded'];status[prediction['status']]+=1
            model_wall.append(prediction['runtime_s'])
            if prediction['fallback_reason']:
                fallback[prediction['fallback_reason']]+=1
            if 'D0_s' in prediction:d0s.append(prediction['D0_s'])
            if prediction['status']=='scored':
                ds.append(prediction['D_s'])
                scored.append(dict(id=i,prefix=start,channel=e['channel'],D0_s=prediction['D0_s'],
                    D_s=prediction['D_s'],base_marginal_s=prediction['base_marginal_s'],
                    world_marginals_s=prediction['world_marginals_s'],veto_selected=e['veto_selected']))
        if e['veto_selected']:
            q=e['next_cover'];route=routes[e['route_index']]
            require(route['after_actual_action_count']==start and route['selected_channel']==e['channel']
                and route['selected_kind']=='source' and route['execution_role']=='incumbent_not_executed',
                'Veto lacks original unexecuted source owner')
            require(list(route['remaining_covers'][0])==list(q),'Veto changed original next cover')
            order=route['result']['order'];channels=route['source_channels']
            require(list(order[0])==['source',channels.index(e['channel'])] and list(order[1])==['cover',0],
                    'Veto is not source then original next cover')
            for index in range(start,end):
                require(index not in owned,'Overlapping coverage ownership');owned.add(index)
                a=actions[index]
                require(a['action']=='measure' and a['phase']=='coverage' and list(a['position'])==list(q),
                        'Owned veto range includes another physical action')
            item=dict(id=i,channel=e['channel'],prefix=start,end=end,accepted_measurements=end-start,
                feedback_counts=dict(Counter(a['result'] for a in actions[start:end])),
                components_s={k:sum(c[k] for c in costs[start:end]) for k in COMPONENTS})
            if e['executed_cover']:
                require(e['status']=='completed' and end>start,'Empty/interrupted block counted as complete')
                completed.append(item);veto_count+=1
            else:partials.append(item)
        require(e['actual_vetoes_after']==veto_count and e['prediction_calls_after']==prediction_count
            and e['prediction_expanded_after']==expanded,'Post-event counter mismatch')
    if audit_item['passed']:
        require(audit_item['cover_feedback']['prefix']['cover_feedback_vetoes']==veto_count,
                'Completed veto count differs from first independent audit')
    return dict(available=True,independently_passed=audit_item['passed'],wire_action_count=len(actions),
        eligibility=dict(eligible),prediction_status=dict(status),fallback_reasons=dict(fallback),
        prediction_calls=prediction_count,prediction_expanded=expanded,model_runtime_s=distribution(model_wall),
        D0_s=distribution(d0s),D_s=distribution(ds),scored_predictions=scored,
        completed_vetoes=veto_count,completed_veto_events=completed,partial_veto_events=partials,
        actual_components_s=totals,phase_components_s=phases,actual_feedback_counts=feedback,
        actual_measurements_per_source=record['row']['measurement_count']/record['row']['source_total'])


def merge_counts(items,key):
    result=Counter()
    for item in items:result.update(item[key])
    return dict(sorted(result.items()))


def analyze_batch(directory, split, count):
    directory=Path(directory).resolve();plan=read(directory/'plan.json');summary=read(directory/'summary.json')
    audit_path=directory/'independent_audit.json';audit=read(audit_path)
    require(plan['split']==split and plan['label']==LABEL,'Unexpected fixed development identity')
    seeds=plan['seed_selection']['seeds']
    require(len(seeds)==len(set(seeds))==count,'Incomplete planned case set')
    expected={f'records/{LABEL}-{seed}.json.gz' for seed in seeds}
    require({p.relative_to(directory).as_posix() for p in (directory/'records').glob('*.json.gz')}==expected,
            'Missing or extra raw record')
    names=expected|{'manifest.json','freeze.json','source.zip','summary.json','plan.json'}
    require(set(audit['input_sha256'])==names,'First audit did not bind every original input')
    for name in names:
        path=(directory/name).resolve()
        require(path.is_relative_to(directory) and sha(path)==audit['input_sha256'][name],'Audited input changed: '+name)
    items={(a['seed'],a['strategy']):a for a in audit['audits']}
    require(len(items)==len(audit['audits'])==audit['records']==count
        and set(items)=={(s,LABEL) for s in seeds},'First audit record set differs')
    require(all(type(item['passed']) is bool for item in items.values())
        and audit['passed_records']==sum(item['passed'] for item in items.values())
        and audit['all_passed'] is (not audit['errors'] and all(item['passed'] for item in items.values())),
        'First audit flags disagree with its full case list')
    summary_rows={r['seed']:r for r in summary['rows']}
    require(len(summary_rows)==len(summary['rows'])==count and set(summary_rows)==set(seeds),
            'Summary omitted or duplicated a fixed case')
    families={t['seed']:t['family'] for t in plan['seed_selection']['trace'] if t['accepted']}
    require(set(families)==set(seeds),'Plan accepted trace differs from seeds')
    cases=[]
    for seed in sorted(seeds):
        name=f'records/{LABEL}-{seed}.json.gz'
        with gzip.open(directory/name,'rt',encoding='utf-8') as stream:record=json.load(stream)
        # No access to record['evaluation'] or any hidden-source position.
        row=record['row'];item=items[(seed,LABEL)]
        require(row==summary_rows[seed] and row['strategy']==LABEL,'Raw row differs from audited summary')
        detail=dict(row=row,family=families[seed],first_audit_passed=item['passed'],audit_errors=item['errors'])
        try: detail['mechanism']=mechanism(record,item)
        except (ValueError,KeyError,TypeError,IndexError) as error:
            detail['mechanism']=dict(available=False,diagnostic_error=type(error).__name__+': '+str(error))
        cases.append(detail)
    rows=[c['row'] for c in cases];aggregate=metrics(rows)
    for key in ('runs','successful','all_clear','mean_time_s','mean_time_per_source_s','p95_time_per_source_s',
                'pooled_time_per_source_s','mean_lower_bound_s','mean_time_over_mean_lower_bound'):
        require(aggregate[key]==summary[key],'Summary aggregate differs: '+key)
    available=[c['mechanism'] for c in cases if c['mechanism']['available']]
    trusted=[c['mechanism'] for c in cases if c['mechanism']['available'] and c['first_audit_passed']]
    combined_phases=defaultdict(Counter)
    for m in available:
        for phase,costs in m['phase_components_s'].items():combined_phases[phase].update(costs)
    mechanisms=dict(available_cases=len(available),unavailable_seeds=[c['row']['seed'] for c in cases if not c['mechanism']['available']],
        independently_passed_mechanism_cases=len(trusted),
        eligibility=merge_counts(available,'eligibility'),prediction_status=merge_counts(available,'prediction_status'),
        fallback_reasons=merge_counts(available,'fallback_reasons'),
        completed_vetoes=sum(m['completed_vetoes'] for m in trusted),
        completed_veto_cases=sum(m['completed_vetoes']>0 for m in trusted),
        partial_veto_attempts=sum(len(m['partial_veto_events']) for m in available),
        prediction_calls=sum(m['prediction_calls'] for m in available),prediction_expanded=sum(m['prediction_expanded'] for m in available),
        scored_D_s=distribution(p['D_s'] for m in available for p in m['scored_predictions']),
        scored_D0_s=distribution(p['D0_s'] for m in available for p in m['scored_predictions']),
        actual_feedback_counts=merge_counts(available,'actual_feedback_counts'),
        mean_measurements_per_source=statistics.mean(m['actual_measurements_per_source'] for m in available) if len(available)==count else None,
        mean_phase_components_s={phase:{k:v/count for k,v in costs.items()}
                                 for phase,costs in sorted(combined_phases.items())} if len(available)==count else None)
    return dict(split=split,input_directory=directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else directory.name,
        input_sha256={**audit['input_sha256'],'independent_audit.json':sha(audit_path)},
        source_sha256=plan['source_sha256'],spec=plan['spec'],
        first_audit=dict(all_passed=audit['all_passed'],passed_records=audit['passed_records'],errors=audit['errors']),
        metrics=aggregate,by_source_count={str(n):metrics([r for r in rows if r['source_total']==n]) for n in range(10,17)},
        by_family={f:metrics([c['row'] for c in cases if c['family']==f]) for f in sorted(set(families.values())-{None})},
        mechanism=mechanisms,failure_seeds=[r['seed'] for r in rows if not r['successful']],
        slowest_seeds=[r['seed'] for r in sorted(rows,key=lambda r:(-r['penalized_time_s']/r['source_total'],r['seed']))[:20]],
        cases=cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random',type=Path,default=ROOT/'results/q4_cover_feedback/development')
    parser.add_argument('--stress',type=Path,default=ROOT/'results/q4_cover_feedback/development-stress')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve existing diagnostic output')
    batches=[analyze_batch(args.random,'development',70),analyze_batch(args.stress,'development-stress',49)]
    require(batches[0]['source_sha256']==batches[1]['source_sha256'] and batches[0]['spec']==batches[1]['spec'],
            'Two development batches used different source or spec')
    result=dict(schema='r39_readonly_actual_diagnostics_v1',script_sha256=sha(__file__),batches=batches,
        scope='All 119 fixed rows, including failed cases, bound to raw wire and original first audits. Complete-veto totals count independently passed records only; unavailable mechanisms remain explicit.',
        limitations=['D and D0 are finite proxy scores, not realized time savings or paired effects.',
            'Coverage-block costs include compulsory original scanning; they are not all extra costs caused by a veto.',
            'No future truth, latent source positions, counterfactual route or cross-batch causal subtraction is used.',
            'Actual post-cover readiness is not reconstructed here; actual feedback outcomes are reported without claiming nominal readiness occurred.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),runs=119,
        completed_vetoes=sum(b['mechanism']['completed_vetoes'] for b in batches),
        unavailable_mechanism_cases=sum(len(b['mechanism']['unavailable_seeds']) for b in batches)),ensure_ascii=False))


if __name__=='__main__':main()

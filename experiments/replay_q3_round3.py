"""Replay saved real choices and inspect compatible witnesses OFFLINE only.

Actual scenario data is owned by the replay evaluator. Witness construction
receives only public controller history and previously sampled nominal worlds.
No diagnostic changes the recorded real trajectory or feeds validation fitting.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q3_fresh import StressEngine
from experiments.audit_q3_fresh_choices import assert_same_observation
from experiments.run_q3_fresh_round3 import make_policy
from simulation.cases import Scenario,Source
from simulator_client.state import Position
from strategies.q3_round3 import Proposal,Round3Rollout,apply_proposal,plain,digest
from strategies.q3_fresh_stepper import Action
from strategies.q3_round3_exploration import compatible_witnesses


def scenario(data):
    return Scenario(**dict(data,sources=tuple(Source(**s) for s in data['sources'])))


def proposal(data):
    return Proposal(data['label'],tuple(Action(**dict(a,position=tuple(a['position']))) for a in data['actions']),
                    data['kind'],tuple(data['original_position']) if data['original_position'] is not None else None,data['anchor'])


def replay(path,*,inspect_risks=False,max_risk_decisions=12,anchors=()):
    raw=gzip.open(path,'rb').read();trace=json.loads(raw);row=trace['row']
    engine=StressEngine(scenario(trace['scenario']),anchors)
    policy,_=make_policy(row['strategy'],engine.client())
    decisions={d['action_index']:d for d in row['planner_stats'].get('decisions',[])}
    directory=path.parent/f"branches-{path.name.split('-')[1]}"/row['strategy']
    expected=trace['search']['action_history'];risks=[];seen=[];risk_count=0
    started=time.perf_counter();cpu=time.process_time();error=None
    policy.client.enter()
    try:
        while policy.next_action() is not None:
            index=len(policy.history)
            if index in decisions:
                record=decisions[index]
                journal=json.loads(gzip.open(directory/record['evidence_file'],'rb').read())
                if journal['controller']['history_sha256']!=digest(policy.history):raise ValueError('Public history differs')
                if journal['controller']['stack']!=plain(policy.stack):raise ValueError('B controller stack differs before choice')
                if journal['controller']['pending']!=plain(policy.pending):raise ValueError('Pending B action differs')
                choices=[proposal(p) for p in journal['proposals']]
                if 'selected_index' in record:
                    selected=record['selected_index']
                else:
                    matches=[i for i,p in enumerate(choices) if p.label==record['selected'] and
                        p.actions[0].channel==expected[index]['channel'] and
                        list(p.actions[0].position)==list(expected[index]['position'])]
                    if len(matches)!=1:raise ValueError('Ambiguous legacy selected candidate')
                    selected=matches[0]
                chosen=choices[selected]
                if (inspect_risks and selected and risk_count<max_risk_decisions and
                        policy.client.state.position.distance_to(Position(*chosen.actions[0].position))>=350):
                    risk_count+=1;planner=Round3Rollout();deadline=time.perf_counter()+10
                    result=dict(action_index=index,selected=chosen.label,complete=False,error=None,
                                nominal_prediction=record.get('confirmation'),witnesses=[],branches=[])
                    try:
                        worlds,labels=compatible_witnesses(policy,chosen,[scenario(w) for w in journal['worlds']],deadline)
                        result.update(witnesses=plain(worlds),labels=labels)
                        delta=[]
                        for wi,w in enumerate(worlds):
                            a=planner._evaluate_proposal(policy,choices[0],w,deadline,wi,0,'offline_risk')
                            b=planner._evaluate_proposal(policy,chosen,w,deadline,wi,selected,'offline_risk')
                            delta.append(b-a)
                        result.update(complete=True,delta_s=delta,max_positive_delta_s=max([0]+delta))
                    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
                    result['branches']=planner.branch_records;risks.append(result)
                apply_proposal(policy,chosen);seen.append(index)
            policy.execute_pending()
            assert_same_observation(policy.history[-1],expected[index],index)
        policy.client.exit()
        if len(policy.history)!=len(expected) or set(seen)!=set(decisions):raise ValueError('Replay length/decision mismatch')
        if abs(policy.client.state.virtual_time_s-row['virtual_time_s'])>1e-5:raise ValueError('Replay total differs')
        if not policy.certified or not engine.evaluation()['all_cleared']:raise ValueError('Replay completion failed')
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    return dict(case_id=row['case_id'],strategy=row['strategy'],replay_valid=error is None,replay_error=error,
        trace_sha256=hashlib.sha256(raw).hexdigest(),source_truth_used_only_in_replay_evaluator=True,
        virtual_time_s=row['virtual_time_s'],physical_lower_bound_s=row['physical_lower_bound_s'],
        guarantee_lower_bound_s=row['guarantee_lower_bound_s'],time_over_physical_lower_bound=row['time_over_physical_lower_bound'],
        time_over_guarantee_lower_bound=row['time_over_guarantee_lower_bound'],risk_diagnostics=risks,
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--strategy',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--risks',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();manifest=json.loads((a.batch/'results.json').read_text())['manifest']
    assert manifest['status']=='completed'
    paths=sorted(a.batch.glob(f'trace-*-{a.strategy}.json.gz'))
    if a.limit is not None:paths=paths[:a.limit]
    reports=[];a.out.mkdir(parents=True,exist_ok=False)
    for path in paths:
        case=json.loads(gzip.open(path,'rb').read())['row']['case_id']
        r=replay(path,inspect_risks=a.risks,anchors=manifest.get('metadata',{}).get('anchors',{}).get(case,()))
        reports.append({k:v for k,v in r.items() if k!='risk_diagnostics'})
        with gzip.open(a.out/(path.stem+'.replay.json.gz'),'wt',encoding='utf-8',compresslevel=1) as f:json.dump(r,f,ensure_ascii=False)
        (a.out/'summary.json').write_text(json.dumps(dict(status='running',reports=reports),indent=2),encoding='utf-8')
        print(json.dumps({k:r[k] for k in ('case_id','strategy','replay_valid','replay_error')}),flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(status='completed',reports=reports,
        all_replays_valid=all(r['replay_valid'] for r in reports),total_cpu_s=sum(r['cpu_s'] for r in reports)),indent=2),encoding='utf-8')


if __name__=='__main__':main()

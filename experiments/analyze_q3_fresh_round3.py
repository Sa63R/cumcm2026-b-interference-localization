"""Round-three paired analysis, relative-regret tails and journal audits."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.analyze_q3_fresh_round2 import analyze,paired_comparison,write_outputs
from experiments.audit_q3_round3_journals import run as audit_journals


def development_map_groups(cases):
    """Group shared geometries, including source-count/radius/error variants.

    This uses case definitions only, never outcomes. Original saved metadata is
    retained in logs; the analysis corrects a legacy label that included N.
    """
    points=[frozenset((s['channel'],s['x'],s['y']) for s in c['sources']) for c in cases]
    parent=list(range(len(cases)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,a in enumerate(points):
        for j,b in enumerate(points[:i]):
            if a<=b or b<=a:parent[root(i)]=root(j)
    members=defaultdict(list)
    for i in range(len(cases)):members[root(i)].append(i)
    groups={}
    for indices in members.values():
        union=sorted(set().union(*(points[i] for i in indices)))
        key='development-map-'+hashlib.sha256(json.dumps(union).encode()).hexdigest()[:20]
        groups.update({cases[i]['case_id']:key for i in indices})
    return groups


def regret(rows,name):
    baseline={r['case_id']:r for r in rows if r['strategy']=='B'}
    selected=[r for r in rows if r['strategy']==name and r['case_id'] in baseline]
    complete=all(r['analysis_success'] and baseline[r['case_id']]['analysis_success'] for r in selected)
    ratios=[(r['virtual_time_s']-baseline[r['case_id']]['virtual_time_s'])/baseline[r['case_id']]['virtual_time_s'] for r in selected]
    if not complete or not ratios:return dict(complete=False,paired_cases=len(selected))
    descending=sorted((max(0,r) for r in ratios),reverse=True)
    positives=sorted((r for r in ratios if r>1e-12),reverse=True)
    k=max(1,math.ceil(.1*len(ratios)));kp=max(1,math.ceil(.1*len(positives)))
    return dict(complete=True,paired_cases=len(ratios),max_signed_relative_regret=max(ratios),
        max_positive_relative_regret=max(descending),tail10_all_cases_positive_mean=statistics.mean(descending[:k]),
        tail10_all_cases_count=k,tail10_positive_cases_only_mean=statistics.mean(positives[:kp]) if positives else 0,
        tail10_positive_cases_count=min(kp,len(positives)),positive_cases=len(positives),
        negative_cases=sum(r < -1e-12 for r in ratios),mean_relative_delta=statistics.mean(ratios),
        case_relative_regrets={r['case_id']:q for r,q in zip(selected,ratios)})


def complete_analysis(paths,with_journals=True):
    started=time.process_time()
    result,rows=analyze(paths)
    definitions=json.loads((ROOT/'research/q3_fresh_round3/development_input.json').read_text(encoding='utf-8'))
    corrected=development_map_groups(definitions['cases'])
    changes={}
    for row in rows:
        if row['case_id'] in corrected:
            old=row['original_case_group'];new=corrected[row['case_id']]
            row['recorded_original_case_group']=old;row['original_case_group']=new
            if old!=new:changes[row['case_id']]=dict(recorded=old,analysis=new)
    result['development_grouping_correction']=dict(method='Connected identical/subset channel-position geometries; ignores radius/error variants; definitions only',cases=changes)
    unique=[r for r in rows if r['deduplication_status']=='primary']
    grouped=defaultdict(list)
    for row in unique:grouped[row['suite_group']].append(row)
    for group,items in grouped.items():
        names=sorted({r['strategy'] for r in items})
        pairs=[('CR',name) for name in names if name not in ('B','CR') and 'CR' in names]
        pairs += [(a,b) for a,b in (('G1','G12'),('G12','G3'),('G3','G3W'),('G3','G3V')) if a in names and b in names]
        pairs += [(p['baseline'],p['candidate']) for p in result['groups'][group]['paired'].values()]
        for a,b in pairs:
            result['groups'][group]['paired'][f'{a}_vs_{b}']=paired_comparison(items,a,b)
        result['groups'][group]['regret_vs_B']={name:regret(items,name) for name in names if name!='B'}
    issues=[];journal_paths=[]
    for path in map(Path,paths):
        file=path/'results.json'
        if not file.is_file():continue
        data=json.loads(file.read_text(encoding='utf-8'));m=data['manifest']
        if m['status']!='completed':continue
        if not m.get('source_unchanged',False):issues.append(dict(batch=path.name,error='source_changed_or_not_certified'))
        if m.get('reuse'):
            for entry in m['source_evidence']:
                if hashlib.sha256((ROOT/entry['source']).read_bytes()).hexdigest()!=entry['sha256']:
                    issues.append(dict(batch=path.name,error='reused_source_hash_mismatch'))
            continue
        journal_paths.append(path)
        for trace in path.glob('trace-*.json.gz'):
            d=json.loads(gzip.open(trace,'rb').read());row=d['row'];stats=row['planner_stats']
            directory=path/f"branches-{trace.name.split('-')[1]}"/row['strategy']
            files=list(directory.glob('decision-*.json.gz'))
            if len(files)!=stats.get('planning_calls',0):issues.append(dict(batch=path.name,case=row['case_id'],error='journal_count_mismatch'))
            for record in stats.get('decisions',[]):
                fp=directory/record.get('evidence_file','MISSING')
                if not fp.is_file():continue
                raw=gzip.open(fp,'rb').read()
                if hashlib.sha256(raw).hexdigest()!=record['evidence_sha256']:
                    issues.append(dict(batch=path.name,case=row['case_id'],error='journal_hash_mismatch'))
                journal=json.loads(raw)['decision']
                if any(journal.get(key)!=record.get(key) for key in ('selected','action_index','completed','paired_delta_s','confirmation','nominal_accept')):
                    issues.append(dict(batch=path.name,case=row['case_id'],error='journal_decision_mismatch'))
    result['protocol_issues']=issues
    if with_journals and journal_paths:result['journal_audit']=audit_journals(journal_paths)
    else:result['journal_audit']=dict(passed=None,reason='explicitly skipped or only legacy reused evidence')
    result['audit']['passed']=result['audit']['passed'] and not issues and result['journal_audit']['passed'] is not False
    result['analysis_cpu_s']=time.process_time()-started
    return result,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--skip-journals',action='store_true')
    a=p.parse_args();paths=[Path(s) if Path(s).is_dir() else ROOT/'results/q3_fresh_round3'/s for s in a.batches]
    d,rows=complete_analysis(paths,not a.skip_journals);write_outputs(d,rows,a.out)
    print(json.dumps(dict(runs=d['unique_unconflicted_runs'],audit_passed=d['audit']['passed'],
                         journal_counts=d['journal_audit'].get('counts'),protocol_issues=d['protocol_issues'])))


if __name__=='__main__':main()

"""Freeze old synthetic development/regression inputs and reuse B/CR evidence."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q3_fresh_round2 import build_suite, summarize


def main():
    old=ROOT/'results/q3_fresh_round2'
    cases,metadata=build_suite('development')
    sources=[]
    for name in ('development_bc','development_cr','holdout_nominal','holdout_pressure'):
        for path in sorted((old/name).glob('trace-*.json.gz')):
            d=json.loads(gzip.open(path,'rb').read())
            if d['row']['strategy'] in ('B','CR'):
                sources.append((path,d))
    nominal=sorted({d['row']['case_id'] for _,d in sources if d['row']['suite_group']=='nominal'},
                   key=lambda c:hashlib.sha256(('round3-regression:'+c).encode()).hexdigest())[:12]
    pressure_pairs=list(range(16))
    pressure_pairs.sort(key=lambda i:hashlib.sha256(f'round3-regression-pressure:{i}'.encode()).hexdigest())
    pressure_pairs=[13]+[i for i in pressure_pairs if i!=13][:5]  # Includes both 970026/27.
    selected_pressure=[]
    by_case={d['row']['case_id']:d for _,d in sources}
    for index in sorted(i for pair in pressure_pairs for i in (2*pair,2*pair+1)):
        path=old/'holdout_pressure'/f'trace-{index:03d}-B.json.gz'
        selected_pressure.append(json.loads(gzip.open(path,'rb').read())['row']['case_id'])
    records=[asdict(c) for c in cases]+[by_case[c]['scenario'] for c in nominal+selected_pressure]
    groups={**metadata['suite_groups'],**{c:'regression_nominal' for c in nominal},
            **{c:'regression_pressure' for c in selected_pressure}}
    original={c['case_id']:c['case_id'] for c in records}
    # Old paired pressure variants share map/error seeds; group them in analysis.
    for c in records:
        if c['case_id'].startswith('q3-r2-'):
            family=c['case_id'].split('-')[3]
            original[c['case_id']]=f"old-pressure-{family}-{c['seed']}"
    payload=dict(cases=records,metadata=dict(suite_groups=groups,original_case_groups=original,
        role='previously exposed synthetic development and regression, not new test',
        regression_nominal_rule='first12 SHA256(round3-regression:case_id)',
        regression_pressure_pair_indices=pressure_pairs,
        explicitly_retained_regression='q3-r2-hiddenfar-16-970027',validation_used=False))
    p=ROOT/'research/q3_fresh_round3/development_input.json'
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    dest=ROOT/'results/q3_fresh_round3/reference_development';dest.mkdir(parents=True,exist_ok=False)
    rows=[];proof=[]
    for index,c in enumerate(records):
        for name in ('B','CR'):
            path,d=next((p,d) for p,d in sources if d['row']['case_id']==c['case_id'] and d['row']['strategy']==name)
            d['row'].update(suite_group=groups[c['case_id']],original_case_group=original[c['case_id']])
            rows.append(d['row'])
            with gzip.open(dest/f'trace-{index:03d}-{name}.json.gz','wt',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False)
            proof.append(dict(source=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=dict(status='completed',cases=len(records),configs=['B','CR'],completed_cases=len(records),
        completed_runs=len(rows),case_ids=[c['case_id'] for c in records],reuse=True,total_cpu_s=0,
        source_unchanged=True,source_evidence=proof,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (dest/'results.json').write_text(json.dumps(dict(manifest=manifest,summary=summarize(rows),rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    print('frozen',len(records),'synthetic cases; reused',len(rows),'B/CR runs')


if __name__=='__main__':main()

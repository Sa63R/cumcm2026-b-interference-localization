"""Replay frozen compressed sessions and verify deterministic serial reruns."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import components as c
from components import bench
from jammers_local.__main__ import replay


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','serial']);a=p.parse_args()
    root=c.ROOT
    if a.phase=='replay':
        rows=[];started=time.perf_counter()
        prior={(r['case_key'],r['method']):r for r in [json.loads(x) for x in (root/'prior_records.jsonl').read_text().splitlines()]}
        known_checks=[]
        for i,path in enumerate(sorted((root/'main/sessions').glob('*.json.gz'))):
            with gzip.open(path,'rt') as stream:session=json.load(stream)
            result=replay(session)
            canonical=[]
            for r in session['history']:
                response={k:v for k,v in r['response'].items() if k not in ('real_timestamp_ms','remaining_real_duration_s')}
                canonical.append(dict(path=r['path'],request=r['request'],response=response))
            expected=session['row']['action_sha256']
            assert hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()==expected
            row=session['row']
            if row['group']=='known_regression_diagnostic' and row['method'] in ('v4','v5','full'):
                key=row['case_key'].removeprefix('known-')
                method='v6' if row['method']=='full' else row['method']
                old=prior[key,method]
                assert row['action_sha256']==old['action_sha256']
                assert row['virtual_time_us']==old['virtual_time_us']
                known_checks.append(dict(case_key=key,method=method,original_trace_equal=True))
            rows.append(dict(file=path.name,**result))
            if (i+1)%300==0:print('replayed',i+1,'matched',sum(r['matched'] for r in rows),flush=True)
        assert len(rows)==4476 and all(r['matched'] for r in rows) and len(known_checks)==9
        bench.dump(root/'replay_verification.json',dict(runs=len(rows),all_matched=True,
            actions=sum(r['compared_actions'] for r in rows),elapsed_seconds=time.perf_counter()-started,known_original_trace_checks=known_checks,records=rows))
        return
    import statistics as st
    main_rows=[json.loads(x) for x in (root/'main/records.jsonl').read_text().splitlines()]
    index={(r['case_key'],r['method']):r for r in main_rows}
    rows=[json.loads(x) for x in (root/'serial/records.jsonl').read_text().splitlines()]
    assert len(rows)==240
    checks=[]
    for r in rows:
        expected=index[r['case_key'],r['method']]
        assert r['error'] is None and r['action_sha256']==expected['action_sha256']
        assert r['virtual_time_us']==expected['virtual_time_us']
        checks.append(dict(case_key=r['case_key'],method=r['method'],full_trace_hash_equal=True))
    summary={name:{field:st.mean(r[field] for r in rows if r['method']==name)
                   for field in ['cpu_seconds','wall_seconds','seconds_per_source','planner_seconds']}
             for name in c.VARIANTS}
    bench.dump(root/'serial/summary.json',dict(cases=20,runs=240,all_equal=True,methods=summary,checks=checks))


if __name__=='__main__':main()

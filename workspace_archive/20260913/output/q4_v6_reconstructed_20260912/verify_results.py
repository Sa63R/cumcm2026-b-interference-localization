"""Replay every batch transcript and compare sequential timing reruns."""
import argparse
import json
import time
import benchmark as bench
from jammers_local.__main__ import replay


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['replay','serial']);a=p.parse_args()
    root=bench.ROOT
    if a.mode=='replay':
        started=time.perf_counter();rows=[]
        for i,path in enumerate(sorted((root/'main/sessions').glob('*.json'))):
            result=replay(json.loads(path.read_text()))
            rows.append(dict(file=path.name,**result))
            if (i+1)%100==0:print('replayed',i+1,'actions',sum(r['compared_actions'] for r in rows),flush=True)
        bench.dump(root/'replay_verification.json',dict(runs=len(rows),all_matched=all(r['matched'] for r in rows),
            compared_actions=sum(r['compared_actions'] for r in rows),elapsed_seconds=time.perf_counter()-started,results=rows))
        assert len(rows)==1110 and all(r['matched'] for r in rows)
        return
    bench.initialize(1.005)
    plan=json.loads((root/'plan.json').read_text())
    expected={(r['case_key'],r['method']):r for r in [json.loads(x) for x in (root/'main/records.jsonl').read_text().splitlines()]}
    destination=root/'serial_runtime';destination.mkdir(exist_ok=True)
    rows=[]
    with (destination/'records.jsonl').open('x') as stream:
        for i,case in enumerate(plan['cases'][:20]):
            methods=plan['methods'][i%3:]+plan['methods'][:i%3]
            for r in bench.run_case((case,methods,str(destination))):
                r['matches_parallel_trace']=r['action_sha256']==expected[r['case_key'],r['method']]['action_sha256']
                rows.append(r);stream.write(json.dumps(r)+'\n');stream.flush()
            print('serial',i+1,'/20',flush=True)
    assert all(not r['error'] and r['matches_parallel_trace'] for r in rows)
    import statistics as st
    bench.dump(destination/'summary.json',dict(cases=20,runs=60,all_trace_equal=True,
        methods={m:{k:st.mean(r[k] for r in rows if r['method']==m) for k in ['cpu_seconds','wall_seconds','seconds_per_source']}
                 for m in plan['methods']},
        note='One worker, same first 20 frozen cases, rotating strategy order, warm imports/models/layout, excludes export and audit; V4 original computation, no speed cache.'))


if __name__=='__main__':main()

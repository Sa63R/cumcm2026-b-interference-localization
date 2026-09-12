"""Replay saved proof identities and strict public counterexamples; no search."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from planning.q4_directional_cover import verify_directional_cover_certificate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)


def main():
    data=json.loads((HERE/'results.json').read_bytes());first=json.loads((HERE/'freeze.json').read_bytes())
    resumed=json.loads((HERE/'resume-freeze.json').read_bytes())
    need(digest(HERE/'run_diagnostic-first-attempt.py')==first['script_sha256'],'Original script changed')
    need(digest(HERE/'run_diagnostic.py')==resumed['script_sha256'],'Resumed script changed')
    need(digest(HERE/'PROTOCOL.md')==first['protocol_sha256']==resumed['protocol_sha256'],'Protocol changed')
    need(digest(HERE/'freeze.json')==resumed['original_freeze_sha256'],'Original freeze changed')
    for name,sha in resumed['dependencies'].items():need(digest(ROOT/name)==sha,'Geometry/parent dependency changed')
    seen=set();statuses=collections.Counter();verified=0
    for item in data['candidates']:
        path=HERE/item['proof_file'];need(path not in seen,'Repeated proof');seen.add(path)
        need(digest(path)==item['proof_sha256'],'Proof SHA changed')
        proof=json.loads(gzip.decompress(path.read_bytes()))
        points=[list(p) for p in sorted(set(map(tuple,item['geometry'])))]
        need(points==proof['stations'],'Proof geometry mismatch')
        for key,value in first['certificate_options'].items():
            if key!='include_leaves':need(proof[key]==value,'Proof options changed')
        need(proof['status']==item['status'],'Status differs')
        statuses[item['status']]+=1
        if proof['status']=='certified':
            need(verify_directional_cover_certificate(points,proof)['passed'],'Leaf verification failed')
        elif proof['status']=='counterexample':
            s=proof['counterexample']['source'];n=proof['counterexample']['normal']
            need(math.hypot(*s)<=1800. and abs(math.hypot(*n)-1.)<=1e-10,'Invalid witness domain')
            for p in points:
                projected=n[0]*(p[0]-s[0])+n[1]*(p[1]-s[1])
                need(math.dist(p,s)>1000.+1e-5 or projected < -1e-5,'Station defeats strict witness')
            verified+=1
        else:need(proof['status']=='inconclusive' and not proof['passed'],'Unknown falsely certified')
    need(seen==set((HERE/'proofs').glob('*.json.gz')),'Missing/unlisted proof file')
    need(dict(statuses)==data['summary']['statuses'],'Summary status count differs')
    need(len(data['records'])==38 and all(r['prefix_audit']['passed'] for r in data['records']),'Missing old prefix audit')
    for prefix in data['prefixes']:
        for evidence in prefix['public_pool_evidence']:
            need(set(evidence['negative_action_indices'])==set(map(str,prefix['unknown_channels'])),'Incomplete channel certificate')
            need(all(ids and max(ids)<prefix['after_actual_action_count'] for ids in evidence['negative_action_indices'].values()),'Not a real earlier observation index')
    result=dict(passed=True,proofs=len(seen),strict_counterexamples_verified=verified,statuses=dict(statuses),
        results_sha256=digest(HERE/'results.json'),scope='Existing file/proof replay only; unknown remains unknown; no strategy/scenario/coverage-search run')
    with (HERE/'verification.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

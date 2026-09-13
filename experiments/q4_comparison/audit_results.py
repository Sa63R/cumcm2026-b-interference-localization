"""Verify complete pairing and archive provenance; no simulator interaction."""
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((args.result_dir/'manifest.json').read_text())
    rows=[json.loads(l) for l in (args.result_dir/'runs.jsonl').read_text().splitlines()]
    policy_changes=[name for name,digest in manifest['hashes'].items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
    original=json.loads((root/'vendor_v4/manifest.json').read_text())
    vendor_changes=[name for name,digest in original.items() if (root/'vendor_v4'/name).exists() and hashlib.sha256((root/'vendor_v4'/name).read_bytes()).hexdigest()!=digest]
    keyed=defaultdict(dict)
    for r in rows:
        if r['method'] in keyed[r['seed']]:raise AssertionError('Duplicate case/method')
        keyed[r['seed']][r['method']]=r
        if not math.isclose(r['virtual_seconds']/r['targets'],r['seconds_per_source'],abs_tol=1e-9):raise AssertionError('Metric mismatch')
    complete=0
    for seed,data in keyed.items():
        if len(data)==5:complete+=1
        if len({r['targets'] for r in data.values()})!=1:raise AssertionError('Mismatched scene source count')
    expected=manifest['cases_per_group']*len(manifest['groups'])
    result=dict(expected_cases=expected,completed_paired_cases=complete,policy_runs=len(rows),successful_runs=sum(r['success'] for r in rows),failures=[(r['seed'],r['method'],r['exception']) for r in rows if not r['success']],policy_changes=policy_changes,modified_vendor_files=vendor_changes,final_coverage_certificates=sum(r.get('coverage_verified') is True for r in rows))
    if policy_changes or vendor_changes:raise AssertionError(result)
    if args.require_complete and (complete!=expected or len(rows)!=expected*5):raise AssertionError(result)
    (args.result_dir/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()

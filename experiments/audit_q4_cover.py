"""Independent prefix/physics audit plus a freshly checked compact Q4 cover."""
import argparse
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.audit_q4_state import wire_audit,observation_audit,digest,require


@lru_cache(maxsize=32)
def certified(points):
    from planning.q4_directional_cover import certify_directional_cover, verify_directional_cover_certificate
    certificate=certify_directional_cover(points)
    verify_directional_cover_certificate(points, certificate)
    require(certificate.get('passed') is True,'Declared cover lacks a complete independent geometric certificate')
    summary={k:v for k,v in certificate.items() if k!='leaves'}
    summary['leaf_witness_sha256']=digest(certificate['leaves'])
    return summary


def audit_record(record):
    row=record['row']; report=record.get('summary')
    result={k:row[k] for k in ('case_id','strategy','successful','virtual_time_s','common_lower_bound_s','time_over_lower_bound')}
    errors=[]
    try:
        require(record.get('evaluation_phase')=='after_policy_termination','Missing evaluation phase')
        require(row['penalized_time_s']==(row['virtual_time_s'] if row['successful'] else 360000),'Incorrect failure penalty')
        result['physical']=wire_audit(record)
        if row['strategy'].startswith('compact_'):
            require(report is not None,'Missing compact summary')
            points=tuple(tuple(p) for p in report['coverage_points'])
            certificate=certified(points)
            result['geometric_certificate']=certificate
            result['observations']=observation_audit(record,certified_stations=points)
        else:
            result['observations']=observation_audit(record)
        from experiments.q4_comparison_bounds import common_bound
        bounds=common_bound(record['evaluation']['ground_truth'])
        lower=bounds['common_lower_bound_s']
        require(abs(lower-row['common_lower_bound_s'])<1e-7,'Common lower bound mismatch')
        require(abs(row['time_over_lower_bound']-row['virtual_time_s']/lower)<1e-9,'T/LB mismatch')
        require(not row['successful'] or bounds['common_lower_bound_rounded_s']<=row['virtual_time_s']+1e-6,'Bound exceeds completed time')
    except (ValueError,AssertionError,KeyError,TypeError) as exc:
        errors.append(str(exc))
    result.update(passed=not errors,errors=errors)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); directory=args.input
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    freeze=json.loads((directory/'freeze.json').read_text(encoding='utf-8'))
    errors=[]
    if digest(manifest)!=freeze['manifest_sha256']: errors.append('Manifest changed')
    with zipfile.ZipFile(directory/'source.zip') as archive:
        for path,expected in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(path)).hexdigest()!=expected: errors.append('Archive changed: '+path)
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected: errors.append('Audit imports differ from freeze: '+path)
    audits=[]; record_rows=[]; keys=set(); truth={}
    for path in sorted((directory/'records').glob('*.json.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as stream: record=json.load(stream)
        row=record['row']; key=(row['seed'],row['strategy'])
        if key in keys: errors.append('Duplicate pair key')
        keys.add(key)
        if record['spec']!=manifest['specs'][row['strategy']]: errors.append('Spec mismatch')
        signature=digest(record['evaluation']['ground_truth'])
        if signature!=row['case_sha256'] or (row['seed'] in truth and truth[row['seed']]!=signature): errors.append('Case identity mismatch')
        truth[row['seed']]=signature
        record_rows.append(row)
        audits.append(audit_record(record))
    expected={(seed,label) for seed in manifest['seeds'] for label in manifest['specs']}
    if expected!=keys: errors.append('Incomplete pair matrix')
    from experiments.run_q4_cover_study import report_rows
    recomputed=report_rows(sorted(record_rows,key=lambda r:(r['seed'],r['strategy'])))
    saved=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if saved!=recomputed: errors.append('Published summary/rows differ from audited records')
    out={'kind':'Q4 compact geometric + physical + prefix + common bound audit',
         'records':len(audits),'passed_records':sum(a['passed'] for a in audits),
         'all_passed':not errors and all(a['passed'] for a in audits),'global_errors':errors,
         'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audits':audits}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(out,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k!='audits'}))
    return 0 if out['all_passed'] else 1


if __name__=='__main__': raise SystemExit(main())

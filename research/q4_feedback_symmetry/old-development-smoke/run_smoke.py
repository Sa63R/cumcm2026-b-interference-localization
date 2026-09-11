"""Six explicitly reused 621 QA cases; never opens the reserved 625 sets."""
from pathlib import Path
import datetime, gzip, hashlib, json, subprocess, sys, traceback, zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from experiments.run_q4_round2 import hashes, one
from experiments.audit_q4_cover import audit_record
from experiments.audit_q4_feedback_symmetry import audit_feedback_symmetry_prefix
from experiments.audit_q4_range import audit_range_prefix
from experiments.audit_q4_scheduling import audit_scheduling_prefix
from experiments.audit_q4_optical import audit_optical_prefix

OUT = Path(__file__).resolve().parent
CASES = ((621001, 'pilot'), (621004, 'pilot'), (621042, 'stress'))
LABELS = ('compact_feedback_mean', 'compact_feedback_centers')

def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    expected=hashes()
    spec_path=ROOT/'research/q4_feedback_symmetry/development-specs.json'
    specs=json.loads(spec_path.read_bytes())
    contract={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'research/q4_feedback_symmetry/MODEL.md',
        'research/q4_feedback_symmetry/PROTOCOL.md',
        'research/q4_feedback_symmetry/development-specs.json')}
    manifest=dict(scope='Implementation QA only: six reused old development cases; no 625 generation or new performance evidence',
        cases=CASES,specs={k:specs[k] for k in LABELS},source_sha256=expected,
        control_sha256=contract,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(OUT/'manifest.json',manifest)
    with zipfile.ZipFile(OUT/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in expected: z.write(ROOT/p,p)
    records=OUT/'records';records.mkdir(exist_ok=False)
    results=[]
    for seed,stage in CASES:
        for label in LABELS:
            item=dict(seed=seed,stage=stage,strategy=label,passed=False)
            try:
                record=one(seed,stage,label,specs[label],expected)
                path=records/f'{label}-{seed}.json.gz'
                with gzip.open(path,'wt',encoding='utf-8') as f: json.dump(record,f,ensure_ascii=False,allow_nan=False)
                item.update(row=record['row'],record_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                generic=audit_record(record)
                params=(record.get('summary') or {}).get('strategy_parameters',{})
                if params.get('range_skipped_scans'): generic['range_prefix']=audit_range_prefix(record)
                if params.get('q4_r2_scheduling'): generic['scheduling_prefix']=audit_scheduling_prefix(record)
                if params.get('optical_cover_log'): generic['optical_prefix']=audit_optical_prefix(record)
                item['generic']=generic
                # This checker independently invokes the original R8 and R12
                # audits via a strict source-contract/spec view, not altered wire.
                view={k:record[k] for k in ('summary','history','row','spec')}
                item['feedback']=audit_feedback_symmetry_prefix(view)
                item['passed']=bool(record['row']['successful'] and generic['passed'] and item['feedback']['passed'])
                if not item['passed']: item['error']='One or more real-run/physical/prefix checks failed'
            except Exception:
                item['error']=traceback.format_exc()
            write(OUT/f'{label}-{seed}-audit.json',item)
            results.append(item)
            print(json.dumps(dict(seed=seed,strategy=label,passed=item['passed'],
                virtual_time_s=item.get('row',{}).get('virtual_time_s'),
                time_over_lower_bound=item.get('row',{}).get('time_over_lower_bound'),
                choice=item.get('feedback',{}).get('selected_id'),status=item.get('feedback',{}).get('status'),
                error=item.get('error')),ensure_ascii=False),flush=True)
            if not item['passed']:
                write(OUT/'summary.json',dict(all_passed=False,completed=len(results),results=results))
                return 1
    unchanged=hashes()==expected and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in contract.items())
    result=dict(all_passed=len(results)==6 and all(i['passed'] for i in results) and unchanged,
        completed=len(results),source_control_unchanged=unchanged,results=results,
        scope=manifest['scope'])
    write(OUT/'summary.json',result)
    return int(not result['all_passed'])

if __name__=='__main__': raise SystemExit(main())

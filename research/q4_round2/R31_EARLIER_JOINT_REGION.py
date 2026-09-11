"""Apply existing certified P/N geometry earlier, on 100 fixed old prefixes."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

CORE=Path(__file__).resolve().parents[2]
BASE=CORE.parent/'q4-r12-joint-continuation'
sys.path[:0]=[str(BASE/'src'),str(BASE)]
from planning.joint_visibility_region import joint_visibility_outer
from experiments.audit_q4_joint_visibility import audit_joint_visibility_certificate


def main():
    src=Path(__file__).with_name('R18_NEGATIVE_SKIP_DIAGNOSTIC.json')
    raw=src.read_bytes(); records=json.loads(raw)['events']
    assert len(records)==100
    all_evidence=[]; rows=[]
    for i,e in enumerate(records):
        outer,proof=joint_visibility_outer(e['canonical_vertices'],e['positive_positions'],e['negative_positions'])
        checked=audit_joint_visibility_certificate(proof)
        assert checked['passed'] is True
        row=dict(ordinal=i,seed=e['seed'],stage=e['stage'],channel=e['channel'],
                 prefix=e['after_actual_action_count'],status=proof['status'],
                 old_radius=proof['old_disk']['radius_m'],new_radius=proof['new_disk']['radius_m'],
                 became_ready=proof['became_ready'],independent_passed=True)
        rows.append(row);all_evidence.append(dict(summary=row,proof=proof,independent_check=checked))
    evidence=gzip.compress(json.dumps(all_evidence,sort_keys=True,allow_nan=False).encode(),mtime=0)
    proof_path=Path(__file__).with_name('R31_EARLIER_JOINT_REGION-proofs.json.gz')
    proof_path.write_bytes(evidence)
    ready={e['seed'] for e in rows if e['became_ready']}
    result=dict(scope='100 previously fixed R18 actual pre-measure prefixes, no current response or truth used',
                input_sha256=hashlib.sha256(raw).hexdigest(),
                source_sha256={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in [
                    'src/planning/joint_visibility_region.py','experiments/audit_q4_joint_visibility.py']},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                evidence_sha256=hashlib.sha256(evidence).hexdigest(),
                statuses=dict(Counter(e['status'] for e in rows)),
                new_ready_prefixes=sum(e['became_ready'] for e in rows),new_ready_cases=sorted(ready),
                proceed_to_implementation=len(ready)>=5,rows=rows)
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['statuses','new_ready_prefixes','new_ready_cases','proceed_to_implementation']}))


if __name__=='__main__':main()

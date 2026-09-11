"""Release a new metric measurement of the already qualified, unchanged R12.

This does not claim that a new development gate passed. Hashes bind historical
evidence and frozen bytes for reproducibility, not an adversarial signature.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_PATH = 'research/q4_per_source_reference/runtime-81aa6e1a.json'
ANCHOR_SHA256 = '056271898588e78d358fbac6e43a20770e170abe0b2ef3abc7148fbef02019c9'
BASE_COMMIT = '81aa6e1a1231a2358cf098fbba3fdd0557b540ef'
LABEL = 'compact_joint_continuation'
SPEC = dict(entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
            kwargs=dict(config='after_active_miss_optical', max_expansions=200))
SCHEMA = 'q4-qualified-reference-release-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(root, name, expected):
    path = (root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
        raise ValueError('Historical reference evidence changed or escapes root: '+name)
    return path


def verify_runtime(root=ROOT):
    root = Path(root).resolve()
    anchor = json.loads(checked(root, ANCHOR_PATH, ANCHOR_SHA256).read_bytes())
    if anchor.get('base_commit') != BASE_COMMIT:
        raise ValueError('Reference runtime anchor commit changed')
    expected = anchor['source_sha256']
    actual = {p.relative_to(root).as_posix(): sha(p) for p in (root/'src').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != expected:
        raise ValueError('All src bytes must equal the unchanged 81aa6e1a reference')
    return anchor


def historical_basis(root=ROOT):
    root = Path(root).resolve()
    anchor = verify_runtime(root)
    qpath = checked(root, anchor['qualification_path'], anchor['qualification_sha256'])
    spath = checked(root, anchor['selection_path'], anchor['selection_sha256'])
    qualification, selection = json.loads(qpath.read_bytes()), json.loads(spath.read_bytes())
    if (qualification.get('passed') is not True or qualification.get('selected') != LABEL
            or selection.get('role') != 'Frozen one candidate before independent cases'
            or selection.get('selected') != LABEL
            or selection.get('specs', {}).get(LABEL) != SPEC
            or qualification.get('selection_sha256') != anchor['selection_sha256']):
        raise ValueError('Original R12 qualification/selection identity is not satisfied')
    if {p:h for p,h in qualification['source_sha256'].items() if p.startswith('src/')} != {
            p:h for p,h in anchor['source_sha256'].items() if p.endswith('.py')}:
        raise ValueError('Qualified runtime differs from full original source identity')
    evidence = qualification.get('evidence_sha256')
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('Historical qualification lacks supporting evidence')
    for name, value in evidence.items():
        checked(root, name, value)
    for name, value in qualification['source_sha256'].items():
        checked(root, name, value)
    return dict(base_commit=BASE_COMMIT, runtime_anchor_path=ANCHOR_PATH,
        runtime_anchor_sha256=ANCHOR_SHA256, runtime_source_sha256=anchor['source_sha256'],
        qualification_path=anchor['qualification_path'], qualification_sha256=anchor['qualification_sha256'],
        selection_path=anchor['selection_path'], selection_sha256=anchor['selection_sha256'],
        historical_evidence_sha256=evidence,
        basis='Previously qualified R12; new count-stratified metric measurement, no new development selection')


def build_release(plan, plan_sha256, root=ROOT):
    if plan['split'] not in {'confirmation', 'stress'} or plan['label'] != LABEL or plan['spec'] != SPEC:
        raise ValueError('Only fixed R12 confirmation/stress reference measurements are authorized')
    if not isinstance(plan_sha256,str) or len(plan_sha256)!=64 or any(c not in '0123456789abcdef' for c in plan_sha256):
        raise ValueError('Need the exact plan file SHA256')
    basis = historical_basis(root)
    if any(plan['source_sha256'].get(p) != h for p,h in basis['runtime_source_sha256'].items()):
        raise ValueError('Plan does not preserve the qualified runtime')
    return dict(schema=SCHEMA, authorized=True, plan_sha256=plan_sha256,
        split=plan['split'], label=LABEL, spec=SPEC, source_sha256=plan['source_sha256'],
        reserved_seeds=plan['seed_selection']['seeds'], reference_basis=basis)


def validate_release(plan, plan_sha256, release, root=ROOT):
    if release != build_release(plan, plan_sha256, root):
        raise ValueError('Reference release differs from historical qualification or exact new plan')

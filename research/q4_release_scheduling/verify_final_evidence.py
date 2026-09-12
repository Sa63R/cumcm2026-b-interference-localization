"""Read-only final byte verification; no simulation or audit rerun."""
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.run_q4_per_source import source_hashes

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research/q4_release_scheduling'
PUBLISHED = 'f0146293f8c08cff1115592fd7e10ae9338ca946'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blobs(ref, paths):
    data = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
        input=''.join(ref+':'+p+'\n' for p in paths).encode())
    offset, result = 0, {}
    for path in paths:
        end = data.index(b'\n', offset)
        _, kind, length = data[offset:end].split()
        assert kind == b'blob'
        offset = end + 1
        length = int(length)
        result[path] = data[offset:offset+length]
        offset += length + 1
    assert offset == len(data)
    return result


if __name__ == '__main__':
    freeze = json.loads((RESEARCH/'source-freeze.json').read_bytes())
    identity = freeze['source_sha256']
    assert identity == source_hashes() and len(identity) == 71
    expected = {**identity, **freeze['plan_sha256'], **freeze['old_qa_files_sha256']}
    expected['research/q4_release_scheduling/source-freeze.json'] = 'fbb1709871ffca92c98cfc9ee1eea85f3209046397c54f4862271dd35467067d'
    published = blobs(PUBLISHED, sorted(expected))
    for name, digest in expected.items():
        assert sha((ROOT/name).read_bytes()) == digest == sha(published[name]), name
    base = freeze['base_commit']
    base_paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', base, 'src'], cwd=ROOT).decode().splitlines()
    assert len(base_paths) == 49
    for name, data in blobs(base, base_paths).items():
        assert data == (ROOT/name).read_bytes(), name
    read_set = {}
    stages = {}
    for stage, count in [('development',70), ('development-stress',49)]:
        folder = ROOT/'results/q4_release_scheduling'/stage
        audit = json.loads((folder/'independent_audit.json').read_bytes())
        assert audit['records'] == audit['passed_records'] == count and audit['all_passed']
        for name, digest in audit['input_sha256'].items():
            assert sha((folder/name).read_bytes()) == digest, name
            read_set[(folder/name).relative_to(ROOT).as_posix()] = digest
        read_set[(folder/'independent_audit.json').relative_to(ROOT).as_posix()] = sha((folder/'independent_audit.json').read_bytes())
        stages[stage] = {'records':count,'first_audit_passed':True}
    selection = json.loads((RESEARCH/'selection.json').read_bytes())
    assert not selection['passed']
    assert selection['criteria']['complete_and_audited']
    assert all(read_set[name] == digest for name,digest in selection['read_set_sha256'].items())
    for name in ('selection.json','development-analysis.json','analyze_development.py','RESULTS.md'):
        read_set[(RESEARCH/name).relative_to(ROOT).as_posix()] = sha((RESEARCH/name).read_bytes())
    result = dict(passed=True, published_commit=PUBLISHED, runtime_files=len(identity),
        base_files=len(base_paths), verified_published_paths=len(expected),
        source_sha256=identity, checked_published_sha256=expected, input_sha256=read_set,
        stages=stages, selection_passed=False, independent_opened=False, official_run=False)
    with (RESEARCH/'final-evidence-verification.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_sha256')}))

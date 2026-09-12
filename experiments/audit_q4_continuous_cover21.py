"""Read-only byte/point/witness audit of R38 outputs; never run a new certificate."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'research/q4_continuous_cover21'
BASE = '81aa6e1a1231a2358cf098fbba3fdd0557b540ef'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def audit():
    rows = [json.loads((HERE/'initial.json').read_bytes())]
    rows += [json.loads((HERE/f'candidate-{i:02d}.json').read_bytes()) for i in range(1, 13)]
    protocol_sha = sha((HERE/'PROTOCOL.md').read_bytes())
    latest_script = sha((ROOT/'experiments/research_q4_continuous_cover21.py').read_bytes())
    initial_script = sha((HERE/'initial-driver-source.py').read_bytes())
    workspaces_seen = set()
    details = []
    for index, row in enumerate(rows):
        packed = (HERE/row['proof_file']).read_bytes()
        raw = gzip.decompress(packed)
        assert sha(packed) == row['proof_gzip_sha256']
        assert sha(raw) == row['proof_raw_sha256']
        proof = json.loads(raw)
        assert sorted(row['points']) == proof['stations']
        assert len(row['points']) == len(set(tuple(p) for p in row['points'])) == 21
        assert row['points'][0] == [0., 0.]
        assert sha(json.dumps(sorted(row['points']), separators=(',', ':')).encode()) == proof['station_sha256']
        assert proof['max_depth'] == 24 and proof['max_cells'] == 800000
        assert proof['range_margin_m'] == 1e-5 and proof['orientation_margin_m'] == 1e-7
        assert proof['reception_radius'] == 1000. and proof['arena_radius'] == 1800.
        assert row['certificate_summary'] == {k: v for k, v in proof.items() if k not in ('stations', 'leaves')}
        assert row['identity']['protocol_sha256'] == protocol_sha
        assert row['identity']['script_sha256'] == (initial_script if index == 0 else latest_script)
        min_guard = None
        if proof['counterexample']:
            x, n = proof['counterexample']['source'], proof['counterexample']['normal']
            assert math.hypot(*x) <= 1800. and abs(math.hypot(*n)-1.) < 1e-10
            guards = []
            for p in row['points']:
                d = math.dist(x, p)
                dot = n[0]*(p[0]-x[0])+n[1]*(p[1]-x[1])
                # Strict exclusion is a disjunction per individual station.
                assert d > 1000.+1e-5 or dot < -1e-5
                guards.append(max(d-1000., -dot))
            min_guard = min(guards)
            assert row['independent_counterexample']['passed']
        else:
            assert proof['status'] == 'inconclusive' and not proof['passed']
            assert proof['reason'] == 'depth_budget' and proof['unresolved_cell']
            assert len(proof['unresolved_cell']['path']) == 24
        if index:
            context = row['context']
            working_raw = (HERE/context['working_set_file']).read_bytes()
            assert sha(working_raw) == context['working_set_sha256']
            constraints = json.loads(working_raw)['constraints']
            independent = min(max(min(1000.-math.dist(p, r['source']),
                sum(r['normal'][k]*(p[k]-r['source'][k]) for k in (0, 1))) for p in row['points'])
                for r in constraints)
            assert abs(independent-context['working_margin_m']) < 1e-8
            assert context['objective_calls'] <= 8000
            assert len(context['optimization_runs']) <= 3
            assert len(context['local_native_indices']) == (0 if index <= 6 else 2)
            for value, bound in zip(context['parameters'], context['bounds']):
                assert bound[0] <= value <= bound[1]
            # Every constraint origin must precede this proposal's certificate.
            assert all(r['origin'] in workspaces_seen for r in constraints)
        workspaces_seen.add(row['name'])
        details.append(dict(name=row['name'], status=proof['status'], visited_cells=proof['visited_cells'],
                            strict_exclusion_margin_m=min_guard, proof_raw_sha256=sha(raw)))
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--', 'src'], cwd=ROOT).decode().splitlines()
    assert len(paths) == 49
    base_hashes = {}
    for path in paths:
        blob = subprocess.check_output(['git', 'show', BASE+':'+path], cwd=ROOT)
        assert (ROOT/path).read_bytes() == blob, path
        base_hashes[path] = sha(blob)
    summary = json.loads((HERE/'optimization-summary.json').read_bytes())
    assert summary['candidates'] == 12 and summary['stop_reason'] == 'twelve_certificate_candidates_exhausted'
    assert summary['completed_at_unix'] < 1789178377.
    return dict(passed=True, scope='Stored bytes, 13 pointsets, 10 strict witnesses, finite margins, 49 baseline blobs; no new proof search',
        certificates=13, counterexamples=10, inconclusive=3, certified=0, records=details,
        original_base=BASE, baseline_source_sha256=base_hashes, protocol_sha256=protocol_sha,
        initial_driver_sha256=initial_script, optimizer_driver_sha256=latest_script,
        audit_script_sha256=sha(Path(__file__).read_bytes()))


if __name__ == '__main__':
    result = audit()
    with (HERE/'independent-record-audit.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'baseline_source_sha256')}))

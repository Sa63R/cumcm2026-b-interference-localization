"""Refine four pinned R30 public layouts; no cases, histories or policies."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
R12 = WORKSPACE / 'q4-r12-joint-continuation'
sys.path.insert(0, str(R12 / 'src'))
from planning.q4_directional_cover import certify_directional_cover, verify_directional_cover_certificate
from simulator_client.state import Position

INPUT_SHA = '6ee1259d4748b2a0b50c0d4b1225ed60455fb537c2e14e8f346966b35c985c6c'
INDICES = (4, 5, 6, 13)
BUDGET = dict(arena_radius=1800., reception_radius=1000., max_depth=20,
              max_cells=800000, range_margin_m=1e-5, orientation_margin_m=1e-7)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def check_witness(points, witness):
    if witness is None:
        return None
    x, n = witness['source'], witness['normal']
    assert math.hypot(*x) <= 1800. and abs(math.hypot(*n) - 1.) < 1e-8
    dots, far = [], []
    for p in points:
        distance = math.hypot(p.x - x[0], p.y - x[1])
        dot = n[0] * (p.x - x[0]) + n[1] * (p.y - x[1])
        if distance <= 1000. + 1e-5:
            assert dot < -1e-5
            dots.append(dot)
        else:
            far.append(distance - 1000.)
    return dict(passed=True, source_radius_m=math.hypot(*x),
                receivable_range_station_count=len(dots),
                max_local_projection_m=max(dots) if dots else None,
                min_unreachable_distance_margin_m=min(far) if far else None)


def run():
    dest, proofdir = HERE / 'R32_REFINE_DIAGNOSTIC.json', HERE / 'R32_PROOFS'
    if dest.exists() or proofdir.exists():
        raise ValueError('Preserve existing R32 evidence')
    raw_input = (HERE / 'R30_ASYMMETRIC_COUNT_DIAGNOSTIC.json').read_bytes()
    assert sha(raw_input) == INPUT_SHA
    old = json.loads(raw_input)
    assert tuple(i for i, r in enumerate(old['results'])
                 if r['certificate']['status'] == 'inconclusive') == INDICES
    for relative, expected in old['source_sha256'].items():
        assert sha((WORKSPACE / relative).read_bytes()) == expected
    for index in INDICES:
        row = old['results'][index]
        prior_gz = (HERE / row['proof_file']).read_bytes()
        prior_raw = gzip.decompress(prior_gz)
        assert sha(prior_gz) == row['proof_gzip_sha256']
        assert sha(prior_raw) == row['proof_raw_sha256']
        prior = json.loads(prior_raw)
        assert prior['stations'] == sorted(row['ordered_stations'])
        assert prior['status'] == 'inconclusive' and prior['reason'] == 'depth_budget'
    protocol_sha = sha((HERE / 'R32_PROTOCOL.md').read_bytes())
    proofdir.mkdir()
    rows = []
    for index in INDICES:
        oldrow = old['results'][index]
        points = tuple(Position(*xy) for xy in oldrow['ordered_stations'])
        proof = certify_directional_cover(points, **BUDGET)
        assert proof['station_sha256'] == oldrow['certificate']['station_sha256']
        verified = verify_directional_cover_certificate(points, proof) if proof['passed'] else None
        if proof['passed']:
            assert verified['passed']
        witness = check_witness(points, proof['counterexample'])
        encoded = encode(proof)
        compressed = gzip.compress(encoded, mtime=0)
        name = Path(oldrow['proof_file']).name
        (proofdir / name).write_bytes(compressed)
        rows.append(dict(r30_index=index, r30_proof_file=oldrow['proof_file'],
            r30_proof_raw_sha256=oldrow['proof_raw_sha256'],
            r30_proof_gzip_sha256=oldrow['proof_gzip_sha256'],
            inner_count=oldrow['inner_count'], outer_count=oldrow['outer_count'],
            inner_radius_m=oldrow['inner_radius_m'], outer_radius_m=oldrow['outer_radius_m'],
            outer_half_step_phase=oldrow['outer_half_step_phase'],
            ordered_stations=oldrow['ordered_stations'],
            native_station_route_ids=oldrow['native_station_route_ids'],
            route_length_m=oldrow['route_length_m'], pure_movement_s=oldrow['pure_movement_s'],
            certificate={k: v for k, v in proof.items() if k not in ('leaves', 'stations', 'runtime_s')},
            independent_leaf_verification=verified, independent_counterexample_check=witness,
            certificate_runtime_s=proof['runtime_s'], proof_file=proofdir.name + '/' + name,
            proof_raw_sha256=sha(encoded), proof_gzip_sha256=sha(compressed)))
        print(json.dumps(dict(r30_index=index, status=proof['status'],
                              cells=proof['visited_cells'], depth=proof['deepest_level'])), flush=True)
    assert sha((HERE / 'R32_PROTOCOL.md').read_bytes()) == protocol_sha
    output = dict(scope='Refinement of four pinned R30 layouts; no new layout or case',
        r30_input_sha256=INPUT_SHA, protocol_sha256=protocol_sha,
        script_sha256=sha(Path(__file__).read_bytes()), source_sha256=old['source_sha256'],
        certificate_budget=BUDGET, results=rows,
        summary=dict(attempted=len(rows), passed=sum(r['certificate']['passed'] for r in rows),
                     counterexamples=sum(r['certificate']['status'] == 'counterexample' for r in rows),
                     inconclusive=sum(r['certificate']['status'] == 'inconclusive' for r in rows)))
    with dest.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(output['summary']))


if __name__ == '__main__':
    run()

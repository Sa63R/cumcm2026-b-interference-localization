"""Check pairing, cost identities, completion, and serial timing consistency."""
from pathlib import Path
import json
import summarize as s


def main():
    rows = s.read('annex_all400.csv')
    assert len(rows) == 3600
    assert all(r['success'] and r['cleared'] == r['true_sources'] and not r['error'] for r in rows)
    assert len({(r['seed'], r['mode']) for r in rows}) == 3600
    for mode in s.LABELS:
        assert {r['seed'] for r in rows if r['mode'] == mode} == set(range(5000, 5400))
    lookup = {(r['seed'], r['mode']): r for r in rows}
    cost_error = max(abs(r['virtual_seconds'] - (r['movement_metres']/5 + r['detects']*5 + r['switches']
                     + r['optical_attempts']*3 + r['cleared']*2)) for r in rows)
    assert cost_error < 1e-8
    runtime = s.read('runtime50.csv')
    assert len(runtime) == 900
    timing_difference = max(abs(r['virtual_seconds'] - lookup[r['seed'], r['mode']]['virtual_seconds']) for r in runtime)
    assert timing_difference < 1e-6
    stress = s.read('stress350.csv')
    assert len(stress) == 3150 and all(r['success'] for r in stress)
    assert all(r['failed_clears'] == 0 for r in rows + stress if r['mode'] != 'optical')
    validation = dict(paired_cases=400, methods=9, random_runs=len(rows), stress_cases=350,
                      stress_runs=len(stress), runtime_runs=len(runtime), all_clear=True,
                      max_cost_identity_error=cost_error, max_runtime_vs_batch_virtual_difference=timing_difference)
    (s.RESULTS / 'final_validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()

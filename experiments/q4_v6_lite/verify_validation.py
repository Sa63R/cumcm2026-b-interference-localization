"""Replay every public response and verify the separate serial timing sample."""
import argparse
import gzip
import hashlib
import json
import statistics
import time
import candidates as c
from candidates import bench
from jammers_local.__main__ import replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['replay', 'serial'])
    args = parser.parse_args()
    root = c.ROOT
    plan = json.loads((root / 'plan.json').read_text())
    rows = [json.loads(line) for line in (root / 'main/records.jsonl').read_text().splitlines()]
    index = {(r['case_key'], r['method']): r for r in rows}
    expected = {(case['key'], method) for case in plan['cases'] for method in c.VARIANTS}
    assert len(index) == len(rows) == len(expected) and set(index) == expected
    if args.phase == 'replay':
        results = []
        checked = set()
        started = time.perf_counter()
        for i, path in enumerate(sorted((root / 'main/sessions').glob('*.json.gz'))):
            with gzip.open(path, 'rt') as stream:
                session = json.load(stream)
            row = session['row']
            key = (row['case_key'], row['method'])
            assert key not in checked and row == index[key]
            checked.add(key)
            result = replay(session)
            canonical = [dict(path=r['path'], request=r['request'],
                             response={k: v for k, v in r['response'].items()
                                       if k not in ('real_timestamp_ms', 'remaining_real_duration_s')})
                         for r in session['history']]
            assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == row['action_sha256']
            assert result['matched']
            results.append(dict(file=path.name, **result))
            if (i + 1) % 300 == 0:
                print('replayed', i + 1, 'all matched', flush=True)
        assert checked == expected
        bench.dump(root / 'replay_verification.json', dict(runs=len(results), all_matched=True,
                   actions=sum(r['compared_actions'] for r in results),
                   elapsed_seconds=time.perf_counter() - started, records=results))
        return
    serial = [json.loads(line) for line in (root / 'serial/records.jsonl').read_text().splitlines()]
    expected_serial = {(case['key'], method) for case in plan['cases'][:50] for method in c.VARIANTS}
    assert len(serial) == len(expected_serial) and {(r['case_key'], r['method']) for r in serial} == expected_serial
    for row in serial:
        original = index[row['case_key'], row['method']]
        assert row['error'] is None and row['all_cleared']
        assert row['action_sha256'] == original['action_sha256']
        assert row['virtual_time_us'] == original['virtual_time_us']
    summary = {method: {field: statistics.mean(r[field] for r in serial if r['method'] == method)
                       for field in ['cpu_seconds', 'wall_seconds', 'planner_seconds']}
               for method in c.VARIANTS}
    bench.dump(root / 'serial/summary.json', dict(cases=50, runs=len(serial), all_trace_hashes_equal=True,
               methods=summary,
               cpu_reduction_percent=100 * (1 - summary['lite']['cpu_seconds'] / summary['full']['cpu_seconds']),
               wall_reduction_percent=100 * (1 - summary['lite']['wall_seconds'] / summary['full']['wall_seconds'])))


if __name__ == '__main__':
    main()

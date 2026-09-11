"""Audit the explicitly scoped 118001..118016 saved training pilot; no simulation."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path

from research.theory_gap_v1.audit_gaps import helpers, PINNED_HELPERS, sha


def audit(directory):
    directory = Path(directory)
    workspace = Path(__file__).resolve().parents[2]
    physical, _ = helpers(workspace/'q3-state-search/research/theory_v1')
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['phase'] == 'probe-swap-pilot'
    assert manifest['seeds'] == list(range(118001, 118017))
    result = dict(scope='saved training only; truth accessed only for post-run physical audit',
                  helper_sha256=PINNED_HELPERS, runs=0, exact_history_ties=0, events=[])
    for seed in manifest['seeds']:
        records = []
        for variant in ('geometric_probe_single', 'geometric_probe_swap'):
            path = directory/'cases'/variant/f'case-{seed}.json.gz'
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
            assert record['row']['seed'] == seed and record['row']['strategy'] == variant
            physical.audit_record(record)
            assert record['row']['successful'] and record['row']['failed_clear_count'] == 0
            result['runs'] += 1
            records.append(record)
        old, new = records
        assert old['row']['case_sha256'] == new['row']['case_sha256']
        old_history = old['summary']['action_history']
        history = new['summary']['action_history']
        if old_history == history:
            result['exact_history_ties'] += 1
        interruptions = new['summary']['source_interruptions']
        assert max(interruptions['probe_counts'].values(), default=0) <= 6
        assert len({e['channel'] for e in interruptions['events']}) == len(interruptions['events'])
        for event in interruptions['events']:
            forced, swapped = event['forced_route'], event['swapped_route']
            assert Counter(map(tuple, forced)) == Counter(map(tuple, swapped))
            assert swapped[1] == forced[0]
            assert swapped[2:] == [t for t in forced if t not in swapped[:2]]
            length = lambda route: sum(math.dist(a, b) for a, b in zip(
                [event['position']]+[t[2:] for t in route], [t[2:] for t in route]))
            assert math.isclose(length(forced), event['forced_length_m'], abs_tol=1e-8)
            assert math.isclose(length(swapped), event['swapped_length_m'], abs_tol=1e-8)
            gain = (length(forced)-length(swapped))/5
            extra = 5. if swapped[0][0] == 'cover' else 0.
            assert math.isclose(event['score_s'], gain-extra-2., abs_tol=1e-9)
            assert event['score_s'] > 10.
            start, end = event['other_start_actions'], event['other_end_actions']
            assert start == event['accepted_actions'] and end > start
            assert end == event['resume_start_actions']
            other = history[start-1:end-1]  # accepted counter includes enter
            before = history[:start-1]
            if swapped[0][0] == 'cover':
                cleared = {a['channel'] for a in before if a['action'] == 'clear' and a['result'] == 'success'}
                assert {a['channel'] for a in other} == set(range(1, 21))-cleared
                assert all(a['action'] == 'measure' and a['phase'] == 'coverage'
                           and a['position'] == swapped[0][2:] for a in other)
            else:
                assert any(a['action'] == 'clear' and a['channel'] == swapped[0][1]
                           and a['result'] == 'success' for a in other)
            resumed = history[end-1:event['locked_final_actions']-1]
            assert any(a['action'] == 'clear' and a['channel'] == event['channel']
                       and a['result'] == 'success' for a in resumed)
            assert event['primary_cleared_final']
            assert event['resume_primary_probes_before'] >= event['primary_probes']
            known_before = {a['channel'] for a in before if a['action'] == 'measure'
                            and a['result'] != 'no_signal'}
            discovered = sorted({a['channel'] for a in other if a['action'] == 'measure'
                                 and a['result'] != 'no_signal'}-known_before)
            result['events'].append(dict(seed=seed, event=event,
                other_newly_discovered_channels=discovered,
                delta_candidate_minus_single_s={k: new['row'][k]-old['row'][k] for k in
                    ('virtual_time_s', 'movement_s', 'detection_s', 'switching_s', 'optical_s', 'removal_s')},
                same_source_clear_order=[a['channel'] for a in old_history if a['action'] == 'clear'] ==
                                        [a['channel'] for a in history if a['action'] == 'clear'],
                candidate_input_sha256=sha(directory/'cases/geometric_probe_swap'/f'case-{seed}.json.gz')))
    return result


if __name__ == '__main__':
    folder = Path('results/geometric_joint/probe_swap_pilot_v1')
    result = audit(folder)
    (folder/'swap_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(runs=result['runs'], exact_history_ties=result['exact_history_ties'],
                         events=len(result['events']))))

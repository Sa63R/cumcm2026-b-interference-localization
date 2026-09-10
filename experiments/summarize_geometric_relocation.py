"""Read saved authorized training records; audit fees and real cover evidence.

The source-truth fee auditor is loaded read-only at the previously pinned state
theory hashes. No simulation, source-route DP or final/extended scene is opened.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from planning import nearest_order
from planning.coverage import omni_coverage_points
from planning.disk_cover import disk_cover_radius
from research.theory_gap_v1.audit_gaps import helpers, PINNED_HELPERS
from simulator_client.state import Position


def _xy(p):
    return [p.x,p.y]


def audit_cover(report):
    history = report['action_history']
    records = report['strategy_parameters']['relocation_log']
    by_count = {r['after_actual_action_count']:r for r in records}
    assert len(by_count) == len(records)
    points = nearest_order(omni_coverage_points(1150.))
    origin = _xy(points[0])
    sites = [dict(id=i,position=_xy(p),planned_changes=0,visited=False,
                  source_plans_before_visit=0) for i,p in enumerate(points[1:])]
    remaining, visited, changes = list(sites),[],[]
    known, cleared, negatives, cover_pairs = set(),set(),set(),{}
    for index in range(len(history)+1):
        record = by_count.get(index)
        if record is not None:
            assert record['executed_discovery_stations'] == visited
            assert record['remaining_before'] == [s['position'] for s in remaining]
            assert all((c,tuple(p)) in negatives for c in set(range(1,21))-known for p in visited)
            assert Counter(record['baseline_order']) == Counter(record['selected_order'])
            assert len(record['evaluated']) <= 4
            tasks = record['frozen_tasks']
            assert len(tasks) == len(record['baseline_order'])
            current = Position.coerce(history[index-1]['position']) if index else Position(0.,0.)
            baseline = [Position(t[2],t[3]) for t in tasks]
            def route_cost(locations,order):
                route = [current]+[locations[i] for i in order]
                return sum(p.distance_to(q) for p,q in zip(route,route[1:]))/5.
            assert abs(route_cost(baseline,record['baseline_order'])-record['baseline_proxy_s']) < 1e-6
            selected = baseline.copy()
            if record['relocated']:
                different = [i for i,(a,b) in enumerate(zip(record['remaining_before'],record['remaining_after'])) if a != b]
                assert len(different) == 1
                position_index = different[0]
                site = remaining[position_index]
                assert site['position'] == record['old_position']
                selected[position_index] = Position.coerce(record['new_position'])
                site['position'] = record['new_position']
                site['planned_changes'] += 1
                assert [s['position'] for s in remaining] == record['remaining_after']
                radius = disk_cover_radius(visited+[s['position'] for s in remaining])
                assert radius <= 1000.-1e-5
                assert abs(radius-record['certified_cover_radius_m']) < 1e-8
                changes.append(dict(site_id=site['id'],distance_m=Position.coerce(record['old_position']).distance_to(selected[position_index]),
                    frozen_proxy_gain_s=record['baseline_proxy_s']-record['selected_proxy_s'],coverage_radius_m=radius))
            assert abs(route_cost(selected,record['selected_order'])-record['selected_proxy_s']) < 1e-6
            assert record['selected_proxy_s'] <= record['baseline_proxy_s']+1e-6
            if record['selected_kind'] == 'cover':
                assert index < len(history) and history[index]['phase'] == 'coverage'
                assert history[index]['position'] == record['selected_point']
            else:
                for site in remaining:
                    site['source_plans_before_visit'] += 1
        if index == len(history):
            break
        action = history[index]
        channel, point = action['channel'],action['position']
        if action['action'] == 'measure':
            if action['result'] in ('direction','near'):
                known.add(channel)
            else:
                negatives.add((channel,tuple(point)))
            if action['phase'] == 'coverage':
                key = tuple(point)
                scanned = cover_pairs.setdefault(key,set())
                scanned.add(channel)
                # Promote a station only after all channels have actual
                # coverage scans or earlier successful unique-source clears.
                if scanned|cleared == set(range(1,21)) and point not in visited:
                    if point != origin:
                        site = next(s for s in remaining if s['position'] == point)
                        site['visited'] = True
                        remaining.remove(site)
                    visited.append(point)
        elif action['result'] == 'success':
            cleared.add(channel)
    assert len(visited) == report['coverage_points_visited']
    assert Counter(map(tuple,report['coverage_points'])) == Counter(map(tuple,visited+[s['position'] for s in remaining]))
    assert not remaining or len(cleared) == 16
    if len(cleared) < 16:
        for channel in set(range(1,21))-known:
            actual = [p for c,p in negatives if c == channel]
            assert disk_cover_radius(actual) <= 1000.-1e-5
    calls = sum(r['oracle_calls'] for r in records)
    assert calls <= 12000
    return dict(sites=sites,relocations=changes,oracle_calls=calls,
                oracle_budget_exhaustions=sum(r['oracle_budget_exhausted'] for r in records),
                planning_s=sum(r['planning_s'] for r in records),
                future_sites_canceled_only_after_16_clears=len(remaining),
                all_actual_channel_coverage_and_task_multiset_checks_passed=True)


def timing(report):
    discovered = {}
    clear_order = []
    for index,a in enumerate(report['action_history']):
        if a['action']=='measure' and a['result'] in ('direction','near') and a['channel'] not in discovered:
            discovered[a['channel']] = dict(position=a['position'],time_s=a['virtual_time_s'],action_index=index)
        if a['action']=='clear' and a['result']=='success':
            clear_order.append(a['channel'])
    return discovered,clear_order


def summarize(directory,theory_dir):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['phase'] not in ('probe-relocation-pilot','probe-relocation-confirmation'):
        raise ValueError('Only authorized geometric relocation training phases are in scope')
    allowed = list(range(116001,116017)) if manifest['phase'].endswith('pilot') else list(range(116017,116081))
    if manifest['seeds'] != allowed:
        raise ValueError('Unexpected training seed manifest')
    names = ('geometric_probe_single','geometric_probe_relocation')
    if {s['name'] for s in manifest['specs']} != set(names):
        raise ValueError('Expected single/relocation-only contrast')
    base,_ = helpers(theory_dir)
    cases = []
    for seed in allowed:
        loaded = []
        for name in names:
            path = directory/'cases'/name/f'case-{seed}.json.gz'
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                record=json.load(stream)
            base.audit_record(record)
            assert record['row']['successful'] and record['row']['failed_clear_count']==0
            loaded.append(record)
        old,new=loaded
        assert old['row']['case_sha256']==new['row']['case_sha256']
        coverage=audit_cover(new['summary'])
        old_discovery,old_order=timing(old['summary'])
        new_discovery,new_order=timing(new['summary'])
        cases.append(dict(seed=seed,saved_s=old['row']['virtual_time_s']-new['row']['virtual_time_s'],
            component_delta_s={k:new['row'][k]-old['row'][k] for k in
                ('movement_s','detection_s','switching_s','optical_s','removal_s')},
            same_clear_order=old_order==new_order,old_clear_order=old_order,new_clear_order=new_order,
            changed_first_detection=[dict(channel=c,before=old_discovery[c],after=new_discovery[c])
                for c in old_discovery if old_discovery[c]['position']!=new_discovery[c]['position']],
            active_measurement_delta=sum(a['phase']=='active_localization' for a in new['summary']['action_history'])-
                                     sum(a['phase']=='active_localization' for a in old['summary']['action_history']),
            physical_cost_audit=True,coverage_audit=coverage))
    sites=[s for c in cases for s in c['coverage_audit']['sites']]
    changes=[r for c in cases for r in c['coverage_audit']['relocations']]
    result=dict(phase=manifest['phase'],cases=len(cases),physical_records_audited=2*len(cases),
        helper_sha256=PINNED_HELPERS,source_helper_directory=str(theory_dir),
        all_full_clear_zero_failed_and_physical_coverage_audits_passed=True,
        relocation_count=len(changes),mean_relocation_distance_m=statistics.mean(r['distance_m'] for r in changes),
        max_relocation_distance_m=max(r['distance_m'] for r in changes),
        mean_oracle_calls=statistics.mean(c['coverage_audit']['oracle_calls'] for c in cases),
        max_oracle_calls=max(c['coverage_audit']['oracle_calls'] for c in cases),
        oracle_budget_exhaustions=sum(c['coverage_audit']['oracle_budget_exhaustions'] for c in cases),
        mean_planning_s=statistics.mean(c['coverage_audit']['planning_s'] for c in cases),
        actual_future_visits=sum(s['visited'] for s in sites),
        replanned_then_actually_visited=sum(s['visited'] and s['planned_changes']>0 for s in sites),
        future_sites_canceled_only_after_16_clears=sum(c['coverage_audit']['future_sites_canceled_only_after_16_clears'] for c in cases),
        max_changes_to_one_future_site=max(s['planned_changes'] for s in sites),
        losses_with_same_clear_order=sum(c['saved_s']<0 and c['same_clear_order'] for c in cases),
        caveat='Frozen plan improvements overlap; full cover need not imply timely discovery. Timing differences are descriptive, not counterfactual causal effects.',
        details=cases)
    (directory/'relocation_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('details','helper_sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--theory-dir',type=Path,default=Path(__file__).resolve().parents[2]/'q3-state-search/research/theory_v1')
    args=parser.parse_args()
    summarize(args.directory,args.theory_dir)

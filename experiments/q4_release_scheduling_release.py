"""Select the single R36 candidate only after both complete development batches.

Completed failed cases and failed audits remain in the read set and disqualify
their candidate. Missing or inconsistent evidence fails closed. No case/policy
is run here. Local hashes bind artifacts; they are not adversarial signatures.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXED_SPECS = {'compact_release_scheduling': {'entrypoint': 'strategies.q4_release_scheduling:run_q4_release_scheduling',
    'kwargs': {'config': 'nominal_release', 'max_expansions': 200}}}
MIN_TRIGGERED_CASES = 5
SPLITS = ('development', 'development-stress')
DEVELOPMENT_LIMIT = 500.


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, path):
    path = (root / path).resolve()
    require(path.is_relative_to(root), 'Evidence path escapes root')
    return path


def audit_item_passed(item):
    release_scheduling = item.get('release_scheduling', {})
    prefix = release_scheduling.get('prefix', {})
    return (release_scheduling.get('passed') is True and release_scheduling.get('generic', {}).get('passed') is True and
            prefix.get('passed') is True and all(prefix.get(k, {}).get('passed') is True
                for k in ('r12', 'r8', 'range', 'scheduling')))


def actual_service_count(record):
    """Executed mixed-ready/forecast macros, never predicted clear successes."""
    wire=[a for a in record['history'] if a['action'] in ('/measure','/clear') and a['response'].get('accepted') is True]
    actions=record['summary']['action_history']
    require(len(wire)==len(actions),'Wire/summary accepted action count differs')
    for action,real in zip(actions,wire):
        require(action['action'] in ('measure','clear') and real['action']=='/'+action['action'],'Action differs from wire')
        point=real['position']; point=[point['x'],point['y']] if isinstance(point,dict) else list(point)
        require(action['channel']==real['channel'] and list(action['position'])==point,'Point/channel differs from wire')
        require(action['result']==real['response'][action['action']+'_result'],'Response differs from wire')
        require(action['virtual_time_s']==real['response']['virtual_time_s'],'Clock differs from wire')
    events=record['summary']['strategy_parameters']['release_scheduling_plan_log']
    require(isinstance(events,list),'Missing release plan log')
    owned,count=set(),0
    for event in events:
        covers=event['remaining_covers']; channels=event['source_channels']
        evidence=event['source_evidence']; releases=event['release_indices']
        require(all(isinstance(x,list) for x in (covers,channels,evidence,releases)),'Missing forecast task arrays')
        require(len(channels)==len(evidence)==len(releases) and len(set(channels))==len(channels),'Forecast task dimensions differ')
        ready=[]
        for channel,item,release in zip(channels,evidence,releases):
            require(type(channel) is int and 1<=channel<=20 and item['channel']==channel and type(item['ready']) is bool,'Invalid actual source readiness')
            require(type(release) is int and 0<=release<=len(covers),'Invalid release prefix')
            require((release==0) if item['ready'] or not covers else release>=1,'Forecast opens a nonready source immediately')
            ready.append(item['ready'])
        start,end=event['after_actual_action_count'],event['end_actual_action_count']
        require(type(start) is int and type(end) is int and 0<=start<=end<=len(actions),'Invalid plan action range')
        kind,selected=event['selected_kind'],event['selected_channel']
        require(kind in ('source','cover'),'Unknown actual macro kind')
        if kind=='source':
            require(type(selected) is int and selected in channels and (not covers or ready[channels.index(selected)]),'Actual first source was not ready')
        else:
            require(bool(covers) and selected is None,'Cover macro lacks its station')
        for i in range(start,end):
            require(i not in owned,'Overlapping plan macros')
            owned.add(i); action=actions[i]
            if kind=='source':
                require(action['channel']==selected and action['phase']!='coverage','Source macro owns another task')
            else:
                require(action['action']=='measure' and action['phase']=='coverage' and list(action['position'])==covers[0],'Cover macro differs from actual scan')
        if end>start and any(ready) and not all(ready) and covers:
            count+=1
    return count


def inspect_development(directory, split, label, root):
    from experiments import run_q4_per_source as runner
    root = Path(root).resolve()
    directory = inside(root, directory)
    require(split in SPLITS and label in FIXED_SPECS, 'Unknown fixed development role')
    plan = read(directory/'plan.json')
    runner.validate_plan(plan)
    require(plan['split'] == split and plan['label'] == label and plan['spec'] == FIXED_SPECS[label], 'Development split or fixed candidate differs')
    manifest, freeze = read(directory/'manifest.json'), read(directory/'freeze.json')
    require(manifest == dict(plan=plan, plan_sha256=sha(directory/'plan.json'), release_sha256=None), 'Development manifest differs')
    require(not (directory/'release.json').exists(), 'Development may not contain an independent release')
    require(freeze['manifest_sha256'] == runner.digest(manifest), 'Development freeze differs')
    require(isinstance(freeze.get('git_commit'), str) and len(freeze['git_commit']) == 40, 'Missing development source commit')
    with zipfile.ZipFile(directory/'source.zip') as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())) and
                set(archive.namelist()) == set(plan['source_sha256']), 'Frozen source archive membership differs')
        for name, expected in plan['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'Frozen source bytes differ: '+name)
    seeds = plan['seed_selection']['seeds']
    expected_files = {f'records/{label}-{seed}.json.gz' for seed in seeds}
    actual_files = {p.relative_to(directory).as_posix() for p in (directory/'records').glob('*.json.gz')}
    require(actual_files == expected_files, 'Missing or extra fixed development records')
    audit = read(directory/'independent_audit.json')
    require(audit.get('records') == len(seeds) and isinstance(audit.get('errors'), list), 'Development audit is incomplete')
    require(audit.get('plan_sha256') == manifest['plan_sha256'], 'Audit plan differs')
    input_names = expected_files | {'manifest.json', 'freeze.json', 'source.zip', 'summary.json', 'plan.json'}
    require(set(audit.get('input_sha256', {})) == input_names, 'Audit did not bind every required input')
    for name in sorted(input_names):
        require(sha(directory/name) == audit['input_sha256'][name], 'Audited input changed: '+name)
    audit_items = audit.get('audits', [])
    require(len(audit_items) == len(seeds), 'Incomplete individual audit list')
    audits = {}
    for item in audit_items:
        key = (item['seed'], item['strategy'])
        require(key not in audits, 'Duplicated individual audit')
        require(type(item.get('passed')) is bool and isinstance(item.get('errors'), list), 'Invalid individual audit status')
        if item['passed']:
            require(item['errors'] == [] and audit_item_passed(item), 'Claimed passed item lacks physical/release_scheduling/inherited audits')
        else:
            require(bool(item['errors']), 'Failed audit lacks its error evidence')
        audits[key] = item
    require(set(audits) == {(seed, label) for seed in seeds}, 'Individual audits differ from the fixed cases')
    passed_count = sum(item['passed'] for item in audits.values())
    require(audit.get('passed_records') == passed_count and
            audit.get('all_passed') is (not audit['errors'] and passed_count == len(seeds)), 'Batch audit flags differ from complete individual audits')
    rows, case_hashes, triggered, action_counts = [], {}, [], {}
    for seed in seeds:
        with gzip.open(directory/f'records/{label}-{seed}.json.gz', 'rt', encoding='utf-8') as stream:
            record = json.load(stream)
        row = record['row']
        require(row['seed'] == seed and row['strategy'] == label and row['stage'] == runner.DESIGNS[split]['stage'], 'Record identity differs')
        require(row['source_total'] == runner.count_from_seed(seed), 'Actual source count differs from planned stratum')
        if record.get('record_kind') == 'infrastructure_failure_without_completed_case':
            require(row.get('successful') is False and row.get('infrastructure_error') is True and
                    row.get('virtual_time_s') is None and audits[(seed, label)]['passed'] is False, 'Invalid infrastructure failure envelope')
            case_hashes[seed] = None
        else:
            require(record['spec'] == plan['spec'] and record.get('evaluation_phase') == 'after_policy_termination', 'Record method or termination marker differs')
            signature = runner.digest(record['evaluation']['ground_truth'])
            require(row.get('case_sha256') == signature, 'Completed scenario SHA differs')
            case_hashes[seed] = signature
            if row.get('successful') is True:
                require(row.get('all_cleared') is True and row.get('completion_certified') is True and
                        row.get('accepted_exit') is True and row.get('cleared_total') == row['source_total'] and
                        row.get('errors') == [], 'Successful development case lacks completion evidence')
        if audits[(seed, label)]['passed']:
            count = actual_service_count(record)
            audited_count = audits[(seed, label)]['release_scheduling']['prefix'].get('forecast_scheduled_macros')
            require(type(audited_count) is int and audited_count == count,
                    'Audited forecast macro count differs from wire-bound plan ranges')
            action_counts[str(seed)] = count
            if count: triggered.append(seed)
        rows.append(row)
    require(audit.get('all_clear') is all(r['successful'] for r in rows), 'Audit all-clear flag differs from complete rows')
    summary = read(directory/'summary.json')
    recomputed = runner.summarize(rows, split, seeds, check_generator_counts=True)
    require(all(summary.get(k) == v for k, v in recomputed.items()), 'Development summary differs from all fixed rows')
    require(type(summary.get('source_unchanged')) is bool and isinstance(summary.get('infrastructure_errors'), list) and
            summary.get('complete') is (not summary['infrastructure_errors'] and summary['source_unchanged']), 'Invalid execution-complete flags')
    complete_pass = (summary['complete'] and summary['all_clear'] and audit['all_passed'])
    read_set = {(directory/name).relative_to(root).as_posix(): sha(directory/name)
                for name in sorted(input_names | {'independent_audit.json'})}
    result = dict(split=split, label=label, directory=directory.relative_to(root).as_posix(), runs=len(rows),
        triggered_case_seeds=triggered, triggered_cases=len(triggered), executed_forecast_macros=sum(action_counts.values()),
        forecast_macros_by_audited_seed=action_counts,
        successful=summary['successful'], all_clear=summary['all_clear'], audit_passed=bool(audit['all_passed']),
        passed_audit_records=passed_count, execution_complete=summary['complete'], source_unchanged=summary['source_unchanged'],
        failed_seeds=[r['seed'] for r in rows if not r['successful']],
        audit_failed_seeds=[s for s in seeds if not audits[(s, label)]['passed']],
        audit_global_errors=audit['errors'], infrastructure_errors=summary['infrastructure_errors'],
        mean_time_per_source_s=summary['mean_time_per_source_s'], p95_time_per_source_s=summary['p95_time_per_source_s'],
        mean_time_s=summary['mean_time_s'], mean_lower_bound_s=summary['mean_lower_bound_s'],
        mean_time_over_mean_lower_bound=summary['mean_time_over_mean_lower_bound'],
        complete_and_all_audits_passed=bool(complete_pass), mean_gate_passed=summary['mean_time_per_source_s'] <= DEVELOPMENT_LIMIT)
    return result, read_set, plan['source_sha256'], case_hashes


def build_selection(development, development_stress, root=ROOT):
    root = Path(root).resolve()
    label = next(iter(FIXED_SPECS))
    a, inputs_a, source, cases_a = inspect_development(development, 'development', label, root)
    b, inputs_b, source_b, cases_b = inspect_development(development_stress, 'development-stress', label, root)
    require(source == source_b, 'Development splits have different frozen source')
    require(not set(inputs_a) & set(inputs_b), 'Development evidence directories overlap')
    require(not set(cases_a) & set(cases_b), 'Development cases overlap')
    triggered = sorted(set(a['triggered_case_seeds']) | set(b['triggered_case_seeds']))
    criteria = dict(complete_and_audited=a['complete_and_all_audits_passed'] and b['complete_and_all_audits_passed'],
        random_mean_at_most_500=a['mean_gate_passed'], stress_mean_at_most_500=b['mean_gate_passed'],
        at_least_five_actual_triggered_cases=len(triggered) >= MIN_TRIGGERED_CASES)
    passed = all(criteria.values())
    return dict(schema='q4-release-scheduling-development-selection-v1', passed=passed,
        selected=label if passed else None, selected_spec=FIXED_SPECS if passed else {},
        attempted_spec=FIXED_SPECS, source_sha256=source, criteria=criteria,
        development_directory=a['directory'], development_stress_directory=b['directory'],
        development=a, development_stress=b, triggered_cases=len(triggered), triggered_case_seeds=triggered,
        gate_limit_s_per_source=DEVELOPMENT_LIMIT, read_set_sha256={**inputs_a, **inputs_b},
        scope='Both complete development batches retained, including failures. Require all-clear audits, both equal-case means <=500 and >=5 distinct cases with actual macros planned with both current-ready and future nonready tasks; zero-action decisions and single-readiness task sets do not count. Not 460 achievement or final promotion.')


def build_release(plan, plan_sha256, selection_path, root=ROOT):
    from experiments import run_q4_per_source as runner
    root = Path(root).resolve()
    runner.validate_plan(plan)
    require(plan['split'] in ('confirmation', 'stress'), 'Release only opens reserved independent splits')
    selection_path = inside(root, selection_path)
    chosen = read(selection_path)
    require(isinstance(chosen, dict) and all(isinstance(chosen.get(k), str) for k in
        ('development_directory', 'development_stress_directory')), 'Selection lacks both development evidence directories')
    actual = build_selection(chosen['development_directory'], chosen['development_stress_directory'], root)
    require(chosen == actual and actual['passed'] is True, 'Stored selection is not the recomputed passed development gate')
    require(actual['selected_spec'] == {plan['label']:plan['spec']}, 'Independent plan is not the selected method')
    require(actual['source_sha256'] == plan['source_sha256'], 'Independent plan differs from actual development source')
    name = selection_path.relative_to(root).as_posix()
    require(name not in actual['read_set_sha256'], 'Selection overwrites its own development evidence')
    return dict(schema='q4-release-scheduling-independent-release-v1', authorized=True,
        plan_sha256=plan_sha256, split=plan['split'], label=plan['label'], spec=plan['spec'],
        source_sha256=plan['source_sha256'], reserved_seeds=plan['seed_selection']['seeds'],
        development_selection_path=name, development_evidence_sha256={name:sha(selection_path), **actual['read_set_sha256']})


def validate_independent_release(plan, plan_sha256, release, root=ROOT):
    require(isinstance(release, dict) and release.get('authorized') is True, 'Explicit independent release required')
    require(isinstance(release.get('development_selection_path'), str), 'Release lacks actual development selection')
    expected = build_release(plan, plan_sha256, release['development_selection_path'], root)
    require(release == expected, 'Release differs from recomputed development gate or complete evidence binding')


def main():
    from experiments import run_q4_per_source as runner
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select')
    select.add_argument('--development', type=Path, required=True)
    select.add_argument('--development-stress', type=Path, required=True)
    select.add_argument('--output', type=Path, required=True)
    release = sub.add_parser('release')
    release.add_argument('--plan', type=Path, required=True)
    release.add_argument('--plan-sha256', required=True)
    release.add_argument('--selection', type=Path, required=True)
    release.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        result = build_selection(args.development, args.development_stress)
    else:
        require(sha(args.plan) == args.plan_sha256, 'Independent plan raw SHA differs')
        result = build_release(read(args.plan), args.plan_sha256, args.selection)
    runner.write_new(args.output, result)
    print(json.dumps({k:result[k] for k in ('schema', 'passed', 'selected', 'triggered_cases', 'authorized') if k in result}))
    return int(result.get('passed') is False)


if __name__ == '__main__':
    raise SystemExit(main())

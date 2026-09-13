"""Audit completed official practice sessions against recorded UI totals."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics


def summarize(root):
    observations = [json.loads(s) for s in (root / 'official_ui_observations.jsonl').read_text().splitlines() if s.strip()]
    by_label = {r['case_label']: r for r in observations}
    correction = json.loads((root / 'v4_case_label_correction.json').read_text())
    rows = []
    for path in sorted((root / 'evidence/practice_results').glob('*/result.json')):
        r = json.loads(path.read_text())
        ui = by_label[r['case_label']]
        code = ui.get('official_case_code')
        if r['case_label'] == correction['raw_operator_case_label']:
            code = correction['official_case_code']
        logs = list((root / 'evidence/Jammers-simulator').rglob(f'*{code}.jlog'))
        assert len(logs) == 1, (code, logs)
        events = [json.loads(s) for s in path.with_name('requests.jsonl').read_text().splitlines()]
        requests = {e['payload']['request_id']: e for e in events if e['event'] == 'request'}
        responses = {e['request_id']: e['response'] for e in events if e['event'] == 'response' and e.get('response', {}).get('accepted')}
        clears = [e for key, e in responses.items() if requests[key]['path'] == '/clear' and e.get('clear_result') == 'success']
        s = r['client_state']
        assert r['status'] == 'policy_completed_and_exited' and s['session'] == 'exited'
        assert r['pending_request'] is None
        assert len(clears) == s['cleared_count'] == ui['total_sources']
        assert abs(s['virtual_time_s'] - ui['ui_virtual_seconds']) <= .00051
        assert len(responses) == s['accepted_actions']
        assert list(requests.values())[-1]['path'] == '/exit'
        assert s['cleared_count'] == 16 or r['policy']['coverage_certificate']['ok']
        p = r['policy'].get('planning', {})
        rows.append(dict(method=r['method'], label=r['case_label'], official_case_code=code,
                         total_sources=ui['total_sources'], omni=ui['omnidirectional_sources'], directional=ui['directional_sources'],
                         cleared=s['cleared_count'], virtual_seconds=s['virtual_time_s'],
                         seconds_per_source=s['virtual_time_s']/s['cleared_count'], wall_seconds=r['wall_seconds'],
                         accepted_actions=s['accepted_actions'], failed_clears=sum(v['failed_clear_count'] for v in s['sources'].values()),
                         time_breakdown=s['time_breakdown'], stop_certificate=r['policy']['stop_certificate'],
                         final_coverage_ok=(r['policy']['coverage_certificate'] or {}).get('ok'),
                         post_last_clear_seconds=s['virtual_time_s']-r['policy']['last_success_virtual_s'],
                         actual_extra_actions=r['policy']['extra_actions'], shifted_sites=r['policy']['shifted_sites'],
                         planning_attempts=p.get('attempts', 0), planning_accepted=p.get('accepted', 0),
                         planning_seconds=p.get('wall_seconds', 0), rollout_runs=p.get('rollout_runs', 0),
                         planning_timeouts=p.get('timeouts',0), invalid_rollouts=p.get('invalid_rollouts',0),
                         posterior_fallbacks=p.get('posterior_fallbacks',0),
                         result_file=str(path.relative_to(root)), official_log=str(logs[0].relative_to(root)),
                         official_log_sha256=hashlib.sha256(logs[0].read_bytes()).hexdigest()))
    groups = []
    for method in ['v4','analytic','rollout','shared','dynamic']:
        rs = [r for r in rows if r['method'] == method]
        if not rs:
            continue
        per_source = [r['seconds_per_source'] for r in rs]
        groups.append(dict(method=method, runs=len(rs), cleared=sum(r['cleared'] for r in rs),
                           mean_seconds_per_source=statistics.mean(per_source),
                           min_seconds_per_source=min(per_source), max_seconds_per_source=max(per_source),
                           mean_virtual_seconds=statistics.mean(r['virtual_seconds'] for r in rs),
                           mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rs),
                           planning_accepted=sum(r['planning_accepted'] for r in rs),
                           planning_attempts=sum(r['planning_attempts'] for r in rs)))
    result = dict(protocol='Official Q4 practice, independent random cases; unpaired descriptive statistics only.',
                  aggregate='Arithmetic mean of per-case virtual seconds divided by actual source count.',
                  runs=rows, summary=groups, audit='passed')
    (root/'official_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(runs=len(rows), cleared=sum(r['cleared'] for r in rows), summary=groups), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)

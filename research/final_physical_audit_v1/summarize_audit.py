"""Summarize an existing physical audit; never generate or replay a scenario."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean


def quantile(values, q):
    """Linear interpolation at (n-1)*q, including the endpoints."""
    values = sorted(values)
    p = (len(values) - 1) * q
    lo = int(p)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (p - lo) * (values[hi] - values[lo])


def summarize(audit, audit_sha256):
    rows = audit['rows']
    if not rows or len(rows) != audit['records']:
        raise ValueError('Empty or inconsistent audit row count')
    groups = defaultdict(list)
    for row in rows:
        groups[row['label']].append(row)
    output = dict(kind='saved_physical_audit_summary_v1', audit_sha256=audit_sha256,
                  records=len(rows), audit_passed=audit['audit_passed'],
                  failed_audits=audit['failed_audits'], cache_hits=audit['cache_hits'],
                  cache_misses=audit['cache_misses'], audit_wall_s=audit['wall_s'],
                  quantile_definition='linear interpolation at (n-1)*q', groups={})
    for label, members in sorted(groups.items()):
        good = [r for r in members if r.get('audit_passed') and
                r.get('eligible_for_full_clear_ratio')]
        attempts = [a for r in members for a in r.get('observations', {}).get('clear_attempts', [])]
        g = dict(runs=len(members), successful=sum(r.get('successful', False) for r in members),
                 ratio_eligible=len(good), mean_time_s=fmean(r['virtual_time_s'] for r in members),
                 mean_penalized_time_s=fmean(r['penalized_time_s'] for r in members),
                 successful_clears=sum(a['outcome'] == 'success' for a in attempts),
                 failed_clears=sum(a['outcome'] != 'success' for a in attempts),
                 pre_certified_successful_clears=sum(a['outcome'] == 'success' and
                                                    a['prefix_certified'] for a in attempts),
                 inferred_verified=sum(r.get('observations', {}).get('inferred_silence_verified', 0)
                                       for r in members),
                 terminal_certified_runs=sum(r.get('observations', {}).get('terminal_certified', False)
                                             for r in members),
                 actual_16clear_cap_runs=sum(r.get('observations', {}).get('source_cap_actual_clears', False)
                                             for r in members), bounds={})
        for kind in ('original', 'cited_six_disk'):
            subset = good if kind == 'original' else [r for r in good if 'cited_six_disk' in r]
            if not subset:
                continue
            lowers = [r['original_lower_s'] if kind == 'original' else
                      r['cited_six_disk']['conditional_machine_lower_s'] for r in subset]
            if min(lowers) <= 0:
                raise ValueError('Nonpositive lower bound')
            times = [r['virtual_time_s'] for r in subset]
            ratios = [t / low for t, low in zip(times, lowers)]
            g['bounds'][kind] = dict(runs=len(subset), mean_lower_s=fmean(lowers),
                ratio_of_means=fmean(times) / fmean(lowers), mean_per_case_ratio=fmean(ratios),
                min_ratio=min(ratios), median_ratio=quantile(ratios, .5),
                p95_ratio=quantile(ratios, .95), max_ratio=max(ratios))
        output['groups'][label] = g
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.audit.read_bytes()
    result = summarize(json.loads(raw), hashlib.sha256(raw).hexdigest())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

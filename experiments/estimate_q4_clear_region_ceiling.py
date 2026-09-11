"""Fixed-action-trace upper bound on travel saved by certified clear shifts.

This reads completed observation logs, not evaluation ground truth. It does not
bound a replanned policy whose action order or measurement positions change.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from localization import CandidateRegion


def one(record):
    regions, current, previous_shift = {}, (0., 0.), 0.
    edge_bounds, changed_count, changed_edges = [], 0, 0
    for action in record['summary']['action_history']:
        channel, point = action['channel'], tuple(action['position'])
        edge_length = math.dist(current, point)
        shift = 0.
        if action['action'] == 'measure':
            if action['result'] == 'direction':
                regions.setdefault(channel, CandidateRegion()).observe(point, action['bearing_deg'])
        elif action['phase'] in ('certified_clear', 'near_clear'):
            changed_count += 1
            radius = (14.99999 if action['phase'] == 'near_clear' else
                      math.sqrt(max(0., 19.99999**2 - regions[channel].enclosing_disk().radius**2)))
            # Incoming cannot exceed the distance from the modified predecessor
            # to the old center. Include its possible shift, not just old length.
            shift = min(radius, 2. * (edge_length + previous_shift))
        edge_bounds.append(min(edge_length, previous_shift + shift) / 5.)
        changed_edges += previous_shift + shift > 0.
        current, previous_shift = point, shift
    row = record['row']
    return {'seed': row['seed'], 'time_s': row['virtual_time_s'],
            'historical_lower_bound_s': row['common_lower_bound_s'],
            'time_over_lower_bound': row['time_over_lower_bound'],
            'certified_or_near_clears': changed_count,
            'continuous_fixed_trace_travel_saving_upper_s': sum(edge_bounds),
            'changed_edges': changed_edges,
            'fixed_trace_travel_saving_upper_s': sum(edge_bounds)+changed_edges*1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob('compact_combo-*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            rows.append(one(json.load(stream)))
    if not rows:
        raise ValueError('No incumbent traces')
    result = {'scope': 'Prior completed observations; diagnostic, not new independent evidence. Fixed action order and measurement/optical-grid positions. Ideal MEC contact-point bound evaluated in floating arithmetic, not interval proof. No claim about replanned total time.',
        'formula': 'rho=sqrt(tau^2-r^2) for true MEC; rho=14.99999 near. delta_i=min(rho_i,2*(old_edge_i+delta_previous)); fixed points delta_i=0. Travel saving <=sum(min(old_edge_i,delta_previous+delta_i))/5, plus at most1 microsecond per potentially changed edge for ledger rounding.',
        'records': len(rows), 'mean_time_s': statistics.mean(r['time_s'] for r in rows),
        'mean_lower_bound_s': statistics.mean(r['historical_lower_bound_s'] for r in rows),
        'mean_fixed_trace_gain_upper_s': statistics.mean(r['fixed_trace_travel_saving_upper_s'] for r in rows),
        'rows': rows}
    result['mean_time_over_mean_lower_bound'] = result['mean_time_s']/result['mean_lower_bound_s']
    result['mean_fixed_trace_gain_upper_fraction'] = result['mean_fixed_trace_gain_upper_s']/result['mean_time_s']
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    main()

"""Post-selection paired quantile uncertainty; never changes the frozen gate."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = {}
    for stage in ('confirmation', 'stress'):
        data = json.loads((ROOT/'results/q4_joint_visibility'/stage/'summary.json').read_bytes())
        pairs = {}
        for row in data['rows']:
            pairs.setdefault(row['seed'], {})[row['strategy']] = row
        keys = sorted(pairs)
        old = np.array([pairs[k]['compact_clear_before_probe']['penalized_time_s'] for k in keys])
        new = np.array([pairs[k]['compact_joint_probe']['penalized_time_s'] for k in keys])
        index = np.random.default_rng(618959).integers(0, len(keys), size=(10000, len(keys)))
        pold, pnew = np.quantile(old[index], .95, axis=1), np.quantile(new[index], .95, axis=1)
        result[stage] = dict(pairs=len(keys), p95_saved_s=float(np.quantile(old, .95)-np.quantile(new, .95)),
            p95_saving_ci95_s=np.quantile(pold-pnew, [.025, .975]).tolist(),
            p95_ratio=float(np.quantile(new, .95)/np.quantile(old, .95)),
            p95_ratio_ci95=np.quantile(pnew/pold, [.025, .975]).tolist())
    result['scope'] = ('Post-selection descriptive paired bootstrap supplement; does not change preregistered qualification. '
        '10000 paired case resamples, seed618959, NumPy linear quantile. Tail intervals are not simultaneous multiple-comparison guarantees.')
    path = ROOT/'research/q4_joint_visibility/tail_bootstrap_supplement.json'
    if path.exists():
        if json.loads(path.read_bytes()) != result:
            raise ValueError('Preserve a different prior analysis')
    else:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

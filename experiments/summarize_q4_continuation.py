"""Post-selection accounting and paired RL comparison; no policy tuning."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.analyze_q4_joint_visibility import analyze
from experiments.evaluate_q4_joint_continuation import comparison


def main():
    research = ROOT / 'research/q4_joint_continuation'
    qualification = json.loads((research / 'qualification.json').read_bytes())
    if not qualification['passed']:
        raise ValueError('Require completed independent qualification')
    result = {}
    for stage in ('confirmation', 'stress'):
        directory = ROOT / 'results/q4_joint_continuation' / stage
        combined = json.loads(directory.with_name(stage + '-rl').joinpath('comparison.json').read_bytes())
        result[stage] = dict(
            actual_cost_accounting=analyze(directory, 'compact_joint_continuation', 'compact_joint_probe'),
            paired_vs_rl=comparison(combined, 'compact_joint_continuation', 'compact_macro_ppo512'),
            paired_vs_original=comparison(combined, 'compact_joint_continuation', 'compact_baseline'))
    result['scope'] = 'Descriptive completed-run accounting and fixed same-case reference comparisons. Not a causal decomposition or a policy selection rule.'
    path = research / 'independent-accounting.json'
    if path.exists():
        if json.loads(path.read_bytes()) != result:
            raise ValueError('Preserve differing prior analysis')
    else:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: {name: value for name, value in v.items() if name != 'actual_cost_accounting'}
                      for k, v in result.items() if k != 'scope'}))


if __name__ == '__main__':
    main()

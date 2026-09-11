"""Apply the predeclared observation-tree development and confirmation gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.run_q4_round2 import hashes, write_json
from experiments.evaluate_q4_round2 import comparison

RESEARCH = ROOT / 'research/q4_r4_observation'
RESULTS = ROOT / 'results/q4_observation'


def read_set(name, evidence):
    directory = RESULTS / name
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    if manifest['source_sha256'] != hashes():
        raise ValueError('Frozen experiment source differs')
    for name in ('independent_audit.json', 'observation_audit.json'):
        if json.loads((directory / name).read_bytes()).get('all_passed') is not True:
            raise ValueError('Need every physical and observation audit to pass')
    for name in ('manifest.json', 'freeze.json', 'summary.json', 'independent_audit.json', 'observation_audit.json'):
        path = directory / name
        evidence[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads((directory / 'summary.json').read_bytes()), manifest


def main(phase):
    destination = RESEARCH / ('development-decision.json' if phase == 'development' else 'qualification.json')
    if destination.exists():
        raise ValueError('Preserve existing decision')
    evidence = {}
    names = ('development', 'development-stress') if phase == 'development' else ('confirmation', 'stress')
    reports, manifests = {}, {}
    for name in names:
        reports[name], manifests[name] = read_set(name, evidence)
    if manifests[names[0]]['specs'] != manifests[names[1]]['specs']:
        raise ValueError('Compared sets use different specifications')
    candidates = [label for label in manifests[names[0]]['specs'] if label.startswith('compact_observation_')]
    if phase == 'development' and set(candidates) != {'compact_observation_d1', 'compact_observation_d2'}:
        raise ValueError('Need both predeclared development candidates')
    if phase == 'confirmation' and len(candidates) != 1:
        raise ValueError('Independent confirmation must contain only one selected candidate')
    accepted, comparisons = [], {}
    for label in candidates:
        comparisons[label] = {name: comparison(report, label, 'compact_combo') for name, report in reports.items()}
        random, stress = (comparisons[label][name] for name in names)
        mean_gate = random['mean_saved_s'] > 0 if phase == 'development' else random['saving_ci95_s'][0] > 0
        passes = (random['all_clear'] and stress['all_clear'] and mean_gate
                  and stress['mean_saved_s'] >= 0 and random['p95_ratio'] <= 1.05 and stress['p95_ratio'] <= 1.05)
        if passes:
            accepted.append(label)
    selected = None
    if accepted:
        selected = min(accepted, key=lambda label: reports[names[0]]['summaries'][label]['mean_time_s'])
        if phase == 'development' and len(accepted) == 2:
            difference = abs(reports[names[0]]['summaries'][accepted[0]]['mean_time_s']
                             - reports[names[0]]['summaries'][accepted[1]]['mean_time_s'])
            if difference < 5.:
                selected = 'compact_observation_d1'
    decision = {'phase': phase, 'selected': selected, 'passed': selected is not None,
        'scope': 'Local synthetic mixed-source Q4 only; fixed configuration selection, no official trial',
        'decision_rule': 'Protocol.md: all-clear and all audits, positive random mean (confirmation CI95 lower>0), nonnegative stress mean, p95<=1.05 both; development tie within5s favors depth1',
        'comparisons_vs_incumbent': comparisons,
        'summaries': {name: report['summaries'] for name, report in reports.items()},
        'evidence_sha256': evidence, 'source_sha256': hashes(),
        'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if phase == 'development' and selected:
        source_specs = manifests[names[0]]['specs']
        specs = {label: source_specs[label] for label in ('compact_baseline', 'compact_combo', selected)}
        selection = {'role': 'Frozen single candidate before opening independent scenarios',
            'selected': selected, 'source_sha256': hashes(), 'specs': specs,
            'reserved_seeds': {'confirmation': list(range(612101, 612165)), 'stress': list(range(612201, 612243))},
            'development_evidence_sha256': evidence, 'evaluator_sha256': decision['evaluator_sha256']}
        for path in (RESEARCH / 'selection.json', RESEARCH / 'selection-specs.json'):
            if path.exists():
                raise ValueError('Preserve prior selection')
        write_json(RESEARCH / 'selection.json', selection)
        write_json(RESEARCH / 'selection-specs.json', specs)
    if phase == 'confirmation':
        selection_path = RESEARCH / 'selection.json'
        selection = json.loads(selection_path.read_bytes())
        for name in names:
            m = manifests[name]
            if (m['specs'] != selection['specs'] or m['seeds'] != selection['reserved_seeds'][name]
                    or m['selection_sha256'] != hashlib.sha256(selection_path.read_bytes()).hexdigest()):
                raise ValueError('Independent experiment differs from frozen selection')
        decision['selection_sha256'] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    write_json(destination, decision)
    print(json.dumps({'phase': phase, 'selected': selected, 'comparisons': comparisons}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('development', 'confirmation'), required=True)
    main(parser.parse_args().phase)

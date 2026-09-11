"""Separate immutable R16 candidate audits, using explicit completed stage paths."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.audit_q4_feedback_symmetry import audit_feedback_symmetry_prefix, SOURCE_CONTRACT

LABELS = {
    'compact_feedback_mean': ('bearing_mean', 'feedback_symmetry_mean_audit.json'),
    'compact_feedback_centers': ('early_centers', 'feedback_symmetry_centers_audit.json'),
}


def audit(directory, label):
    directory = Path(directory)
    if label not in LABELS:
        raise ValueError('Unknown feedback symmetry label')
    config, name = LABELS[label]
    destination = directory / name
    if destination.exists():
        raise ValueError('Preserve prior feedback symmetry audit')
    generic = json.loads((directory / 'independent_audit.json').read_bytes())
    if generic.get('all_passed') is not True:
        raise ValueError('Require complete independent physical/coverage audit first')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    seeds = manifest['seeds']
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Missing or repeated manifest seeds')
    spec = manifest['specs'][label]
    if (spec['entrypoint'] != 'strategies.q4_feedback_symmetry:run_q4_feedback_symmetry'
            or spec.get('kwargs', {}).get('config', 'bearing_mean') != config):
        raise ValueError('Candidate label/spec differs')
    if not SOURCE_CONTRACT or any(manifest['source_sha256'].get(p) != v for p, v in SOURCE_CONTRACT.items()):
        raise ValueError('Manifest does not bind frozen source contract')
    details, errors, evidence = [], [], {}
    for seed in seeds:
        path = directory / 'records' / f'{label}-{seed}.json.gz'
        evidence[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        decoded = json.loads(gzip.decompress(path.read_bytes()))
        if (decoded['row']['seed'] != seed or decoded['row']['strategy'] != label
                or decoded['spec'] != spec):
            raise ValueError('Record scenario identity/spec differs from manifest')
        record = {k: decoded[k] for k in ('summary', 'history', 'row', 'spec')}
        del decoded
        try:
            result = audit_feedback_symmetry_prefix(record)
            details.append(dict(strategy=label, seed=seed, **result))
        except (ValueError, AssertionError, KeyError, TypeError) as exc:
            errors.append(dict(strategy=label, seed=seed, error=str(exc)))
    for p in ('manifest.json', 'freeze.json', 'independent_audit.json'):
        evidence[p] = hashlib.sha256((directory / p).read_bytes()).hexdigest()
    summary = {k: sum(a[k] for a in details) for k in ('events', 'changed_routes', 'initial_measurements')}
    result = dict(all_passed=not errors, records=len(seeds), passed_records=len(details),
                  audits=details, errors=errors, summaries={label: summary}, input_sha256=evidence,
                  scope='Actual origin feedback, independent D7 permutation/score, locked full chain and inherited R8/R12 prefixes; no truth input')
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('audits', 'input_sha256')}))
    return int(not result['all_passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--label', choices=sorted(LABELS))
    args = parser.parse_args()
    labels = [args.label] if args.label else [k for k in LABELS if k in json.loads((args.input/'manifest.json').read_bytes())['specs']]
    if not labels:
        raise ValueError('No feedback symmetry candidate')
    raise SystemExit(max(audit(args.input.resolve(), label) for label in labels))

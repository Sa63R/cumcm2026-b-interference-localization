"""Exact-frontier and inherited R8/R12 audit after complete physical audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.audit_q4_exact_frontier import audit_exact_frontier_prefix, SOURCE_CONTRACT


def audit(directory):
    destination = directory / 'exact_frontier_audit.json'
    if destination.exists():
        raise ValueError('Preserve prior exact-frontier audit')
    generic = json.loads((directory / 'independent_audit.json').read_bytes())
    if generic.get('all_passed') is not True:
        raise ValueError('Require complete independent physical/coverage audit first')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    if not manifest['seeds'] or len(set(manifest['seeds'])) != len(manifest['seeds']):
        raise ValueError('Missing or duplicate scenario identities')
    if any(manifest['source_sha256'].get(p) != value for p, value in SOURCE_CONTRACT.items()):
        raise ValueError('Frozen source contract does not match audited exact-frontier implementation')
    labels = [k for k, spec in manifest['specs'].items()
              if spec['entrypoint'] == 'strategies.q4_exact_frontier:run_q4_exact_frontier']
    if not labels:
        raise ValueError('No exact-frontier arm')
    details, groups, evidence, errors = [], {}, {}, []
    for label in labels:
        items = []
        for seed in manifest['seeds']:
            path = directory / 'records' / f'{label}-{seed}.json.gz'
            evidence[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                decoded = json.load(stream)
            if (decoded['row']['seed'] != seed or decoded['row']['strategy'] != label
                    or decoded['spec'] != manifest['specs'][label]):
                raise ValueError('Scenario identity or spec differs from manifest')
            record = {k: decoded[k] for k in ('summary', 'history', 'row', 'spec')}
            del decoded
            try:
                result = audit_exact_frontier_prefix(record)
                items.append(result)
                details.append({'strategy': label, 'seed': seed, **result})
            except (ValueError, AssertionError, KeyError, TypeError) as exc:
                errors.append({'strategy': label, 'seed': seed, 'error': str(exc)})
        if len(items) == len(manifest['seeds']):
            groups[label] = {k: sum(x[k] for x in items) for k in ('route_calls','dp_calls','applied','astar_expanded','dp_states','dp_arcs')}
            groups[label]['records'] = len(items)
    for name in ('manifest.json', 'freeze.json', 'independent_audit.json'):
        evidence[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    result = {'all_passed': not errors, 'records': len(details)+len(errors),
        'passed_records': len(details), 'summaries': groups, 'errors': errors,
        'input_sha256': evidence, 'audits': details,
        'scope': 'Actual-prefix finite-route refinement, independent backward Bellman, separate A*/DP work, inherited R8/R12; no truth input'}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'audits')}))
    return int(not result['all_passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    raise SystemExit(audit(parser.parse_args().input.resolve()))

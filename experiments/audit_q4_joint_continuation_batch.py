"""Read-only R12 recursive prefix and inherited R8 audit after the complete physical audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.audit_q4_joint_continuation import (
    audit_joint_continuation_prefix, summarize_joint_continuation_audits)
from experiments.audit_q4_clear_before_probe import audit_clear_before_probe_prefix


def audit(directory):
    destination = directory / 'joint_continuation_prefix_audit.json'
    if destination.exists():
        raise ValueError('Preserve prior joint-visibility prefix audit')
    generic = json.loads((directory / 'independent_audit.json').read_bytes())
    if generic.get('all_passed') is not True:
        raise ValueError('Require complete independent physical/coverage audit first')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    labels = [k for k, spec in manifest['specs'].items()
              if spec['entrypoint'] == 'strategies.q4_joint_continuation:run_q4_joint_continuation']
    if not labels:
        raise ValueError('No joint-visibility arm')
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
                inherited = audit_clear_before_probe_prefix(record)
                result = audit_joint_continuation_prefix(record)
                items.append(result)
                details.append({'strategy': label, 'seed': seed, **result, 'inherited_r8_audit': inherited})
            except (ValueError, AssertionError, KeyError, TypeError) as exc:
                errors.append({'strategy': label, 'seed': seed, 'error': str(exc)})
        if len(items) == len(manifest['seeds']):
            groups[label] = summarize_joint_continuation_audits(items)
    for name in ('manifest.json', 'freeze.json', 'independent_audit.json'):
        evidence[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    result = {'all_passed': not errors, 'records': len(details)+len(errors),
        'passed_records': len(details), 'summaries': groups, 'errors': errors,
        'input_sha256': evidence, 'audits': details,
        'scope': 'Recursive actual-miss certificate binding, actual auxiliary updates and complete optical strip coverage; no truth input to helpers'}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'audits')}))
    return int(not result['all_passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    raise SystemExit(audit(parser.parse_args().input.resolve()))

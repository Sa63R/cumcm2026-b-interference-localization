"""Audit and summarize frozen clear-region trials after the generic audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.audit_q4_clear_region import audit_clear_region_prefix


def summarize_region_audits(items):
    fields = ('events', 'changed_positions', 'executed_events', 'fallbacks', 'proxy_saved_s', 'decision_wall_s')
    result = {key: sum(item[key] for item in items) for key in fields}
    result['records'] = len(items)
    result['changed_position_fraction'] = result['changed_positions']/result['executed_events'] if result['executed_events'] else None
    result['mean_planning_wall_s'] = result['decision_wall_s']/result['events'] if result['events'] else None
    return result


def audit(directory):
    destination = directory / 'clear_region_audit.json'
    if destination.exists():
        raise ValueError('Preserve existing clear-region audit')
    generic = json.loads((directory / 'independent_audit.json').read_bytes())
    if generic.get('all_passed') is not True:
        raise ValueError('Require complete independent physical/coverage audit first')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    labels = [k for k, spec in manifest['specs'].items()
              if spec['entrypoint'] == 'strategies.q4_clear_region:run_q4_clear_region']
    if not labels:
        raise ValueError('No clear-region arm')
    groups, details, evidence, errors = {}, [], {}, []
    for label in labels:
        items = []
        for seed in manifest['seeds']:
            path = directory / 'records' / f'{label}-{seed}.json.gz'
            evidence[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                decoded = json.load(stream)
            if decoded['row']['seed'] != seed or decoded['row']['strategy'] != label:
                raise ValueError('Scenario identity differs from file')
            # Only accepted observations and protocol metadata enter the helper.
            record = {k: decoded[k] for k in ('summary', 'history', 'row', 'spec')}
            del decoded
            try:
                result = audit_clear_region_prefix(record)
                items.append(result)
                details.append({'strategy': label, 'seed': seed, **result})
            except (ValueError, AssertionError, KeyError, TypeError) as exc:
                errors.append({'strategy': label, 'seed': seed, 'error': str(exc)})
        if len(items) == len(manifest['seeds']):
            groups[label] = summarize_region_audits(items)
    for name in ('manifest.json', 'freeze.json', 'independent_audit.json'):
        evidence[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    result = {'all_passed': not errors, 'records': len(details) + len(errors),
              'passed_records': len(details), 'summaries': groups, 'errors': errors,
              'input_sha256': evidence, 'audits': details,
              'scope': 'Actual-prefix geometry and local proxy audit; not a full-route optimality certificate'}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'audits')}))
    return int(not result['all_passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    raise SystemExit(audit(parser.parse_args().input.resolve()))

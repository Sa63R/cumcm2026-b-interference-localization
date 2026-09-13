"""Verify the handoff files, optionally reproducing 15 archived policy runs.

Hash verification needs only Python's standard library. --reproduce also
requires requirements.txt. No files or archived results are modified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_manifest():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        file = ROOT / item['path']
        require(file.resolve().is_relative_to(ROOT), 'Invalid manifest path')
        raw = file.read_bytes()
        require(len(raw) == item['bytes'], f"Size mismatch: {item['path']}")
        require(hashlib.sha256(raw).hexdigest() == item['sha256'],
                f"SHA-256 mismatch: {item['path']}")
    return len(manifest['files'])


def business_actions(data):
    volatile = {'real_timestamp_ms', 'remaining_real_duration_s'}
    return [dict(path=a['path'], request=a['request'],
                 response={k: v for k, v in a['response'].items() if k not in volatile})
            for a in data['history']]


def reproduce():
    sys.path.insert(0, str(ROOT / 'code'))
    import runner
    runner.check_inputs()
    runner.initialize()
    folder = ROOT / 'code/results/paired2000'
    indices = {0, 999, 1999}
    rows = [json.loads(line) for line in (folder / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = sorted((r for r in rows if r['index'] in indices),
                      key=lambda r: (r['index'], r['method']))
    require(len(selected) == 15, 'Expected 3 cases x 5 methods')
    checked = []
    for archived in selected:
        old = json.loads(gzip.decompress((folder / archived['evidence']).read_bytes()))
        row, data = runner.run_one(runner.Scenario.generate(3, archived['seed']), archived['method'])
        require(row['success'], f"Policy failed: {archived['seed']} {archived['method']}")
        # Scenario.as_dict keeps sources as a tuple; JSON archives use a list.
        require(json.dumps(data['state']['scenario'], sort_keys=True) ==
                json.dumps(old['state']['scenario'], sort_keys=True), 'Generated scene mismatch')
        require(row['virtual_us'] == archived['virtual_us'], 'Virtual microsecond mismatch')
        require(business_actions(data) == business_actions(old), 'Policy action/response mismatch')
        checked.append(dict(index=archived['index'], method=archived['method'],
                            virtual_us=row['virtual_us'], actions=row['replay_actions']))
    # All project modules must load from this bundle; third-party packages are
    # intentionally supplied by the recipient's Python environment.
    for name in ('runner', 'q3_base', 'q3_optimized', 'q3_v3', 'new_methods',
                 'probe_score', 'route_search', 'jammers_local.core'):
        require(Path(sys.modules[name].__file__).resolve().is_relative_to(ROOT),
                f'Project dependency outside bundle: {name}')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    result = dict(status='verified', files_verified=check_manifest())
    if args.reproduce:
        result['reproduced_runs'] = reproduce()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

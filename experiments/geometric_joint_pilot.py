"""Small serial training-only ablations for the independent geometric branch."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

from experiments.research_v1_eval import (PROTOCOL, ROOT, identity, paired_comparison,
                                         random_scenario, read_json, run_case, summarize, write_json)

PHASES = {'pilot': (103001, 103017), 'confirmation': (103017, 103081),
          'route-clear-pilot': (105001, 105017), 'route-clear-confirmation': (105017, 105081),
          'probe-cost-pilot': (107001, 107017), 'probe-cost-confirmation': (107017, 107081)}
DEFAULT_SPECS = ['v1_baseline_efficient.json', 'v1_baseline_rollout.json',
                 'v1_geometric_clear_only.json', 'v1_geometric_joint.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, default='pilot')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path, action='append')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    paths = args.spec or [ROOT/'research'/name for name in DEFAULT_SPECS]
    specs = [read_json(path) for path in paths]
    if len({spec['name'] for spec in specs}) != len(specs):
        parser.error('spec names must be unique')
    protocol = read_json(PROTOCOL)
    seeds = list(range(*PHASES[args.phase]))
    manifest = dict(phase=args.phase, seeds=seeds, source_access='Training only; no official simulator.',
                    specs=specs, identities={spec['name']: identity(spec, protocol) for spec in specs},
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or read_json(args.output/'manifest.json') != manifest:
            parser.error('Nonempty output requires matching --resume manifest')
    write_json(args.output/'manifest.json', manifest)
    rows, diagnostics = [], []
    started = time.perf_counter()
    for seed in seeds:
        for spec in specs:
            path = args.output/'cases'/spec['name']/f'case-{seed}.json.gz'
            if path.exists():
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    record = json.load(stream)
            else:
                record = run_case(random_scenario(3, seed), spec, protocol)
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, 'wt', encoding='utf-8') as stream:
                    json.dump(record, stream, ensure_ascii=False, allow_nan=False)
            row = record['row']
            rows.append(row)
            sharing = (record.get('summary') or {}).get('joint_observations', {})
            diagnostics.append(dict(seed=seed, strategy=spec['name'],
                **{key: sharing.get(key) for key in ('shared_measurements', 'actual_sharing_time_s', 'by_trigger')}))
            print(json.dumps(dict(seed=seed, strategy=spec['name'], successful=row['successful'],
                                  seconds=row['virtual_time_s'], failed_clears=row['failed_clear_count'],
                                  shared=sharing.get('shared_measurements', 0))), flush=True)
        write_json(args.output/'rows.json', rows)
    by_method = {spec['name']: [r for r in rows if r['strategy'] == spec['name']] for spec in specs}
    summary = dict(manifest=manifest, wall_s=time.perf_counter()-started,
                   groups={name: summarize(values, len(seeds)) for name, values in by_method.items()},
                   comparisons={}, joint_diagnostics=diagnostics)
    for base in ('efficient_frozen', 'rollout_frozen', 'geometric_clear_only', 'geometric_joint', 'geometric_probe_single'):
        if base in by_method:
            summary['comparisons'][base] = {name: paired_comparison(by_method[base], values, protocol)
                for name, values in by_method.items() if name != base}
    for name, group in summary['groups'].items():
        group['mean_components_s'] = {key: statistics.mean(r[key] for r in by_method[name])
                                     for key in ('movement_s', 'switching_s', 'detection_s', 'optical_s', 'removal_s')}
    write_json(args.output/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('manifest', 'joint_diagnostics')}, indent=2))


if __name__ == '__main__':
    main()

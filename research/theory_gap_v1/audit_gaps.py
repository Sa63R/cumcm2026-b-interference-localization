"""Read-only, cached-only physical audit of explicitly named validation sets.

External theory helpers are required at pinned hashes; no copying, simulator
construction, DP recomputation, cache writing, or held-out evaluation occurs.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
PROTOCOL_SHA256 = 'a8b6db5884a025d5295da19021267c3e4524eb9eb3b76cd337f9a743b653d87a'
PINNED_HELPERS = {
    'audit_eval_bounds.py': '2b9b565df6babc627af36043bdec8206b1aa51ae9097634ba31aad8cfdf068c2',
    'certify_bounds.py': '00b8041daa00edbc9ddffb8faef2071204f9f8ae80453d9c06990d064317a484',
    'audit_cited_six_disk.py': '795b0cf67dcb0f31816b9f683e4420aa21dec61139af535f6afcf84e64a5a63b',
    'six_disk_certificate.py': '5a4b59df1e68d2df34b350aa16527b460ffd2762aa265074607185615412f0df',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def helpers(theory_dir):
    for name, expected in PINNED_HELPERS.items():
        if sha(theory_dir/name) != expected:
            raise ValueError(f'External theory helper changed: {name}; explicit re-audit needed')
    sys.path.insert(0, str(theory_dir.resolve()))
    base = importlib.import_module('audit_eval_bounds')
    cited = importlib.import_module('audit_cited_six_disk')
    for module in (base, cited):
        if Path(module.__file__).resolve().parent != theory_dir.resolve():
            raise ValueError('Unexpected theory helper import path')
    return base, cited


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values)-1)*fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low]*(high-index)+values[high]*(index-low) if low != high else values[low]


def distribution(values):
    return dict(mean=statistics.mean(values), minimum=min(values),
                p05=quantile(values, .05), median=quantile(values, .5),
                p95=quantile(values, .95), maximum=max(values))


def statistics_for(rows):
    mean_time = statistics.mean(r['virtual_time_s'] for r in rows)
    result = dict(runs=len(rows), successful_runs=sum(r['successful'] for r in rows),
                  failed_clear_count=sum(r['failed_clear_count'] for r in rows),
                  mean_time_s=mean_time, mean_components_s={key: statistics.mean(r[key] for r in rows)
                      for key in ('movement_s', 'switching_s', 'detection_s', 'optical_s', 'removal_s')})
    for name in ('original', 'cited_six_disk'):
        lowers = [r[name+'_lower_s'] for r in rows]
        ratios = [r['virtual_time_s']/lower for r, lower in zip(rows, lowers)]
        result[name] = dict(mean_lower_s=statistics.mean(lowers),
            ratio_of_means=mean_time/statistics.mean(lowers),
            per_case_ratio=distribution(ratios),
            per_case_gap_s=distribution([r['virtual_time_s']-lower for r, lower in zip(rows, lowers)]),
            worst_ratio_seed=rows[max(range(len(rows)), key=lambda i: ratios[i])]['seed'])
    return result


def validate_seed_manifest(manifest):
    if (manifest.get('split') != 'validation' or manifest.get('seeds') != list(range(6000, 6048))
            or manifest['identity']['protocol_sha256'] != PROTOCOL_SHA256):
        raise ValueError('Only the complete frozen common validation 6000..6047 is in scope')


def build(inputs, theory_dir):
    started = time.perf_counter()
    base, cited = helpers(theory_dir)
    cache_path = theory_dir/'results/validation_geometry_cache.json'
    baseline_path = theory_dir/'results/validation_bounds_hot.json'
    cache_sha, baseline_sha = sha(cache_path), sha(baseline_path)
    cache, old = base.load_cache(cache_path), read(baseline_path)
    if old['version'] != base.VERSION or old['records'] != 96 or old['scenarios'] != 48:
        raise ValueError('Expected previously audited 48-scenario, two-baseline evidence')
    reference = {}
    for row in old['rows']:
        seed = row['seed']
        if seed not in range(6000, 6048):
            raise ValueError('Baseline audit outside validation scope')
        if seed in reference and reference[seed]['case_sha256'] != row['case_sha256']:
            raise ValueError('Baseline strategies disagree about scenario identity')
        reference[seed] = row
    if set(reference) != set(range(6000, 6048)):
        raise ValueError('Incomplete baseline identity anchor')
    all_rows, input_provenance = [], {}
    for label, directory in inputs.items():
        directory = Path(directory)
        manifest = read(directory/'manifest.json')
        validate_seed_manifest(manifest)
        files = sorted(directory.glob('case-*.json.gz'))
        if {p.name for p in files} != {f'case-{s}.json.gz' for s in reference}:
            raise ValueError('Missing, extra or out-of-scope case files; payloads not opened')
        rows_on_disk = read(directory/'rows.json')
        loaded_rows = []
        for seed in range(6000, 6048):
            path = directory/f'case-{seed}.json.gz'
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
            row = record['row']
            loaded_rows.append(row)
            if (row['seed'] != seed or row['strategy'] != manifest['strategy']
                    or row['case_sha256'] != reference[seed]['case_sha256']
                    or base.digest(record['spec']) != manifest['identity']['spec_sha256']):
                raise ValueError('Case hash / strategy / spec mismatch')
            sources, physical_actions = base.audit_record(record)
            if not row['successful']:
                raise ValueError('Gap report requires a complete successful set; do not silently drop failures')
            geometry = [[c, sources[c]['x'], sources[c]['y']] for c in sorted(sources)]
            key = base.digest(dict(version=base.VERSION, geometry=geometry))
            if key != reference[seed]['geometry_cache_key'] or key not in cache:
                raise ValueError('No identical audited geometry cache entry; DP is deliberately disabled')
            physical = cache[key]
            if physical['source_route_lower_m'] != reference[seed]['source_route_lower_m']:
                raise ValueError('Cache / baseline source-route bound mismatch')
            bounds = base.improved_bound(physical['source_route_lower_m'], sources)
            rounding = .5e-6*physical_actions+1e-6
            lower = max(0., bounds['lower_bound_continuous_s']-rounding)
            auditable = {**physical, **bounds, 'source_total':len(sources), 'virtual_time_s':row['virtual_time_s'],
                'successful':True, 'eligible_for_full_clear_ratio':True,
                'movement_rounding_allowance_s':rounding, 'conditional_machine_lower_s':lower}
            enhanced = cited.enhance_row(auditable)['cited_six_disk']
            all_rows.append(dict(label=label, **row, original_lower_s=lower,
                cited_six_disk_lower_s=enhanced['conditional_machine_lower_s'],
                cited_increment_s=enhanced['increment_machine_s'],
                physical_clairvoyant_lower_s=physical['physical_clairvoyant_lower_s'],
                physical_clairvoyant_upper_s=physical['feasible_clairvoyant_upper_s'],
                spatial_certification_dominates=bounds['spatial_certification_lower_m']>physical['source_route_lower_m'],
                empty_action_and_entry_switch_lower_s=bounds['empty_action_and_entry_switch_lower_s'],
                movement_rounding_allowance_s=rounding, physical_actions=physical_actions,
                geometry_cache_key=key, input_file=str(path), input_sha256=sha(path)))
        if rows_on_disk != loaded_rows:
            raise ValueError('rows.json disagrees with individually audited physical records')
        input_provenance[label] = dict(directory=str(directory), manifest_sha256=sha(directory/'manifest.json'),
                                      rows_sha256=sha(directory/'rows.json'), identity=manifest['identity'])
    if sha(cache_path) != cache_sha or sha(baseline_path) != baseline_sha:
        raise ValueError('Read-only input mutated during audit')
    return dict(version='q3-common-validation-cached-gap-v1', inputs=input_provenance,
        helper_directory=str(theory_dir), helper_sha256=PINNED_HELPERS,
        cache_file=str(cache_path), cache_sha256=cache_sha, baseline_audit_sha256=baseline_sha,
        protocol_sha256=PROTOCOL_SHA256, source_sha256=sha(Path(__file__)),
        simulator_requests_sent=False, generated_scenarios=0, dp_calls=0, cache_hits=len(all_rows),
        source_cache_modified=False, truth_access='Physical-history audit only after policy termination',
        bound_scope='Conditional on all-clear certification for every legal Q3 hidden scene; sample success does not prove it.',
        cited_theorem=cited.theorem_premise(), ideal_expected_information_bound_added=False,
        scenario_count=len(reference), wall_s=time.perf_counter()-started,
        groups={label:statistics_for([r for r in all_rows if r['label']==label]) for label in inputs}, rows=all_rows)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(result['groups'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    maximum = max(r['virtual_time_s']/r['original_lower_s'] for r in result['rows'])
    for ax, name, title in zip(axes, ('original','cited_six_disk'), ('Original conditional bound','Optional cited six-disk bound')):
        data = [[r['virtual_time_s']/r[name+'_lower_s'] for r in result['rows'] if r['label']==label] for label in labels]
        ax.boxplot(data, tick_labels=labels, orientation='horizontal', showmeans=True)
        ax.axvline(1, color='gray', ls='--', lw=1)
        ax.set(xlabel='Per-case total time / conditional lower bound', title=title, xlim=(.94,maximum+.06))
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Common Linux validation: 48 matched cases; full physical ledgers audited\nRatios describe a conditional relaxation gap, not proven remaining achievable savings', fontsize=11)
    for suffix in ('png','svg'):
        fig.savefig(output/f'gap_distribution.{suffix}', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE/'inputs.json')
    parser.add_argument('--theory-dir', type=Path, default=WORKSPACE/'q3-state-search/research/theory_v1')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    inputs = {name:WORKSPACE/path for name,path in read(args.inputs).items()}
    result = build(inputs, args.theory_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    plot(result, args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','inputs')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

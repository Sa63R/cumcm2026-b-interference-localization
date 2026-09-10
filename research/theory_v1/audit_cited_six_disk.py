"""Optional, versioned enhancement of a completed audit_eval_bounds batch.

Reads only explicitly supplied audit JSON, never eval.gz, simulator state, or
geometry cache. The old audit and its default numbers are left unchanged.
The cited geometric theorem is a premise, not proved by this postprocessor.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from audit_eval_bounds import VERSION as BASE_VERSION, atomic_json
from certify_bounds import improved_bound
from six_disk_certificate import certificate

VERSION = 'q3-optional-cited-six-disk-v1'
LABEL = 'cited_six_disk'


@lru_cache(maxsize=1)
def theorem_premise():
    record = certificate()
    if [r['minimum_s'] for r in record['mixed_cover_integer_relaxation']] != [33, 34]:
        raise ValueError('External-theorem cost certificate changed; version review required')
    return dict(source_url=record['source_url'], source_page=150,
                source_sha256=record['download_sha256'],
                status='Original author published theorem; complete 1979 proof not independently read.',
                normalized_radius_comparison='1/1.7989 > 5/9, conditional on published decimal prefix',
                full_original_proof_read=False, cost_certificate='six_disk_certificate.py')


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def close(actual, expected, name):
    if not math.isclose(finite(actual, name), expected, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f'Base audit field inconsistent with frozen formula: {name}')


def enhance_row(row):
    """Preserve every old field; append a conditional mixed-strategy bound."""
    theorem_premise()
    n = row['source_total']
    channels = row['source_route_order_channels']
    if (type(n) is not int or not 10 <= n <= 16 or len(channels) != n
            or len(set(channels)) != n
            or any(type(c) is not int or not 1 <= c <= 20 for c in channels)):
        raise ValueError('Invalid source count / occupied-channel facts')
    length = finite(row['source_route_lower_m'], 'source_route_lower_m')
    old = improved_bound(length, channels)
    for key in ('lower_bound_continuous_s', 'empty_action_and_entry_switch_lower_s',
                'successful_clear_action_s', 'spatial_certification_lower_m',
                'combined_movement_lower_m'):
        close(row[key], old[key], key)
    if row['certification_applied'] is not (n < 16):
        raise ValueError('Base audit N=16 certification branch is inconsistent')
    rounding = finite(row['movement_rounding_allowance_s'], 'movement_rounding_allowance_s')
    if rounding < 1e-6:
        raise ValueError('Missing base movement rounding margin')
    old_machine = max(0., old['lower_bound_continuous_s']-rounding)
    close(row['conditional_machine_lower_s'], old_machine, 'conditional_machine_lower_s')
    actual = finite(row['virtual_time_s'], 'virtual_time_s')
    if actual < 0 or type(row['successful']) is not bool:
        raise ValueError('Invalid elapsed time / success flag')
    if row['eligible_for_full_clear_ratio'] is not row['successful']:
        raise ValueError('Base audit ratio eligibility inconsistent with success flag')
    empty = 20-n
    extra = 3*empty if n < 16 else 0
    continuous = old['lower_bound_continuous_s']+extra
    machine = max(0., continuous-rounding)
    # A violation contradicts the declared robust-policy premise, or the base
    # record. A successful row alone is not a proof of that policy premise.
    if row['successful'] and actual+1e-8 < machine:
        raise ValueError('Successful row violates cited conditional bound; check policy premise and base audit')
    added = dict(version=VERSION, active=n < 16, empty_channels=empty,
                 empty_action_and_entry_switch_lower_s=old['empty_action_and_entry_switch_lower_s']+extra,
                 increment_continuous_s=extra, lower_bound_continuous_s=continuous,
                 movement_rounding_allowance_s=rounding,
                 conditional_machine_lower_s=machine,
                 increment_machine_s=machine-old_machine,
                 time_over_conditional_lower=actual/machine if row['successful'] else None,
                 nonnegative_gap_s=max(0., actual-machine) if row['successful'] else None)
    return {**row, LABEL: added}


def enhance_batch(batch, *, allow_heldout=False):
    if batch.get('version') != BASE_VERSION:
        raise ValueError('Expected original audit_eval_bounds version; enhanced batches cannot be re-applied')
    if (batch.get('simulator_requests_sent') is not False
            or batch.get('truth_access') != 'post-termination records only'):
        raise ValueError('Expected completed offline audit provenance')
    old_rows = batch['rows']
    if not old_rows or batch['records'] != len(old_rows):
        raise ValueError('Base audit record count inconsistent')
    if not allow_heldout and any(type(r['seed']) is not int or not 6000 <= r['seed'] <= 6047 for r in old_rows):
        raise ValueError('Default scope is validation 6000..6047; no automatic held-out expansion')
    identities = {(r['case_sha256'], r['strategy']) for r in old_rows}
    scenarios = {r['case_sha256'] for r in old_rows}
    if len(identities) != len(old_rows) or batch['scenarios'] != len(scenarios):
        raise ValueError('Duplicate rows or inconsistent scenario count')
    rows = [enhance_row(row) for row in old_rows]
    groups = []
    for strategy in dict.fromkeys(r['strategy'] for r in rows):
        selected = [r for r in rows if r['strategy'] == strategy]
        valid = [r for r in selected if r['successful']]
        mean_time = statistics.mean(r['virtual_time_s'] for r in valid) if valid else None
        old_mean = statistics.mean(r['conditional_machine_lower_s'] for r in valid) if valid else None
        new_mean = statistics.mean(r[LABEL]['conditional_machine_lower_s'] for r in valid) if valid else None
        groups.append(dict(strategy=strategy, runs=len(selected), successes=len(valid),
            failures=len(selected)-len(valid), n16_rows=sum(r['source_total'] == 16 for r in selected),
            mean_success_time_s=mean_time, mean_success_base_conditional_lower_s=old_mean,
            mean_success_cited_conditional_lower_s=new_mean,
            ratio_of_success_means_base=mean_time/old_mean if valid else None,
            ratio_of_success_means_cited=mean_time/new_mean if valid else None,
            mean_bound_increment_s=statistics.mean(r[LABEL]['increment_machine_s'] for r in selected),
            minimum_increment_s=min(r[LABEL]['increment_machine_s'] for r in selected),
            maximum_increment_s=max(r[LABEL]['increment_machine_s'] for r in selected)))
    return dict(version=VERSION, comparison_label=LABEL, base_version=BASE_VERSION,
        optional_enhancement=True, original_default_bounds_modified=False,
        simulator_requests_sent=False, eval_gz_files_opened=0, dp_calls=0,
        truth_access='Existing post-termination batch audit facts only; no new ground truth read.',
        theorem_premise=theorem_premise(),
        bound_scope='Conditional on all-clear certification for every legal Q3 hidden scene; this tool does not prove that policy guarantee.',
        full_scene_policy_guarantee_established_by_observed_success=False,
        zero_failed_clear_observations_used_to_strengthen_bound=False,
        enhancement='For N<16, replace old empty-channel action term by old+3*(20-N); N=16 unchanged.',
        rounding='Preserve original per-physical-action rounding allowance; apply it once to enhanced continuous bound.',
        base_geometry_cache_recomputed_or_modified=False, allow_heldout=allow_heldout,
        records=len(rows), scenarios=len(scenarios), groups=groups, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='One explicitly supplied completed base audit JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-heldout', action='store_true', help='Only for an explicitly authorized, frozen held-out batch')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from the original audit file')
    started = time.perf_counter()
    contents = args.input.read_bytes()
    result = enhance_batch(json.loads(contents), allow_heldout=args.allow_heldout)
    result.update(source_batch_file=str(args.input), source_batch_sha256=hashlib.sha256(contents).hexdigest(),
                  enhancement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_s=time.perf_counter()-started)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Offline audit and paired comparison of the frozen 48 CPU development worlds.

No policies, Torch, simulator clients, scenario generators or networks are run.
The physical cache and post-termination records are the only numerical inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/theory_v1"))
from audit_eval_bounds import VERSION, audit_record, atomic_json, digest, load_cache

POLICIES = ("rl_trial1", "state_certified_tail", "state_v1")
SEEDS = list(range(2100001, 2100049))
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    first = math.floor(index)
    return ordered[first] + (ordered[min(first + 1, len(ordered) - 1)] - ordered[first]) * (index - first)


def paired_comparison(baseline, candidate, repeats=20000, seed=20260911):
    a = [r["virtual_time_s"] for r in baseline]
    b = [r["virtual_time_s"] for r in candidate]
    differences = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    seconds, percents = [], []
    for _ in range(repeats):
        indices = rng.choices(range(len(a)), k=len(a))
        denominator = sum(a[i] for i in indices)
        difference = sum(differences[i] for i in indices)
        seconds.append(difference / len(a))
        percents.append(100 * difference / denominator)
    breakdown = {key: statistics.mean(x[key] - y[key] for x, y in zip(baseline, candidate))
                 for key in COMPONENTS}
    assert math.isclose(sum(breakdown.values()), statistics.mean(differences), abs_tol=1e-6)
    return dict(pairs=len(a), pairing="identical case_sha256, aligned by frozen scenario seed",
                baseline=baseline[0]["policy"], candidate=candidate[0]["policy"],
                positive_savings_means="candidate faster than baseline",
                mean_savings_s=statistics.mean(differences),
                aggregate_savings_percent=100 * sum(differences) / sum(a),
                wins=sum(x > 1e-6 for x in differences), losses=sum(x < -1e-6 for x in differences),
                ties=sum(abs(x) <= 1e-6 for x in differences),
                bootstrap=dict(method="paired percentile bootstrap over scenarios", repeats=repeats, seed=seed,
                               confidence=0.95, mean_savings_s_ci=[quantile(seconds, .025), quantile(seconds, .975)],
                               aggregate_savings_percent_ci=[quantile(percents, .025), quantile(percents, .975)],
                               scope="Development-world uncertainty only; not independent model-selection or final-test evidence"),
                mean_component_savings_s=breakdown)


def analyze(output):
    manifest_path = output / "manifest.json"
    manifest = read(manifest_path)
    if manifest["seeds"] != SEEDS or set(manifest["policies"]) != set(POLICIES):
        raise ValueError("Analysis is restricted to the frozen three-policy 48-world development comparison")
    if not all((output / policy / "summary.json").is_file() for policy in POLICIES):
        raise ValueError("All three complete summaries must exist before analysis")
    cache_path = output / "physical_bound_cache.json"
    cache = load_cache(cache_path)
    archive_hashes = {}
    for name in POLICIES:
        policy = manifest["policies"][name]
        archive = output / (name + "-source.zip")
        archive_hashes[name] = sha(archive)
        if archive_hashes[name] != policy["archive_sha256"]:
            raise ValueError(f"Frozen source archive hash mismatch: {name}")
        with zipfile.ZipFile(archive) as source:
            for rel, expected in policy["source_sha256"].items():
                if hashlib.sha256(source.read(rel)).hexdigest() != expected:
                    raise ValueError(f"Frozen source file hash mismatch: {name}/{rel}")
        if any(policy["source_sha256"].get(rel) != expected for rel, expected in manifest["common_source_sha256"].items()):
            raise ValueError("Physical engine/evaluation source differs between strategies")
    remote = {}
    for seed in SEEDS:
        path = output / f"remote_trial1/case-{seed}.json.gz"
        if sha(path) != manifest["remote_records"][str(seed)]["record_sha256"]:
            raise ValueError("Remote evidence changed after freeze")
        remote[seed] = record(path)
        audit_record(remote[seed])
    groups, rows, all_records = {}, [], {}
    for name in POLICIES:
        paths = sorted((output / name).glob("case-*.json.gz"))
        if {p.name for p in paths} != {f"case-{s}.json.gz" for s in SEEDS}:
            raise ValueError(f"Missing or extra local records: {name}")
        group_rows, local_records = [], {}
        for path in paths:
            data = record(path)
            raw = data["row"]
            seed = raw["seed"]
            sources, moving_actions = audit_record(data)
            if (raw["case_sha256"] != remote[seed]["row"]["case_sha256"]
                    or raw["case_sha256"] != manifest["remote_records"][str(seed)]["case_sha256"]):
                raise ValueError("Local/remote world mismatch")
            if raw["strategy"] != manifest["policies"][name]["spec"]["name"]:
                raise ValueError("Unexpected policy label")
            geometry = [[c, sources[c]["x"], sources[c]["y"]] for c in sorted(sources)]
            cache_key = digest(dict(version=VERSION, geometry=geometry))
            if cache_key not in cache:
                raise ValueError("Frozen physical cache misses a development world")
            length = cache[cache_key]["source_route_lower_m"]
            n = len(sources)
            lower = length / 5 + 5 * n
            old_lower = lower + (30 * (20 - n) if n < 16 else 0)
            if not math.isclose(lower, cache[cache_key]["physical_clairvoyant_lower_s"], abs_tol=1e-8):
                raise ValueError("Physical cache is inconsistent")
            if raw["successful"] and raw["virtual_time_s"] + .5e-6 * moving_actions + 1e-6 < lower:
                raise ValueError("Successful trajectory violates physical lower bound")
            row = dict(policy=name, strategy=raw["strategy"], seed=seed, case_id=raw["case_id"],
                       case_sha256=raw["case_sha256"], source_total=n, successful=raw["successful"],
                       all_cleared=raw["all_cleared"], failed_clear_count=raw["failed_clear_count"],
                       virtual_time_s=raw["virtual_time_s"], penalized_time_s=raw["penalized_time_s"],
                       program_runtime_s=raw["program_runtime_s"], measurement_count=raw["measurement_count"],
                       action_count=raw["action_count"], **{k: raw[k] for k in COMPONENTS},
                       source_route_lower_m=length, primary_lower_bound_s=lower,
                       previous_rl_lower_bound_s=old_lower,
                       time_over_primary_lower=raw["virtual_time_s"] / lower if raw["successful"] else None,
                       time_over_previous_rl_lower=raw["virtual_time_s"] / old_lower if raw["successful"] else None,
                       input_sha256=sha(path), geometry_cache_key=cache_key)
            group_rows.append(row)
            local_records[seed] = data
        group_rows.sort(key=lambda row: row["seed"])
        times = [r["virtual_time_s"] for r in group_rows]
        all_success = all(r["successful"] for r in group_rows)
        summary = read(output / name / "summary.json")
        if (summary["runs"] != 48 or not summary["complete"]
                or summary["successful_runs"] != sum(r["successful"] for r in group_rows)
                or not math.isclose(summary["raw_mean_total_time_s"], statistics.mean(times), abs_tol=1e-8)):
            raise ValueError("Stored summary disagrees with audited records")
        groups[name] = dict(runs=48, successful_runs=sum(r["successful"] for r in group_rows),
            all_clear_runs=sum(r["all_cleared"] for r in group_rows),
            failed_clear_count=sum(r["failed_clear_count"] for r in group_rows),
            mean_time_s=statistics.mean(times), p95_time_s=quantile(times, .95), max_time_s=max(times),
            sum_time_s=sum(times), sum_primary_lower_bound_s=sum(r["primary_lower_bound_s"] for r in group_rows),
            sum_previous_rl_lower_bound_s=sum(r["previous_rl_lower_bound_s"] for r in group_rows),
            mean_primary_lower_bound_s=statistics.mean(r["primary_lower_bound_s"] for r in group_rows),
            mean_previous_rl_lower_bound_s=statistics.mean(r["previous_rl_lower_bound_s"] for r in group_rows),
            mean_program_runtime_s=statistics.mean(r["program_runtime_s"] for r in group_rows),
            mean_measurement_count=statistics.mean(r["measurement_count"] for r in group_rows),
            mean_action_count=statistics.mean(r["action_count"] for r in group_rows),
            mean_time_breakdown_s={k: statistics.mean(r[k] for r in group_rows) for k in COMPONENTS},
            ratios_eligible=all_success,
            total_time_over_primary_lower=sum(times) / sum(r["primary_lower_bound_s"] for r in group_rows) if all_success else None,
            total_time_over_previous_rl_lower=sum(times) / sum(r["previous_rl_lower_bound_s"] for r in group_rows) if all_success else None)
        rows.extend(group_rows)
        all_records[name] = local_records
    by_policy = {name: [r for r in rows if r["policy"] == name] for name in POLICIES}
    rl = all_records["rl_trial1"]
    drift = [rl[s]["row"]["virtual_time_s"] - remote[s]["row"]["virtual_time_s"] for s in SEEDS]
    reproduction = dict(pairs=48, same_world_hashes=48,
        exact_virtual_time_matches=sum(rl[s]["row"]["virtual_time_s"] == remote[s]["row"]["virtual_time_s"] for s in SEEDS),
        exact_history_matches=sum(rl[s]["history"] == remote[s]["history"] for s in SEEDS),
        local_mean_time_s=groups["rl_trial1"]["mean_time_s"],
        remote_mean_time_s=statistics.mean(remote[s]["row"]["virtual_time_s"] for s in SEEDS),
        mean_local_minus_remote_s=statistics.mean(drift), max_absolute_time_difference_s=max(map(abs, drift)),
        mean_absolute_time_difference_s=statistics.mean(map(abs, drift)),
        note="Main comparison uses this machine's common frozen engine. Remote acceleration/platform is not trajectory-equivalent. The engine hashes float.hex positions for fixed bearing errors, so one-bit coordinate differences can change later observations; first-divergence evidence is described in the batch README.")
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope=manifest["scope"],
        cases=48, local_records=144, remote_records=48, audit_passed_records=192,
        all_paired_case_hashes_match=True, common_physics_sources_verified=True,
        lower_bounds=dict(primary="L/5 + 5*N", previous_rl="primary + 30*(20-N) if N<16, otherwise primary",
            route_scope="L is the cached exact graph-DP lower bound for visiting clearance disks, not the attainable unknown-scene optimum",
            ratio_aggregation="sum(time)/sum(per-case lower bound), not mean of case ratios",
            primary_is_main_comparison=True, cache_misses=0),
        groups=groups, rl_vs_certified_tail=paired_comparison(by_policy["state_certified_tail"], by_policy["rl_trial1"]),
        rl_vs_state_v1=paired_comparison(by_policy["state_v1"], by_policy["rl_trial1"]),
        remote_reproduction=reproduction, manifest_sha256=sha(manifest_path), cache_sha256=sha(cache_path),
        source_archive_sha256=archive_hashes, analysis_source_sha256=sha(__file__),
        trajectory_audit_source_sha256=sha(ROOT / "research/theory_v1/audit_eval_bounds.py"))
    csv_path = output / "per_case.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    atomic_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paired_rl_state_20260911")
    args = parser.parse_args()
    print(json.dumps(analyze(args.output.resolve()), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

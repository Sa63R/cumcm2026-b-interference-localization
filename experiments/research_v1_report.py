"""Read-only paired reports from complete, identity-recorded Q3 evaluations.

Does not simulate cases, load policy weights, or alter evaluation records.
Final-suite acceptance requires both predeclared final partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.research_v1_eval import (
    PROTOCOL, digest, paired_comparison, percentile, read_json, summarize, write_json,
)

COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")


def load_evaluation(directory, protocol):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    rows = read_json(directory / "rows.json")
    split = manifest["split"]
    if split == "future_reserved" or split not in protocol["partitions"]:
        raise ValueError("Unsupported evaluation partition")
    partition = protocol["partitions"][split]
    expected = list(range(partition["seed_start"], partition["seed_stop_exclusive"]))
    if manifest["seeds"] != expected:
        raise ValueError("Manifest does not cover the complete declared partition")
    if manifest["identity"]["protocol_sha256"] != digest(protocol):
        raise ValueError("Evaluation protocol differs from report protocol")
    if sorted(row["seed"] for row in rows) != expected:
        raise ValueError("Missing, duplicate, or unexpected case seed")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case id")
    for row in rows:
        if row["strategy"] != manifest["strategy"]:
            raise ValueError("Row strategy does not match manifest")
        values = [row[key] for key in (*COMPONENTS, "virtual_time_s", "penalized_time_s", "program_runtime_s")]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Non-finite or negative evaluation metric")
        certified = (row["all_cleared"] and row["completion_certified"] and row["accepted_exit"]
                     and row["cleared_total"] == row["source_total"] and not row["errors"])
        if bool(row["successful"]) != bool(certified):
            raise ValueError("Success flag contradicts recorded completion facts")
        if not row["successful"] and row["penalized_time_s"] != protocol["limits"]["virtual_seconds_per_case"]:
            raise ValueError("Failed case does not retain the declared time penalty")
        if row["successful"] and row["penalized_time_s"] != row["virtual_time_s"]:
            raise ValueError("Successful case has an unexplained penalty")
    return dict(directory=str(directory), manifest=manifest, rows=rows,
                rows_sha256=hashlib.sha256((directory / "rows.json").read_bytes()).hexdigest())


def compare_directories(baseline, candidates, protocol):
    base = load_evaluation(baseline, protocol)
    report = dict(schema_version=1, split=base["manifest"]["split"],
                  protocol_sha256=digest(protocol), baseline=base["manifest"]["strategy"],
                  inputs={}, methods={}, comparisons={}, final_suite_acceptance=None,
                  evidence_scope="A single partition report is not full final-suite acceptance.",
                  runtime_scope="Recorded wall time may include concurrent load; use a serial benchmark for speed claims.")
    for label, item in [("baseline", base)] + [(name, load_evaluation(path, protocol)) for name, path in candidates.items()]:
        if item["manifest"]["split"] != report["split"]:
            raise ValueError("Cannot pair different evaluation partitions")
        report["inputs"][label] = {k: v for k, v in item.items() if k != "rows"}
        rows = item["rows"]
        report["methods"][label] = dict(
            strategy=item["manifest"]["strategy"], **summarize(rows, len(base["rows"])),
            penalized_p95_total_time_s=percentile([row["penalized_time_s"] for row in rows], .95),
            mean_components_s={key: statistics.mean(row[key] for row in rows) for key in COMPONENTS},
            mean_measurement_count=statistics.mean(row["measurement_count"] for row in rows),
            failure_case_ids=[row["case_id"] for row in rows if not row["successful"] or row["failed_clear_count"]],
        )
        if label == "baseline":
            continue
        contrast = paired_comparison(base["rows"], rows, protocol)
        criteria = protocol["acceptance"]
        contrast["performance_target_met_on_supplied_cases"] = bool(
            contrast["all_pairs_successful_no_candidate_failed_clear"]
            and contrast["mean_reduction_fraction"] >= criteria["minimum_mean_total_time_reduction_fraction"]
            and contrast["saving_ci95_s"][0] > criteria["paired_saving_bootstrap_ci95_lower_seconds_strictly_above"]
            and contrast["p95_time_ratio"] <= criteria["maximum_p95_total_time_ratio"])
        by_id = {row["case_id"]: row for row in base["rows"]}
        contrast["paired_cases"] = sorted([
            dict(case_id=row["case_id"], seed=row["seed"], baseline_s=by_id[row["case_id"]]["penalized_time_s"],
                 candidate_s=row["penalized_time_s"], saved_s=by_id[row["case_id"]]["penalized_time_s"]-row["penalized_time_s"],
                 candidate_successful=row["successful"], failed_clear_count=row["failed_clear_count"])
            for row in rows], key=lambda row: row["saved_s"])
        report["comparisons"][label] = contrast
    return report


def final_acceptance(random_report, stress_report, protocol):
    """Require matching frozen identities across both complete final partitions."""
    if random_report["split"] != "final_random" or stress_report["split"] != "final_stress":
        raise ValueError("Final acceptance requires random and stress final reports")
    if random_report["protocol_sha256"] != digest(protocol) or stress_report["protocol_sha256"] != digest(protocol):
        raise ValueError("Final report protocol mismatch")
    if random_report["inputs"].keys() != stress_report["inputs"].keys():
        raise ValueError("Final reports contain different methods")
    expected = [protocol["partitions"][split]["seed_stop_exclusive"]-protocol["partitions"][split]["seed_start"]
                for split in ("final_random", "final_stress")]
    baseline_valid = all(report["methods"]["baseline"]["complete"]
                         and report["methods"]["baseline"]["runs"] == count
                         and report["methods"]["baseline"]["successful_runs"] == count
                         for report, count in zip((random_report, stress_report), expected))
    results = {}
    for label in random_report["inputs"]:
        if random_report["inputs"][label]["manifest"]["identity"] != stress_report["inputs"][label]["manifest"]["identity"]:
            raise ValueError("Code, configuration or checkpoint changed between final partitions")
        if label == "baseline":
            continue
        random_method, stress_method = (report["methods"][label] for report in (random_report, stress_report))
        complete = all(method["complete"] and method["runs"] == count for method, count in zip((random_method, stress_method), expected))
        reliable = all(method["successful_runs"] == method["runs"] and method["failed_clear_count"] == 0
                       for method in (random_method, stress_method))
        random_performance = random_report["comparisons"][label]["performance_target_met_on_supplied_cases"]
        # Stress families are not an IID draw from the random research prior.
        # Report their tails separately; never pool them to improve random CI.
        stress_tail = stress_report["comparisons"][label]["p95_time_ratio"] <= protocol["acceptance"]["maximum_p95_total_time_ratio"]
        results[label] = dict(baseline_complete_and_successful=baseline_valid,
                              all_final_cases_complete=complete, all_final_cases_reliable=reliable,
                              random_performance_target=random_performance, stress_p95_guard=stress_tail,
                              first_version_practical_target_met=baseline_valid and complete and reliable and random_performance and stress_tail)
    return dict(schema_version=1, acceptance=results,
                scope="Random paired performance and separate stress robustness; no official-simulator claim or global-optimality claim.")


def markdown(report):
    lines = [f"# Q3 paired {report['split']} report", "", report["evidence_scope"], "",
             "| Method | Success | Failed clears | Mean s | P95 s | Savings | 95% CI saved s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, method in report["methods"].items():
        contrast = report["comparisons"].get(label)
        saving = f"{100*contrast['mean_reduction_fraction']:.2f}%" if contrast else "reference"
        ci = " / ".join(f"{x:.2f}" for x in contrast["saving_ci95_s"]) if contrast else "-"
        lines.append(f"| {label} | {method['successful_runs']}/{method['runs']} | {method['failed_clear_count']} | "
                     f"{method['penalized_mean_total_time_s']:.2f} | {method['penalized_p95_total_time_s']:.2f} | {saving} | {ci} |")
    lines += ["", "Means, P95 and paired comparisons retain failed cases with the predeclared penalty.",
              "", report["runtime_scope"], "", "## Worst paired regressions", ""]
    for label, contrast in report["comparisons"].items():
        lines.append(f"- {label}: " + "; ".join(f"{case['case_id']} ({-case['saved_s']:+.2f} s)" for case in contrast["paired_cases"][:3]))
    return "\n".join(lines) + "\n"


def figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(report["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    bottoms = [0.0] * len(methods)
    for key in COMPONENTS:
        values = [report["methods"][method]["mean_components_s"][key] for method in methods]
        axes[0].bar(methods, values, bottom=bottoms, label=key.removesuffix("_s"))
        bottoms = [a+b for a, b in zip(bottoms, values)]
    penalties = [max(0., report["methods"][method]["penalized_mean_total_time_s"]-bottom)
                 for method, bottom in zip(methods, bottoms)]
    if max(penalties) > 1e-5:
        axes[0].bar(methods, penalties, bottom=bottoms, color="lightgray", hatch="//", label="failure penalty")
    axes[0].set(ylabel="Mean penalized virtual time (s)", title="Time components (failures retained)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    for label, contrast in report["comparisons"].items():
        saved = [case["saved_s"] for case in contrast["paired_cases"]]
        axes[1].plot(range(1, len(saved)+1), saved, label=label)
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set(xlabel="Sorted case rank within each method", ylabel="Paired penalized seconds saved",
                title=f"{report['split']}: positive means faster")
    axes[1].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", action="append", help="label=directory; repeat for each method")
    parser.add_argument("--final-random-report", type=Path)
    parser.add_argument("--final-stress-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.final_random_report or args.final_stress_report:
        if not (args.final_random_report and args.final_stress_report) or args.baseline or args.candidate or args.figures:
            parser.error("Final acceptance takes exactly two reports, --output, and no per-partition options")
        report = final_acceptance(read_json(args.final_random_report), read_json(args.final_stress_report), read_json(PROTOCOL))
        write_json(args.output / "final_acceptance.json", report)
        print(json.dumps(report, indent=2))
        return
    if not args.baseline or not args.candidate:
        parser.error("Paired report requires --baseline and one or more --candidate")
    candidates = {}
    for entry in args.candidate:
        label, path = entry.split("=", 1)
        if not label or label == "baseline" or label in candidates:
            parser.error("Candidate labels must be unique, nonempty, and different from baseline")
        candidates[label] = Path(path)
    report = compare_directories(args.baseline, candidates, read_json(PROTOCOL))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", report)
    (args.output / "comparison.md").write_text(markdown(report), encoding="utf-8")
    if args.figures:
        figures(report, args.output)
    print(json.dumps({label: {k: v for k, v in result.items() if k != "paired_cases"}
                      for label, result in report["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()

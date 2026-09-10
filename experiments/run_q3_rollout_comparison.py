"""Fixed-case Q3 rollout experiments against the current default efficient policy."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.run_q3_comparison import bootstrap_mean_interval, percentile
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies import run_search
from strategies.rollout import RolloutConfig

BASELINE = "baseline_efficient"
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def source_hashes():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "experiments/run_q3_comparison.py"]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_configs(path):
    raw = path.read_bytes()
    entries = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Configs must be a nonempty list of name/rollout_config objects")
    names = {BASELINE}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "rollout_config"}:
            raise ValueError("Each entry must contain exactly name and rollout_config")
        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or not name.isprintable() or name in names:
            raise ValueError("Configuration names must be printable, nonempty, unique, and not reserved")
        if not isinstance(entry["rollout_config"], dict):
            raise ValueError("rollout_config must be an object")
        RolloutConfig.parse(entry["rollout_config"])
        names.add(name)
    canonical(entries)  # Fail before a run for nonfinite JSON configuration values.
    return entries, raw


def trace_path(case_id, name):
    token = hashlib.sha256((case_id + "\0" + name).encode("utf-8")).hexdigest()
    return "traces/" + token + ".json.gz"


def save_trace(output, record):
    destination = output / record["row"]["trace"]
    if destination.exists():
        raise ValueError(f"Completed trace already exists: {destination}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_case(case, name, config, max_actions):
    simulator = LocalResearchSimulator(case)
    client = simulator.client()
    result, errors, diagnostic = None, [], None
    interrupted = False
    started = time.perf_counter()
    try:
        arguments = {"problem": 3, "variant": "efficient" if name == BASELINE else "rollout",
                     "max_actions": max_actions}
        if name != BASELINE:
            arguments["rollout_config"] = json.loads(canonical(config))
        # Neither case, seed, simulator, source count nor truth is an argument.
        # The baseline deliberately receives NO efficient_config override.
        result = run_search(client, **arguments)
        errors += [str(value) for value in (result.error, result.exit_error) if value]
    except KeyboardInterrupt:
        interrupted = True
        errors.append("KeyboardInterrupt: experiment interrupted")
        diagnostic = traceback.format_exc()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        diagnostic = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {type(exc).__name__}: {exc}")
        simulator.finish_for_evaluation()
    elapsed = time.perf_counter() - started
    evaluation = simulator.evaluation()  # Only after legal exit or harness termination.
    summary = result.as_dict() if result else None
    planning = (summary.get("planning") or {}) if summary else {}
    certified = bool(result and result.completion_certified_under_model)
    if client.state.cleared_count != evaluation["cleared_total"]:
        errors.append("InvariantViolation: client count differs from evaluator")
    if result and (result.cleared_count != evaluation["cleared_total"]
                   or not math.isclose(result.virtual_time_s, evaluation["virtual_time_s"], abs_tol=1e-6)):
        errors.append("InvariantViolation: strategy count/time differs from evaluator")
    if not math.isclose(evaluation["virtual_time_s"], sum(evaluation["time_breakdown_s"].values()), abs_tol=1e-6):
        errors.append("InvariantViolation: virtual-time decomposition differs")
    if certified and not evaluation["all_cleared"]:
        errors.append("InvariantViolation: false completeness certificate")
    exited = client.state.session == "exited" and client.pending_request is None
    success = bool(evaluation["all_cleared"] and certified and exited and not errors)
    row = {
        "problem": 3, "case_id": case.case_id, "case_kind": "random" if "-random-" in case.case_id else "hard",
        "seed": case.seed, "strategy": name, "source_total": evaluation["source_total"],
        "cleared_total": evaluation["cleared_total"], "cleared_fraction": evaluation["cleared_fraction"],
        "all_cleared": evaluation["all_cleared"], "completion_certified": certified,
        "successful": success, "accepted_exit": exited, "interrupted": interrupted,
        "completion_reason": result.completion_reason if result else "uncaught_exception",
        "error": " | ".join(errors), "virtual_time_s": evaluation["virtual_time_s"],
        "time_per_source_s": evaluation["virtual_time_s"] / evaluation["source_total"],
        "mean_time_per_cleared_s": evaluation["mean_time_per_cleared_s"], "program_runtime_s": elapsed,
        "measurement_count": evaluation["measurement_count"], "failed_clear_count": evaluation["failed_clear_count"],
        "action_count": evaluation["action_count"], "movement_m": 5 * evaluation["time_breakdown_s"]["movement_s"],
        **evaluation["time_breakdown_s"], "case_sha256": value_hash(case.evaluation_config()),
        "planning_searches": planning.get("searches", 0),
        "planning_candidate_evaluations": planning.get("candidate_evaluations", 0),
        "planning_changed_decisions": planning.get("changed_decisions", 0),
        "planning_wall_time_s": planning.get("planning_wall_time_s", 0.0),
        "planning_fallback_counts": planning.get("fallback_counts", {}),
        "config_sha256": value_hash(config), "trace": trace_path(case.case_id, name),
    }
    return {"data_origin": "synthetic_research", "row": row,
            "summary": summary, "evaluation": evaluation,
            "evaluation_phase": "after_session_termination", "history": simulator.observation_history(),
            "final_client_state": client.state.snapshot(), "config": config, "exception_traceback": diagnostic}


def summaries(rows):
    groups = []
    baselines = {row["case_id"]: row for row in rows if row["strategy"] == BASELINE}
    for kind in ("random", "hard"):
        for strategy in dict.fromkeys(row["strategy"] for row in rows):
            subset = [row for row in rows if row["case_kind"] == kind and row["strategy"] == strategy]
            if not subset:
                continue
            times = [row["virtual_time_s"] for row in subset]
            pairs = [(baselines[row["case_id"]], row) for row in subset
                     if row["case_id"] in baselines and row["successful"] and baselines[row["case_id"]]["successful"]]
            savings = [base["virtual_time_s"] - row["virtual_time_s"] for base, row in pairs]
            means = [row["mean_time_per_cleared_s"] for row in subset if row["mean_time_per_cleared_s"] is not None]
            successful = [row for row in subset if row["successful"] and row["all_cleared"]]
            tail_count = max(1, math.ceil(len(times) * .10))
            fallbacks = Counter()
            for row in subset:
                fallbacks.update(row["planning_fallback_counts"])
            groups.append({
                "case_kind": kind, "strategy": strategy, "runs": len(subset),
                "successful_runs": sum(row["successful"] for row in subset),
                "failed_runs": sum(not row["successful"] for row in subset),
                "all_clear_runs": sum(row["all_cleared"] for row in subset),
                "mean_virtual_time_s": statistics.mean(times), "median_virtual_time_s": statistics.median(times),
                "successful_mean_virtual_time_s": statistics.mean(row["virtual_time_s"] for row in successful) if successful else None,
                "successful_mean_time_per_source_s": statistics.mean(row["time_per_source_s"] for row in successful) if successful else None,
                "all_run_time_note": "Unprefixed timing summaries include failed runs, whose early termination can reduce recorded time; they are not evidence of superior performance. Successful-prefixed means require complete successful clearance.",
                "p90_virtual_time_s": percentile(times, .90), "p95_virtual_time_s": percentile(times, .95),
                "max_virtual_time_s": max(times),
                "upper_tail_10pct_mean_virtual_time_s": statistics.mean(sorted(times, reverse=True)[:tail_count]),
                "mean_case_time_per_cleared_s": statistics.mean(means) if means else None,
                "mean_runtime_s": statistics.mean(row["program_runtime_s"] for row in subset),
                "p95_runtime_s": percentile([row["program_runtime_s"] for row in subset], .95),
                "mean_measurements": statistics.mean(row["measurement_count"] for row in subset),
                "mean_failed_clears": statistics.mean(row["failed_clear_count"] for row in subset),
                "mean_movement_m": statistics.mean(row["movement_m"] for row in subset),
                "mean_time_breakdown_s": {key: statistics.mean(row[key] for row in subset) for key in COMPONENTS},
                "planning_total_searches": sum(row["planning_searches"] for row in subset),
                "planning_total_candidate_evaluations": sum(row["planning_candidate_evaluations"] for row in subset),
                "planning_total_changed_decisions": sum(row["planning_changed_decisions"] for row in subset),
                "planning_mean_wall_time_s": statistics.mean(row["planning_wall_time_s"] for row in subset),
                "planning_max_wall_time_s": max(row["planning_wall_time_s"] for row in subset),
                "planning_runs_with_no_changed_decisions": sum(row["planning_changed_decisions"] == 0 for row in subset),
                "planning_fallback_counts": dict(sorted(fallbacks.items())),
                "successful_pairs": len(pairs), "paired_mean_seconds_saved": statistics.mean(savings) if savings else None,
                "paired_mean_saving_ci95_s": bootstrap_mean_interval(savings) if kind == "random" else None,
                "paired_wins": sum(value > 1e-6 for value in savings),
                "paired_losses": sum(value < -1e-6 for value in savings),
                "paired_ties": sum(abs(value) <= 1e-6 for value in savings),
                "all_rows_eligible_for_time_comparison": len(pairs) == len(subset),
                "paired_note": "Positive savings/wins are relative to the default efficient policy; only successful pairs enter the CI and win/loss counts. Failed runs remain in the full table.",
            })
    return groups


def write_tables(output, rows):
    baselines = {row["case_id"]: row for row in rows if row["strategy"] == BASELINE}
    if rows:
        fields = list(rows[0]) + ["paired_baseline_time_s", "paired_seconds_saved", "paired_win", "pair_successful"]
        with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                item = dict(row)
                item["planning_fallback_counts"] = canonical(item["planning_fallback_counts"])
                base = baselines.get(row["case_id"])
                if base:
                    saved = base["virtual_time_s"] - row["virtual_time_s"]
                    valid_pair = base["successful"] and row["successful"]
                    item.update(paired_baseline_time_s=base["virtual_time_s"], paired_seconds_saved=saved,
                                paired_win=saved > 1e-6 if valid_pair else None, pair_successful=valid_pair)
                writer.writerow(item)


def execute(args):
    configs, config_raw = read_configs(args.configs)
    if "rollout_config" not in inspect.signature(run_search).parameters:
        raise ValueError("The rollout_config strategy interface is not available yet")
    seed_start = args.random_seed_start if args.random_seed_start is not None else (5000 if args.stage == "holdout" else 4000)
    if args.random_count < 1 or seed_start < 0 or args.max_actions < 2:
        raise ValueError("Require random-count >= 1, random-seed-start >= 0, max-actions >= 2")
    cases = [random_scenario(3, seed) for seed in range(seed_start, seed_start + args.random_count)]
    if args.include_hard:
        cases.extend(difficult_scenarios(3))
    case_configs = [case.evaluation_config() for case in cases]
    hashes = source_hashes()
    fixed = {"stage": args.stage, "random_count": args.random_count,
             "random_seed_interval_inclusive": [seed_start, seed_start + args.random_count - 1],
             "include_hard": args.include_hard, "configs": configs, "config_content_sha256": value_hash(configs),
             "baseline": {"name": BASELINE, "variant": "efficient", "efficient_config_override": None},
             "max_actions": args.max_actions, "cases_sha256": value_hash(case_configs), "source_sha256_start": hashes}
    if args.output.exists():
        if not args.allow_existing:
            raise ValueError("Output exists; pass --allow-existing only to resume the same fixed comparison")
        manifest_path = args.output / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Existing output has no valid experiment manifest")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in fixed.items()):
            raise ValueError("Cannot resume: source hashes, cases, configuration, or experiment settings differ")
        if value_hash(json.loads((args.output / "cases.json").read_text(encoding="utf-8"))) != fixed["cases_sha256"]:
            raise ValueError("Archived fixed scenarios changed")
    else:
        args.output.mkdir(parents=True)
        (args.output / "traces").mkdir()
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unavailable"
        metadata = {
            "data_origin": "synthetic_research", "official_practice": False, "official_formal": False,
            "problem": 3, "created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": commit,
            "python": sys.version, "platform": platform.platform(), **fixed,
            "original_config_file_sha256": hashlib.sha256(config_raw).hexdigest(),
            "belief_prior": "The particle belief prior is a modelling assumption, not a verified official source-generation law.",
            "algorithm_note": "Candidate-plan rollout with particle beliefs; not POMCP and not a claim of POMCP equivalence.",
            "runtime_scope": "In-process local simulator plus solver/planning and cleanup; excludes evaluation/serialization. Official HTTP/server runtime can differ.",
            "comparison": "Same fixed source case and location-fixed error function for every policy; hidden truth is passed only to the local engine, never to the policy.",
            "holdout_note": "Stage and seed ranges are explicit; this script does not tune or choose the best config. Independently select unused holdout seeds before evaluating; shared hard cases are not independent holdout data.",
            "resume_note": "Allow-existing resumes identical settings and preserves completed records, including failed/interrupted attempts; it does not rerun or overwrite them.",
            "bootstrap": {"samples": 2000, "seed": 20260910, "unit": "paired random case", "interval": "percentile 95%"},
        }
        (args.output / "configs.json").write_bytes(config_raw)
        write_json(args.output / "cases.json", case_configs)
    policies = [(BASELINE, None)] + [(item["name"], item["rollout_config"]) for item in configs]
    rows, completed = [], set()
    for case in cases:
        for name, config in policies:
            path = args.output / trace_path(case.case_id, name)
            if path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    record = json.load(stream)
                row = record["row"]
                if (row["case_id"] != case.case_id or row["strategy"] != name
                        or row["case_sha256"] != value_hash(case.evaluation_config())
                        or row["config_sha256"] != value_hash(config)
                        or row["trace"] != trace_path(case.case_id, name)):
                    raise ValueError("Saved trace does not match the fixed case/config identity")
                rows.append(row)
                completed.add((case.case_id, name))
    metadata.update(run_status="running", expected_runs=len(cases) * len(policies))
    write_json(args.output / "manifest.json", metadata)
    interrupted = False
    try:
        for case in cases:
            for name, config in policies:
                if (case.case_id, name) in completed:
                    continue
                record = run_case(case, name, config, args.max_actions)
                save_trace(args.output, record)
                rows.append(record["row"])
                completed.add((case.case_id, name))
                write_tables(args.output, rows)
                row = record["row"]
                print(f"[{len(rows)}/{metadata['expected_runs']}] {case.case_id} {name}: "
                      f"{row['cleared_total']}/{row['source_total']} {row['virtual_time_s']:.3f}s virtual "
                      f"{row['program_runtime_s']:.3f}s wall success={row['successful']}", flush=True)
                if row["interrupted"]:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        interrupted = True
    finally:
        final_hashes = source_hashes()
        metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), completed_runs=len(rows),
                        source_sha256_end=final_hashes, source_changed_during_run=hashes != final_hashes,
                        run_status="interrupted" if interrupted else "completed" if len(rows) == metadata["expected_runs"] else "failed")
        write_json(args.output / "manifest.json", metadata)
        write_tables(args.output, rows)
        write_json(args.output / "summary.json", {"data_origin": "synthetic_research", "stage": args.stage,
                   "expected_runs": metadata["expected_runs"], "completed_runs": len(rows),
                   "source_consistent": hashes == final_hashes, "groups": summaries(rows)})
    if interrupted:
        return 130
    if hashes != final_hashes:
        return 2
    return 0 if len(rows) == metadata["expected_runs"] and all(row["successful"] for row in rows) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=12)
    parser.add_argument("--random-seed-start", type=int)
    parser.add_argument("--include-hard", action="store_true", help="Also evaluate the seven shared deterministic hard cases")
    parser.add_argument("--allow-existing", action="store_true", help="Resume matching fixed settings; preserve existing per-run records")
    parser.add_argument("--stage", choices=("development", "holdout"), default="development")
    parser.add_argument("--max-actions", type=int, default=20000)
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

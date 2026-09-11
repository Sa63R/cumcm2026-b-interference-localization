"""CPU-only mechanism checks on two fixed training cases, with physical T/LB.

This exercises new actions and termination; it is NOT a strategy selection or
an estimate of trained performance. No database or official simulator is used.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research/theory_v1")]

from experiments.research_v1_eval import run_case, write_json
from research_rl import run_rl_search
from research_rl.action_sets import action_schema
from simulation import random_scenario
from audit_eval_bounds import audit_record, physical_bounds


class ObservedClient:
    __slots__ = ("_client",)

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        if name not in {"state", "remaining_real_time_s", "pending_request",
                        "enter", "measure", "clear", "exit"}:
            raise AssertionError(f"Policy attempted non-observation access: {name}")
        return getattr(self._client, name)


def run_exercise(client, *, mode="base", selection="teacher", **kwargs):
    def selector(features, context, teacher):
        if selection == "exercise":
            return next((i for i, row in enumerate(features) if row[0] == 1 and row[18] > 1), teacher)
        return teacher
    return run_rl_search(ObservedClient(client), policy=selector, feature_version="v3",
                         probe_candidates=mode, **kwargs)


def check_extra_evidence(record):
    """Reconstruct accepted position/channel evidence without controller state."""
    summary = record["summary"]
    learning = summary["learning"]
    measured, detected, cleared = defaultdict(set), set(), set()
    extras = 0
    previous = (0.0, 0.0)
    for action in summary["action_history"]:
        point, channel = tuple(action["position"]), action["channel"]
        family = action.get("rl_discovery_family")
        if family:
            if (action["action"] != "measure" or channel in detected | cleared
                    or channel in measured[point] or len(detected | cleared) >= 16):
                raise ValueError("New discovery action lacks fresh unknown-channel evidence")
            if family == "current" and point != previous:
                raise ValueError("Current-only action moved")
            if family not in ("current", "target"):
                raise ValueError("Unknown discovery action family")
            extras += 1
        if action["action"] == "measure":
            measured[point].add(channel)
            if action["result"] in ("near", "direction"):
                detected.add(channel)
        elif action["result"] == "success":
            cleared.add(channel)
        previous = point
    if learning.get("anypoint_measurements", 0) != extras or extras > 32:
        raise ValueError("New measurement budget/accounting mismatch")
    if "measurement_ledger" in learning:
        saved = {tuple(entry["position"]): set(entry["channels"])
                 for entry in learning["measurement_ledger"]}
        if saved != dict(measured):
            raise ValueError("Ledger differs from actual measurements")
    if summary["coverage_complete"] or (record["row"]["successful"] and len(cleared) < 16):
        if any(measured[tuple(point)] | cleared != set(range(1, 21))
               for point in summary["coverage_points"]):
            raise ValueError("Extra-site observations forged seven-site coverage")
    return extras


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output directory; existing audit evidence is never overwritten")
    args.output.mkdir(parents=True)
    protocol = {"limits": {"max_actions": 20000, "real_seconds_per_case": 60,
                           "virtual_seconds_per_case": 360000}}
    specs = [dict(name="base_teacher", entrypoint="research.audit_anypoint_scan:run_exercise",
                  kwargs={"mode": "base", "selection": "teacher"})]
    for mode in ("anypoint_current", "anypoint_targets"):
        for selection in ("teacher", "exercise"):
            specs.append(dict(name=f"{mode}_{selection}", entrypoint="research.audit_anypoint_scan:run_exercise",
                              kwargs={"mode": mode, "selection": selection}))
    specs.append(dict(name="anypoint_targets_fallback3", entrypoint="research.audit_anypoint_scan:run_exercise",
                      kwargs={"mode": "anypoint_targets", "selection": "exercise", "max_decisions": 3}))
    rows, files = [], {}
    for seed in (110402, 110403):
        case = random_scenario(3, seed)
        bound = None
        for spec in specs:
            record = run_case(case, spec, protocol)
            sources, _ = audit_record(record)  # First truth access is after exit.
            if bound is None:
                bound = physical_bounds(sources.values())["physical_clairvoyant_lower_s"]
            extra = check_extra_evidence(record)
            name = f"{seed}-{spec['name']}.eval.json.gz"
            target = args.output / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False)
            temporary.replace(target)
            files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
            row = dict(record["row"], physical_lower_bound_s=bound,
                       time_over_physical_lower_bound=record["row"]["virtual_time_s"] / bound,
                       anypoint_measurements=extra,
                       fallback_counts=record["summary"]["learning"]["fallback_counts"])
            rows.append(row)
            print(f"{seed} {spec['name']}: T={row['virtual_time_s']:.6f}s LB={bound:.6f}s "
                  f"T/LB={row['time_over_physical_lower_bound']:.6f} extra={extra} "
                  f"success={row['successful']}", flush=True)
    result = dict(
        purpose="mechanism and physical-accounting smoke only; no trained-performance or selection claim",
        seeds=[110402, 110403], access="CPU-only local synthetic training cases; no official/database/final-set access",
        lower_bound="exact open subset-DP over origin/20m source-disk graph; LB=L/5+5N; no empty-channel term",
        schemas=[action_schema(mode) for mode in ("anypoint_current", "anypoint_targets")],
        checks="accepted-action physics, feedback, microsecond costs, exact evidence, no phantom cover, bounded new actions",
        specs=specs, rows=rows, record_sha256=files,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output / "summary.json", result)
    return 0 if all(row["successful"] and row["failed_clear_count"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Separately frozen all-directional Q4 validation; never changes the generator.

Freeze is a read-only planning step for scenarios. Only the explicit run command
can generate reserved cases. Policies receive the existing observation wrapper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import importlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from simulation.cases import Scenario, Source
from simulation.engine import LocalResearchSimulator
from experiments.run_q4_state_study import ObservationOnlyClient, digest, write_json
from experiments.run_q4_round2 import BASE, BASE_SPEC, hashes as common_hashes, report_rows, audit as common_audit
from experiments.q4_comparison_bounds import common_bound
import hashlib

COMBO = "compact_combo"
COMBO_SPEC = {"entrypoint": "strategies.q4_range_scheduling:run_q4_range_scheduling",
              "kwargs": {"config": "onroute", "max_expansions": 200}}
FAMILIES = ("minimum_radius", "boundary_outward", "cluster", "positive_error",
            "negative_error", "alternating_error", "narrow_strip")
PROTOCOL = {"version": "all-directional-v1", "problem": 4,
    "scope": "Additional local synthetic all-directional validation; not official scores or fitted distribution",
    "random": list(range(616001, 616017)), "stress": list(range(616031, 616045)),
    "stress_assignment": [{"family": f, "sources": n} for f in FAMILIES for n in (10, 16)],
    "failure_penalty_s": 360000, "max_actions": 20000, "max_active_probes": 6,
    "selection": "Fixed baseline and combo; at most one externally qualified candidate before freezing both sets",
    "interpretation": "Supplementary robustness evidence; all records and penalties retained; no retuning on these cases"}


@dataclass(frozen=True)
class AllDirectionalScenario(Scenario):
    """Legal Q4 subclass bypassing only Scenario's mixed-type generator rule."""
    def __post_init__(self):
        if type(self.problem) is not int or self.problem != 4:
            raise ValueError("All-directional scenarios require Q4")
        if type(self.seed) is not int or not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("Explicit integer seed and nonempty case ID required")
        if not isinstance(self.sources, tuple) or not 10 <= len(self.sources) <= 16:
            raise ValueError("Q4 requires 10..16 sources")
        if any(not isinstance(s, Source) for s in self.sources):
            raise ValueError("Use validated Source instances")
        # Reuse all public Source checks rather than weakening coordinates/R.
        for s in self.sources:
            Source(**asdict(s))
            if s.orientation_deg is None:
                raise ValueError("Every source must be directional")
        if len({s.channel for s in self.sources}) != len(self.sources):
            raise ValueError("Exclusive source channels required")
        if self.error_mode not in {"uniform", "zero", "positive_extreme", "negative_extreme", "alternating_extreme"}:
            raise ValueError("Unknown observation error mode")


def build_case(seed, *, count=None, family=None):
    """Public assumed distribution, also usable with nonreserved unit-test seeds."""
    if type(seed) is not int or family is not None and family not in FAMILIES:
        raise ValueError("Invalid construction seed/family")
    rng = random.Random(seed + 4*1_000_003 + 61_600_000)
    count = rng.randint(10, 16) if count is None else count
    if type(count) is not int or not 10 <= count <= 16:
        raise ValueError("Source count must be 10..16")
    channels, phase = rng.sample(list(range(1, 21)), count), rng.uniform(0, 2*math.pi)
    sources = []
    for i, channel in enumerate(channels):
        angle, radius = rng.uniform(0, 2*math.pi), 1800*math.sqrt(rng.random())
        x, y = radius*math.cos(angle), radius*math.sin(angle)
        reception, orientation = rng.uniform(1000, 1500), rng.uniform(0, 360)
        if family is not None:
            reception = 1000.
        if family == "boundary_outward":
            angle = phase + 2*math.pi*i/count
            x, y, orientation = 1799.9*math.cos(angle), 1799.9*math.sin(angle), math.degrees(angle) % 360
        elif family == "cluster":
            x, y = 1100*math.cos(phase)+rng.uniform(-.25, .25), 1100*math.sin(phase)+rng.uniform(-.25, .25)
        elif family == "narrow_strip":
            along, across = 900+30*i, (-1)**i*.2
            x, y = along*math.cos(phase)-across*math.sin(phase), along*math.sin(phase)+across*math.cos(phase)
        sources.append(Source(channel, x, y, reception, orientation))
    error = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
             "alternating_error": "alternating_extreme", "cluster": "alternating_extreme",
             "narrow_strip": "alternating_extreme"}.get(family, "uniform")
    return AllDirectionalScenario(f"q4-all-directional-{family or 'random'}-n{count}-{seed}",
        4, seed, tuple(sources), error, "Predeclared all-directional supplementary research distribution")


def make_case(seed, stage):
    if stage not in ("random", "stress") or seed not in PROTOCOL[stage]:
        raise ValueError("Not a reserved all-directional case")
    if stage == "random":
        return build_case(seed)
    assignment = PROTOCOL["stress_assignment"][PROTOCOL["stress"].index(seed)]
    return build_case(seed, count=assignment["sources"], family=assignment["family"])


def hashes():
    result = common_hashes()
    result[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def validate_specs(specs, qualification=None):
    if (not isinstance(specs, dict) or specs.get(BASE) != BASE_SPEC or specs.get(COMBO) != COMBO_SPEC
            or len(specs) not in (2, 3)):
        raise ValueError("Need unchanged baseline and combo, plus at most one qualified candidate")
    extras = set(specs)-{BASE, COMBO}
    if extras:
        name = next(iter(extras))
        if not name.startswith("compact_") or not isinstance(qualification, dict):
            raise ValueError("Extra candidate needs explicit qualification")
        if (qualification.get("passed") is not True or qualification.get("selected") != name
                or qualification.get("spec") != specs[name]):
            raise ValueError("Qualification must pass and identify the exact single candidate spec")
    elif qualification is not None:
        raise ValueError("Unexpected qualification without an extra candidate")
    for spec in specs.values():
        if (not isinstance(spec, dict) or set(spec) != {"entrypoint", "kwargs"}
                or not isinstance(spec["entrypoint"], str) or ":" not in spec["entrypoint"]
                or not spec["entrypoint"].startswith("strategies.") or not isinstance(spec["kwargs"], dict)
                or set(spec["kwargs"]) & {"problem", "max_actions", "max_active_probes"}):
            raise ValueError("Invalid frozen policy spec or overridden common budget")


def freeze_plan(path, specs=None, qualification=None):
    specs = specs if specs is not None else {BASE: BASE_SPEC, COMBO: COMBO_SPEC}
    validate_specs(specs, qualification)
    path = Path(path)
    if path.exists():
        raise ValueError("Preserve existing freeze")
    payload = {"kind": "q4_all_directional_plan_v1", "protocol": PROTOCOL,
        "source_sha256": hashes(), "specs": specs, "qualification": qualification,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cases_generated": False}
    payload["payload_sha256"] = digest(payload)
    write_json(path, payload)
    return payload


def validate_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "payload_sha256"}
    if (plan.get("payload_sha256") != digest(payload) or plan.get("kind") != "q4_all_directional_plan_v1"
            or plan.get("protocol") != PROTOCOL or plan.get("source_sha256") != hashes()
            or plan.get("cases_generated") is not False):
        raise ValueError("Freeze identity/protocol/source changed before generation")
    validate_specs(plan["specs"], plan["qualification"])


def run_one_case(case, label, spec, expected, *, stage="constructed"):
    """Own standard harness; evaluation truth is requested after termination only."""
    if hashes() != expected or not isinstance(case, AllDirectionalScenario):
        raise ValueError("Source freeze or all-directional case type mismatch")
    sim = LocalResearchSimulator(case)
    client = ObservationOnlyClient(sim.client())
    result, errors, diagnostic = None, [], None
    began = time.perf_counter()
    try:
        module, name = spec["entrypoint"].split(":", 1)
        result = getattr(importlib.import_module(module), name)(client, problem=4,
            max_actions=20000, max_active_probes=6, **spec["kwargs"])
        errors.extend(str(e) for e in (result.error, result.exit_error) if e)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        diagnostic = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {exc}")
        sim.finish_for_evaluation()
    elapsed = time.perf_counter()-began
    evaluation, history = sim.evaluation(), sim.observation_history()
    truth = evaluation["ground_truth"]
    all_directional = bool(truth["sources"]) and all(s["orientation_deg"] is not None for s in truth["sources"])
    if not all_directional:
        errors.append("All-directional supplementary case contains an omni source")
    certified = bool(result and result.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if result and (result.cleared_count != evaluation["cleared_total"] or
                   abs(result.virtual_time_s-evaluation["virtual_time_s"]) > 1e-6):
        errors.append("Strategy/evaluation mismatch")
    if abs(sum(evaluation["time_breakdown_s"].values())-evaluation["virtual_time_s"]) > 1e-6:
        errors.append("Physical cost sum mismatch")
    if certified and not evaluation["all_cleared"]:
        errors.append("False complete certificate")
    success = bool(certified and exited and evaluation["all_cleared"] and not errors)
    bound = common_bound(truth)
    lower = bound["common_lower_bound_s"]
    row = {"case_id": case.case_id, "seed": case.seed, "stage": stage, "problem": 4, "strategy": label,
        "case_sha256": digest(truth), "successful": success, "all_cleared": evaluation["all_cleared"],
        "completion_certified": certified, "accepted_exit": exited, "source_total": evaluation["source_total"],
        "cleared_total": evaluation["cleared_total"], "virtual_time_s": evaluation["virtual_time_s"],
        "penalized_time_s": evaluation["virtual_time_s"] if success else 360000,
        "program_runtime_s": elapsed, "measurement_count": evaluation["measurement_count"],
        "failed_clear_count": evaluation["failed_clear_count"], "action_count": evaluation["action_count"],
        **evaluation["time_breakdown_s"], "errors": errors, "all_sources_directional": all_directional,
        "common_lower_bound_s": lower, "time_over_lower_bound": evaluation["virtual_time_s"]/lower,
        "penalized_time_over_lower_bound": (evaluation["virtual_time_s"] if success else 360000)/lower}
    return {"row": row, "summary": result.as_dict() if result else None, "evaluation": evaluation,
        "history": history, "spec": spec, "evaluation_phase": "after_policy_termination",
        "exception_traceback": diagnostic, "common_lower_bound": bound}


def one(seed, stage, label, spec, expected):
    if hashes() != expected:
        raise ValueError("Source changed before constructing case")
    return run_one_case(make_case(seed, stage), label, spec, expected, stage=stage)


def run_stage(path, stage, plan, workers=3):
    validate_plan(plan)  # Must precede any reserved scenario generation.
    if stage not in ("random", "stress") or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Invalid stage/workers")
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"stage": stage, "seeds": PROTOCOL[stage], "specs": plan["specs"],
        "source_sha256": plan["source_sha256"], "baseline_commit": "1e58ce9fe3f72c8043d8ac35cce8a90e102d81af",
        "scope": PROTOCOL["scope"], "failure_penalty_s": 360000,
        "selection_sha256": plan["payload_sha256"], "all_directional_plan": plan}
    write_json(directory/"manifest.json", manifest)
    write_json(directory/"freeze.json", {"manifest_sha256": digest(manifest), "git_commit": plan["git_commit"]})
    with zipfile.ZipFile(directory/"source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in plan["source_sha256"]:
            archive.write(ROOT/p, p)
    (directory/"records").mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(one, seed, stage, label, spec, plan["source_sha256"])
                for seed in PROTOCOL[stage] for label, spec in plan["specs"].items()]
        for future in as_completed(jobs):
            record = future.result()
            row = record["row"]
            with gzip.open(directory/"records"/f"{row['strategy']}-{row['seed']}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(record, stream, allow_nan=False)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("seed", "strategy", "successful", "virtual_time_s",
                "common_lower_bound_s", "time_over_lower_bound", "errors")}), flush=True)
    validate_plan(plan)
    write_json(directory/"summary.json", report_rows(sorted(rows, key=lambda r: (r["seed"], r["strategy"]))))


def audit_directional_record(record):
    if record.get("evaluation_phase") != "after_policy_termination":
        raise ValueError("Missing post-termination evaluation marker")
    truth = record["evaluation"]["ground_truth"]
    sources = tuple(Source(**s) for s in truth["sources"])
    AllDirectionalScenario(truth["case_id"], truth["problem"], truth["seed"], sources,
                           truth["error_mode"], truth.get("description", ""))
    row = record["row"]
    if (row.get("all_sources_directional") is not True or row["case_sha256"] != digest(truth)
            or row["source_total"] != len(sources) or row["seed"] != truth["seed"]):
        raise ValueError("All-directional truth/count/identity mismatch")
    return {"passed": True, "seed": row["seed"], "strategy": row["strategy"], "sources": len(sources)}


def audit_stage(path):
    directory = Path(path).resolve()
    destination = directory/"all_directional_audit.json"
    if destination.exists() or (directory/"independent_audit.json").exists():
        raise ValueError("Preserve existing audits")
    manifest = json.loads((directory/"manifest.json").read_bytes())
    plan = manifest["all_directional_plan"]
    validate_plan(plan)
    if (manifest["specs"] != plan["specs"]
            or manifest["source_sha256"] != plan["source_sha256"]
            or manifest["selection_sha256"] != plan["payload_sha256"]):
        raise ValueError("Manifest differs from frozen all-directional plan")
    if manifest["stage"] not in ("random", "stress") or manifest["seeds"] != PROTOCOL[manifest["stage"]]:
        raise ValueError("Unexpected all-directional partition")
    common_code = common_audit(directory)
    errors, items, keys = [], [], set()
    for path in sorted((directory/"records").glob("*.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            row = record["row"]
            if (row["seed"] not in manifest["seeds"] or row["strategy"] not in manifest["specs"]
                    or row["stage"] != manifest["stage"] or record["spec"] != manifest["specs"][row["strategy"]]):
                raise ValueError("Directional record differs from frozen partition/spec")
            item = audit_directional_record(record)
            key = (item["seed"], item["strategy"])
            if key in keys:
                raise ValueError("Duplicate directional case")
            keys.add(key)
            if manifest["stage"] == "stress":
                expected = PROTOCOL["stress_assignment"][PROTOCOL["stress"].index(item["seed"])]["sources"]
                if item["sources"] != expected:
                    raise ValueError("Stress source-count stratum mismatch")
            items.append(item)
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            errors.append(f"{path.name}: {exc}")
    if keys != {(s, label) for s in manifest["seeds"] for label in manifest["specs"]}:
        errors.append("Missing/extra directional pairing")
    result = {"all_passed": not common_code and not errors, "records": len(items), "errors": errors,
              "audits": items, "scope": "Source type checked from saved post-policy evaluation; common physical/prefix audit retained"}
    write_json(destination, result)
    return int(not result["all_passed"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--output", type=Path, required=True)
    f.add_argument("--specs", type=Path); f.add_argument("--qualified-selection", type=Path)
    r = sub.add_parser("run"); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--freeze", type=Path, required=True); r.add_argument("--stage", choices=("random", "stress"), required=True)
    r.add_argument("--workers", type=int, default=3)
    a = sub.add_parser("audit"); a.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_plan(args.output, json.loads(args.specs.read_bytes()) if args.specs else None,
                    json.loads(args.qualified_selection.read_bytes()) if args.qualified_selection else None)
    elif args.command == "run":
        run_stage(args.output, args.stage, json.loads(args.freeze.read_bytes()), args.workers)
    else:
        return audit_stage(args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Predeclared one-action counterfactual diagnostic; CPU only, no training.

Exactly seeds 3000101..3000116. B deterministically replays A's accepted public
prefix, forces one predeclared legal probe, and resumes the same frozen policy.
Only response wall-clock timestamps are excluded from prefix equality.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[name] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/theory_v1")]

import numpy as np
import torch
from research_rl.cpu_runtime import require_cpu
from research_rl.network import load_policy
from research_rl.controller import feature_schema
from research_rl.action_sets import controller_for
from simulation import LocalResearchSimulator, random_scenario
from audit_eval_bounds import audit_record, physical_bounds, digest

SEEDS = tuple(range(3000101, 3000117))
WEIGHT_SHA = "3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4"
DEST = ROOT / "research/autonomy/paired_action_diagnostics"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), ROOT / "research/theory_v1/audit_eval_bounds.py",
              ROOT / "research/theory_v1/certify_bounds.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def model_digest(weights):
    h = hashlib.sha256()
    for key, tensor in sorted(weights.items()):
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pin_cpu():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        handle = kernel.GetCurrentProcess()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise OSError("Cannot inspect diagnostic CPU affinity")
        selected = next(i for i in range(64) if process.value & (1 << i))
        if not kernel.SetProcessAffinityMask(handle, 1 << selected):
            raise OSError("Cannot pin this diagnostic to one CPU")
    else:
        selected = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {selected})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return selected


def public_history(history):
    """Retain every public field except independently advancing wall time."""
    result = copy.deepcopy(history)
    for row in result:
        row["response"].pop("real_timestamp_ms", None)
    return result


class ForkPolicy:
    """Wrap the actor, with access only to legal candidates/observations."""
    def __init__(self, frozen, history_reader, expected=None):
        self.frozen, self.history_reader, self.expected = frozen, history_reader, expected
        self.architecture, self.action_distribution, self.action_schema = (
            frozen.architecture, frozen.action_distribution, frozen.action_schema)
        self.control = None
        self.candidates, self.regions = None, None
        self.decisions, self.events = [], []
        self.fork = None
        self.forced_actions = 0
        self.prefix_verified = False

    def prepare_candidates(self, candidates, regions):
        self.candidates, self.regions = candidates, regions
        self.frozen.prepare_candidates(candidates, regions)

    def eligibility(self, original):
        chosen, current = self.candidates[original], self.control.client.state.position
        if chosen.kind != "probe" or chosen.point != current:
            return None
        vertices = self.regions[chosen.channel].vertices
        if not vertices:
            return None
        maximum = lambda point: max(math.hypot(x-point.x, y-point.y) for x, y in vertices)
        original_max = maximum(current)
        if original_max <= 1000:
            return None
        legal = [i for i, c in enumerate(self.candidates)
                 if c.kind == "probe" and c.channel == chosen.channel and maximum(c.point) <= 1000]
        if not legal:
            return None
        alternative = min(legal, key=lambda i: (current.distance_to(self.candidates[i].point),
            self.candidates[i].point.x, self.candidates[i].point.y, i))
        return dict(original_index=original, alternative_index=alternative,
            original_candidate=asdict(chosen), alternative_candidate=asdict(self.candidates[alternative]),
            original_max_vertex_distance_m=original_max,
            alternative_max_vertex_distance_m=maximum(self.candidates[alternative].point),
            same_channel_guaranteed_alternatives=[dict(index=i, candidate=asdict(self.candidates[i]),
                travel_distance_m=current.distance_to(self.candidates[i].point),
                max_vertex_distance_m=maximum(self.candidates[i].point)) for i in legal],
            public_region_vertices=list(vertices))

    def signature(self, features, context, original):
        c = self.control
        return digest(dict(candidates=[asdict(x) for x in self.candidates],
            features=np.asarray(features, dtype=np.float32).tolist(),
            context=np.asarray(context, dtype=np.float32).tolist(), original_index=original,
            current_position=asdict(c.client.state.position), current_channel=c.client.state.current_channel,
            virtual_time_s=c.client.state.virtual_time_s,
            regions={str(k):list(v.vertices) for k,v in sorted(self.regions.items())},
            detected=sorted(c.detected), cleared=sorted(c.cleared), probe_counts=c.probe_counts,
            focus=c.focus, scan_focus=asdict(c.scan_focus) if c.scan_focus is not None else None,
            scan_ledger=[dict(point=asdict(p), channels=sorted(channels)) for p,channels in c.scan_ledger.items()]))

    def __call__(self, features, context, teacher):
        original = self.frozen(features, context, teacher)[0]
        step = len(self.decisions)
        signature = self.signature(features, context, original)
        selected = original
        qualifies = self.eligibility(original) if self.fork is None else None
        if self.expected is None and qualifies is not None:
            prefix = public_history(self.history_reader())
            self.fork = dict(step=step, prefix=prefix, prefix_sha256=digest(prefix),
                prefix_action_count=len(prefix), prefix_virtual_time_s=self.control.client.state.virtual_time_s,
                decision_signature_sha256=signature, **qualifies)
        elif self.expected is not None and step <= self.expected["fork"]["step"]:
            expected_decision = self.expected["prefix_decisions"][step]
            if signature != expected_decision["signature_sha256"] or original != expected_decision["original_index"]:
                raise ValueError("Deterministic prefix policy/input/candidate replay differs")
            if step == self.expected["fork"]["step"]:
                target = self.expected["fork"]
                prefix = public_history(self.history_reader())
                if prefix != target["prefix"] or qualifies is None:
                    raise ValueError("Accepted public prefix differs or predeclared predicate fails")
                for key in qualifies:
                    if qualifies[key] != target[key]:
                        raise ValueError("Public eligibility/alternative differs at fork")
                self.fork = copy.deepcopy(target)
                self.prefix_verified = True
                selected = target["alternative_index"]
                self.forced_actions += 1
        self.decisions.append(dict(step=step, original_index=original, selected_index=selected,
            signature_sha256=signature, candidate=asdict(self.candidates[selected]),
            before_virtual_time_s=self.control.client.state.virtual_time_s,
            before_public_action_count=len(self.history_reader())))
        return selected

    def record(self, features, context, action, teacher, selection, cost):
        if action != self.decisions[-1]["selected_index"]:
            raise ValueError("Actual accepted action differs from selected index")
        self.events.append(dict(step=len(self.events), cost_s=float(cost),
            after_virtual_time_s=self.control.client.state.virtual_time_s,
            after_public_action_count=len(self.history_reader())))


def execute(seed, policy, expected=None):
    # The engine alone owns the scenario; actor/wrapper never receive its truth.
    simulator = LocalResearchSimulator(random_scenario(3, seed), max_real_duration_s=300)
    wrapper = ForkPolicy(policy, simulator.observation_history, expected)
    control = controller_for("v3", {"version":1,"name":"base"})(simulator.client(), wrapper,
        recorder=wrapper.record, max_decisions=256, feature_version="v3")
    wrapper.control = control
    report = control.run()
    if control.client.state.session != "exited" or control.client.pending_request is not None:
        raise ValueError("Policy did not reach accepted safe exit; no invented continuation")
    evaluation, history = simulator.evaluation(), simulator.observation_history()
    truth = evaluation["ground_truth"]  # Post-termination audit only.
    errors = [str(v) for v in (report.error, report.exit_error) if v]
    if errors:
        raise ValueError("Controller reported an execution error; refuse a fabricated replay: " + "; ".join(errors))
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model and not errors)
    row = dict(case_id=truth["case_id"], seed=seed, case_sha256=digest(truth), successful=success,
        all_cleared=evaluation["all_cleared"], completion_certified=report.completion_certified_under_model,
        accepted_exit=True, source_total=evaluation["source_total"], cleared_total=evaluation["cleared_total"],
        measurement_count=evaluation["measurement_count"], failed_clear_count=evaluation["failed_clear_count"],
        action_count=evaluation["action_count"], virtual_time_s=evaluation["virtual_time_s"],
        errors=errors, **evaluation["time_breakdown_s"])
    evidence = dict(row=row, evaluation=evaluation, summary=report.as_dict(), history=history,
        evaluation_phase="after_policy_termination")
    sources, _ = audit_record(evidence)
    if len(wrapper.decisions) != len(wrapper.events) or len(wrapper.events) != report.learning["decisions"]:
        raise ValueError("Missing selected-action events")
    initial = report.learning["initial_scan_virtual_time_s"]
    tail = row["virtual_time_s"] - wrapper.events[-1]["after_virtual_time_s"]
    if tail < -1e-6 or not math.isclose(initial + sum(e["cost_s"] for e in wrapper.events) + tail,
                                      row["virtual_time_s"], abs_tol=2e-6):
        raise ValueError("Full physical cost accounting mismatch")
    if expected is not None and (wrapper.forced_actions != 1 or not wrapper.prefix_verified):
        raise ValueError("Replay failed to execute exactly one verified intervention")
    return dict(evaluation_record=evidence, decisions=wrapper.decisions, events=wrapper.events,
        fork=wrapper.fork, forced_actions=wrapper.forced_actions, prefix_verified=wrapper.prefix_verified,
        physical_audit_pass=True, cost_accounting_pass=True, additional_completion_tail_s=tail), sources


def pair_row(seed, baseline, alternative, bound):
    a = baseline["evaluation_record"]["row"]
    lb = bound["physical_clairvoyant_lower_s"]
    row = dict(seed=seed, eligible=alternative is not None, case_sha256=a["case_sha256"],
        source_total=a["source_total"], lower_bound_s=lb, a_total_s=a["virtual_time_s"],
        a_time_over_lower_bound=a["virtual_time_s"]/lb, a_successful=a["successful"],
        a_all_cleared=a["all_cleared"], a_failed_clear_count=a["failed_clear_count"], a_accepted_exit=True)
    if alternative is None:
        return row | dict(missing_reason="no predeclared qualifying decision in the original full run")
    b, fork = alternative["evaluation_record"]["row"], baseline["fork"]
    if a["case_sha256"] != b["case_sha256"]:
        raise ValueError("Fork runs used different physical worlds")
    index, before = fork["step"], fork["prefix_virtual_time_s"]
    ae, be = baseline["events"][index], alternative["events"][index]
    if (baseline["decisions"][index]["before_virtual_time_s"] != before
            or alternative["decisions"][index]["before_virtual_time_s"] != before):
        raise ValueError("Fork starts at different charged physical time")
    da, db = baseline["decisions"][index], alternative["decisions"][index]
    response_a = baseline["evaluation_record"]["history"][da["before_public_action_count"]]["response"]
    response_b = alternative["evaluation_record"]["history"][db["before_public_action_count"]]["response"]
    immediate = be["cost_s"]-ae["cost_s"]
    future_a, future_b = a["virtual_time_s"]-ae["after_virtual_time_s"], b["virtual_time_s"]-be["after_virtual_time_s"]
    delta = b["virtual_time_s"]-a["virtual_time_s"]
    if not math.isclose(delta, immediate + future_b-future_a, abs_tol=2e-6):
        raise ValueError("Paired immediate plus future cost difference is inconsistent")
    if be["after_public_action_count"]-db["before_public_action_count"] != 1 or response_b["measure_result"] == "no_signal":
        raise ValueError("Guaranteed legal alternative did not produce one received probe")
    return row | dict(missing_reason=None, fork_step=index, prefix_action_count=fork["prefix_action_count"],
        prefix_sha256=fork["prefix_sha256"], public_prefix_verified=True, forced_actions=1,
        a_probe_result=response_a["measure_result"], b_probe_result=response_b["measure_result"],
        a_immediate_cost_s=ae["cost_s"], b_immediate_cost_s=be["cost_s"],
        b_minus_a_immediate_s=immediate, a_future_cost_s=future_a, b_future_cost_s=future_b,
        b_minus_a_future_s=future_b-future_a, b_total_s=b["virtual_time_s"], b_minus_a_total_s=delta,
        a_minus_b_saving_s=-delta, b_time_over_lower_bound=b["virtual_time_s"]/lb,
        b_successful=b["successful"], b_all_cleared=b["all_cleared"],
        b_failed_clear_count=b["failed_clear_count"], b_accepted_exit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    require_cpu()
    affinity = pin_cpu()
    if sha(args.checkpoint) != WEIGHT_SHA:
        raise ValueError("Not the predeclared original parent checkpoint")
    if DEST.exists():
        raise ValueError("Evidence already exists; never overwrite, resample or silently resume")
    policy = load_policy(args.checkpoint, device="cpu", deterministic=True)
    if (policy.feature_version != "v3" or policy.architecture["name"] != "mlp"
            or policy.action_distribution["name"] != "flat" or policy.action_schema["name"] != "base"):
        raise ValueError("Expected unchanged v3/base/flat MLP parent")
    weights, sources = model_digest(policy.model.state_dict()), source_hashes()
    manifest = dict(version=1, started_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        checkpoint_sha256=WEIGHT_SHA, tensor_sha256=weights, source_sha256=sources,
        source_manifest_sha256=digest(sources), code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            platform=platform.system(), cpu_affinity=[affinity], torch_threads=1, interop_threads=1,
            workers=0, gpu_build=False), feature_schema=feature_schema("v3"),
        protocol=dict(deterministic=True, maximum_policy_decisions=256, max_actions=20000,
            predicate="first original current-position active probe with max feasible-region vertex distance >1000m and >=1 same-channel legal base probe with max distance <=1000m",
            alternative="minimum immediate movement distance, then x,y,original candidate index",
            no_eligible="retain missing; no replacement seeds or later prefixes",
            continuation="same frozen deterministic policy and unchanged safety/fallback controller",
            replay="fresh identical-seed engine; exact accepted public prefix and per-decision signatures before forcing one action",
            prefix_ignored_fields=["response.real_timestamp_ms"],
            ignored_field_reason="wall-clock metadata necessarily advances between independent runs; no physical/action feedback fields omitted",
            noise="unchanged engine hashes scenario seed/channel/exact coordinates; same position/channel has same realized bearing error in both branches",
            truth_boundary="engine owns scenario; selection uses only accepted observations/candidates/region; evaluation and LB only after termination",
            scope="fixed training cases, empirical one-action intervention under one frozen continuation; not globally optimal or a deployment oracle",
            lower_bound="post-termination L/5+5N, subset DP on distances between 20m clearance disks",
            no_training=True, no_counterfactual_selection_by_future_outcome=True))
    DEST.mkdir(parents=True)
    atomic_json(DEST / "manifest.json", manifest)
    rows, hashes = [], {}
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            if source_hashes() != sources:
                raise ValueError("Source changed; do not mix versions")
            baseline, truths = execute(seed, policy)
            fork = baseline["fork"]
            expected = (dict(fork=copy.deepcopy(fork), prefix_decisions=copy.deepcopy(
                baseline["decisions"][:fork["step"]+1])) if fork is not None else None)
            alternative = execute(seed, policy, expected)[0] if expected is not None else None
            if model_digest(policy.model.state_dict()) != weights:
                raise ValueError("Frozen policy weights changed")
            bound = physical_bounds(truths.values())
            row = pair_row(seed, baseline, alternative, bound)
            path = DEST / f"case-{seed}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(dict(row=row, bound=bound, a=baseline, b=alternative,
                    source_manifest_sha256=manifest["source_manifest_sha256"], checkpoint_sha256=WEIGHT_SHA),
                    stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            hashes[path.name] = sha(path)
            rows.append(row)
            atomic_json(DEST / "progress.json", dict(completed_cases=len(rows), expected_cases=16, rows=rows))
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if source_hashes() != sources or sha(args.checkpoint) != WEIGHT_SHA:
            raise ValueError("Source/checkpoint identity changed")
    except Exception as exc:
        atomic_json(DEST / "incomplete.json", dict(complete=False, completed_cases=len(rows),
            expected_cases=16, exception_type=type(exc).__name__, reason=str(exc), record_sha256=hashes,
            no_resampling=True))
        raise
    pairs = [r for r in rows if r["eligible"]]
    summary = dict(complete=True, expected_cases=16, completed_cases=len(rows), eligible_cases=len(pairs),
        missing_cases=[r["seed"] for r in rows if not r["eligible"]],
        all_baseline_successful=all(r["a_successful"] for r in rows),
        all_alternative_successful=all(r["b_successful"] for r in pairs),
        baseline_failed_clears=sum(r["a_failed_clear_count"] for r in rows),
        alternative_failed_clears=sum(r["b_failed_clear_count"] for r in pairs),
        b_faster_count=sum(r["b_minus_a_total_s"] < -1e-6 for r in pairs),
        b_slower_count=sum(r["b_minus_a_total_s"] > 1e-6 for r in pairs),
        tied_count=sum(abs(r["b_minus_a_total_s"]) <= 1e-6 for r in pairs),
        verified_identical_prefixes=sum(r["public_prefix_verified"] for r in pairs),
        paired_mean_savings_s=statistics.mean(r["a_minus_b_saving_s"] for r in pairs) if pairs else None,
        paired_mean_immediate_difference_s=statistics.mean(r["b_minus_a_immediate_s"] for r in pairs) if pairs else None,
        paired_mean_future_difference_s=statistics.mean(r["b_minus_a_future_s"] for r in pairs) if pairs else None,
        paired_a_ratio_of_sums=sum(r["a_total_s"] for r in pairs)/sum(r["lower_bound_s"] for r in pairs) if pairs else None,
        paired_b_ratio_of_sums=sum(r["b_total_s"] for r in pairs)/sum(r["lower_bound_s"] for r in pairs) if pairs else None,
        runtime_s=time.perf_counter()-started, source_and_weights_unchanged=True, record_sha256=hashes,
        scope="single first eligible intervention per fixed training case, deterministic frozen continuation; descriptive causal within this simulator/world, not optimal action value or general improvement")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (DEST / "per_case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(DEST / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

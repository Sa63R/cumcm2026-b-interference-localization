"""Frozen-parent credit diagnostic on exactly 16 declared training cases.

CPU only, no optimizer, network, official simulator, database, or held-out split.
Run: python research/autonomy/credit_diagnostic.py --checkpoint <parent.pt>
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
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
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"] = os.environ["ROCR_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research/theory_v1")]

import numpy as np
import torch
from research_rl import train
from research_rl.cpu_runtime import require_cpu
from research_rl.controller import ALGORITHM_VERSIONS, feature_schema
from research_rl.network import checkpoint_architecture
from research_rl.distributions import checkpoint_distribution
from research_rl.action_sets import checkpoint_action_schema
from audit_eval_bounds import audit_record, physical_bounds, digest

SEEDS = tuple(range(3000001, 3000017))
WEIGHT_SHA = "3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4"
DEST = ROOT / "research/autonomy/credit_diagnostics"
KINDS = ("cover", "probe", "clear", "fallback")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pin_cpu():
    """Constrain this process, not any collector/training process, to <=2 CPUs."""
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        handle = kernel.GetCurrentProcess()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise OSError("Cannot inspect this diagnostic process CPU affinity")
        available = [i for i in range(64) if process.value & (1 << i)]
        selected = available[:2]
        if not selected or not kernel.SetProcessAffinityMask(handle, sum(1 << i for i in selected)):
            raise OSError("Cannot apply diagnostic CPU affinity cap")
    else:
        selected = sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0, selected)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return selected


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), ROOT / "research/autonomy/protocol.json",
              ROOT / "research/theory_v1/audit_eval_bounds.py", ROOT / "research/theory_v1/certify_bounds.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def model_digest(weights):
    value = hashlib.sha256()
    for key, tensor in sorted(weights.items()):
        value.update(key.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def captured_episode(task):
    """Call unchanged train.episode while retaining its objects for later audit.

    Factories only retain references and observe the existing recorder. They do
    not replace the policy, transitions, costs, value estimates, or RNG logic.
    """
    original_sim, original_controller = train.LocalResearchSimulator, train.controller_for
    capture, events = {}, []

    def simulator(*args, **kwargs):
        capture["simulator"] = original_sim(*args, **kwargs)
        return capture["simulator"]

    def controller(version, schema):
        cls = original_controller(version, schema)
        def construct(*args, **kwargs):
            recorder = kwargs["recorder"]
            def observe(features, context, action, target, selection, cost):
                recorder(features, context, action, target, selection, cost)
                report = capture["controller"].report
                events.append(dict(instantaneous_cost_s=float(cost),
                    physical_history_start=events[-1]["physical_history_stop"] if events else 0,
                    physical_history_stop=len(report.action_history),
                    virtual_time_after_s=capture["controller"].client.state.virtual_time_s))
            kwargs["recorder"] = observe
            capture["controller"] = cls(*args, **kwargs)
            return capture["controller"]
        return construct

    train.LocalResearchSimulator, train.controller_for = simulator, controller
    try:
        records, metrics = train.episode(task)
    finally:
        train.LocalResearchSimulator, train.controller_for = original_sim, original_controller
    control, simulator = capture["controller"], capture["simulator"]
    report, client = control.report, control.client
    if client.state.session != "exited" or client.pending_request is not None:
        raise ValueError("No accepted complete exit; terminal GAE cannot be fabricated")
    evaluation = simulator.evaluation()  # Only after the unchanged helper has ended.
    history = simulator.observation_history()
    errors = [str(v) for v in (report.error, report.exit_error) if v]
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model and not errors)
    truth = evaluation["ground_truth"]
    row = dict(case_id=truth["case_id"], seed=truth["seed"], case_sha256=digest(truth),
        successful=success, all_cleared=evaluation["all_cleared"],
        completion_certified=report.completion_certified_under_model, accepted_exit=True,
        source_total=evaluation["source_total"], cleared_total=evaluation["cleared_total"],
        measurement_count=evaluation["measurement_count"], failed_clear_count=evaluation["failed_clear_count"],
        action_count=evaluation["action_count"], virtual_time_s=evaluation["virtual_time_s"],
        errors=errors, **evaluation["time_breakdown_s"])
    evidence = dict(row=row, evaluation=evaluation, summary=report.as_dict(), history=history,
                    evaluation_phase="after_policy_termination")
    sources, _ = audit_record(evidence)
    if not records or len(records) != len(events) or len(records) != report.learning["decisions"]:
        raise ValueError("Missing or incomplete policy transition records")
    if errors or report.completion_reason not in ("coverage_exhausted_and_all_detected_cleared", "source_count_upper_bound_reached"):
        raise ValueError("Non-normal completion; this diagnostic refuses a fabricated terminal target")
    initial = metrics["initial_scan_virtual_time_s"]
    # The controller's fallback metric also includes selected source-fallback
    # macros, whose costs are already in recorder events. Only time after the
    # last recorded policy action is an additional completion tail.
    tail = max(0., row["virtual_time_s"] - events[-1]["virtual_time_after_s"])
    metrics["diagnostic_completion_tail_s"] = tail
    if not math.isclose(initial + sum(e["instantaneous_cost_s"] for e in events) + tail, row["virtual_time_s"], abs_tol=2e-6):
        raise ValueError("Selected-action, initial and completion-tail cost ledger mismatch")
    if not math.isclose(metrics["reward_cost_s"] + initial, row["virtual_time_s"] + metrics["failure_penalty_s"], abs_tol=2e-6):
        raise ValueError("Full episode reward cost mismatch")
    for i, (record, event) in enumerate(zip(records, events)):
        expected = event["instantaneous_cost_s"]
        if i == len(records)-1:
            expected += tail + metrics["failure_penalty_s"]
        if not math.isclose(-1000*record["reward"], expected, abs_tol=2e-6):
            raise ValueError("Per-decision reward and physical action cost disagree")
    return records, metrics, events, evidence, physical_bounds(sources.values())


def distance_group(distance):
    return next(label for maximum, label in ((9, "00-09"), (29, "10-29"), (59, "30-59"),
        (119, "60-119"), (math.inf, "120+")) if distance <= maximum)


def credit_rows(records, metrics, events, seed):
    gae95 = train.compute_returns([dict(r) for r in records], gae_lambda=.95)
    gae1 = train.compute_returns([dict(r) for r in records], gae_lambda=1.0)
    costs = [-r["reward"] * 1000 for r in records]
    costs[-1] -= metrics["failure_penalty_s"]
    remaining = np.cumsum(np.asarray(costs[::-1], dtype=np.float64))[::-1]
    result = []
    for i, (record, first, second, event) in enumerate(zip(records, gae95, gae1, events)):
        chosen = record["features"][record["action"]]
        flags = chosen[:4].tolist()
        if flags.count(1.) != 1 or sum(flags) != 1.:
            raise ValueError("Unknown selected action kind in v3 feature prefix")
        kind = KINDS[flags.index(1.)]
        a95, a1, value_cost = first["advantage"] * 1000, second["advantage"] * 1000, -record["value"] * 1000
        mc_penalized = float(remaining[i]) + metrics["failure_penalty_s"]
        if not math.isclose(a1, value_cost - mc_penalized, abs_tol=1e-7):
            raise ValueError("lambda=1 does not telescope to exact reward-to-go minus V")
        distance = len(records) - 1 - i
        result.append(dict(seed=seed, step=i, kind=kind, distance_to_last_policy_step=distance,
            distance_group=distance_group(distance), action=int(record["action"]),
            candidate_count=len(record["features"]), value_cost_prediction_s=value_cost,
            mc_remaining_real_cost_s=float(remaining[i]), mc_remaining_penalized_cost_s=mc_penalized,
            value_error_real_s=value_cost - float(remaining[i]),
            value_error_penalized_s=value_cost - mc_penalized,
            gae95_advantage_s=a95, gae1_advantage_s=a1,
            sign_disagreement=bool(np.sign(a95) != np.sign(a1)),
            opposite_nonzero_sign=bool(a95 * a1 < 0),
            charged_reward_cost_s=-record["reward"] * 1000,
            fixed_tail_charged_here_s=metrics["diagnostic_completion_tail_s"] if distance == 0 else 0.,
            **event))
    return result


def stats(rows):
    if not rows:
        return dict(transitions=0)
    first, second = ([row[key] for row in rows] for key in ("gae95_advantage_s", "gae1_advantage_s"))
    errors = [r["value_error_real_s"] for r in rows]
    return dict(transitions=len(rows), episodes=len({r["seed"] for r in rows}),
        sign_disagreements=sum(r["sign_disagreement"] for r in rows),
        sign_disagreement_fraction=statistics.mean(r["sign_disagreement"] for r in rows),
        opposite_nonzero_signs=sum(r["opposite_nonzero_sign"] for r in rows),
        gae95_positive=sum(x > 0 for x in first), gae1_positive=sum(x > 0 for x in second),
        gae95_mean_s=statistics.mean(first), gae1_mean_s=statistics.mean(second),
        gae95_variance_s2=statistics.pvariance(first), gae1_variance_s2=statistics.pvariance(second),
        value_error_bias_s=statistics.mean(errors), value_error_mae_s=statistics.mean(abs(x) for x in errors),
        value_error_rmse_s=math.sqrt(statistics.mean(x*x for x in errors)))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    require_cpu()
    affinity = pin_cpu()
    if sha(args.checkpoint) != WEIGHT_SHA:
        raise ValueError("Checkpoint does not match the predeclared original parent SHA")
    if DEST.exists():
        raise ValueError("Evidence destination exists; never overwrite or silently resample")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture, distribution, schema = checkpoint_architecture(payload), checkpoint_distribution(payload), checkpoint_action_schema(payload)
    if (payload["algorithm"] != ALGORITHM_VERSIONS["v3"] or payload["feature_schema"] != feature_schema("v3")
            or architecture["name"] != "mlp" or distribution["name"] != "flat" or schema["name"] != "base"):
        raise ValueError("Expected the original v3/base/flat MLP")
    before = model_digest(payload["model"])
    sources = source_hashes()
    metadata = dict(version=1, scope="fixed 16-case training-only factual credit diagnostic; no optimization or counterfactual claim",
        seeds=list(SEEDS), action_seeds=[9114100+i for i in range(16)], action_seed_indexing="zero-based fixed seed order",
        checkpoint_sha256=WEIGHT_SHA, checkpoint_model_tensor_sha256=before,
        checkpoint_public_metadata=dict(algorithm=payload["algorithm"], hidden=payload["hidden"],
            architecture=architecture, distribution=distribution, action_schema=schema,
            feature_schema=payload["feature_schema"], torch_version=payload.get("torch_version", "not recorded"),
            original_source_manifest_sha256=payload.get("source_manifest", {}).get("sha256")),
        source_sha256=sources, source_manifest_sha256=digest(sources),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        episode_helper_sha256=hashlib.sha256(inspect.getsource(train.episode).encode()).hexdigest(),
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            platform=platform.system(), machine=platform.machine(), cpu_affinity=affinity,
            torch_threads=1, interop_threads=1, workers=0, gpu_build=False),
        method=dict(gamma=1, lambdas=[.95, 1.0], normalized=False, max_decisions=256,
            helper_action_limit=20000, real_seconds_per_case=300, stochastic_actions=True,
            reward_scale=1000, failure_penalty_s=360000,
            terminal="only normally completed episodes with accepted exit; V_terminal=0",
            capture="unchanged train.episode; factories retain objects and recorder timing only",
            value_error="-1000*V minus complete remaining physical cost; positive means cost overprediction",
            distance="number of subsequent selected policy decisions, not seconds; fixed tail charged at final decision",
            lower_bound="L/5+5N; exact clearance-disk graph subset DP after termination",
            group_weighting="transition-weighted descriptive statistics; transitions are correlated; no causal or IID inference",
            source_limitation="Original checkpoint is evaluated with current integrated source/runtime; no remote bitwise reproduction claim",
            anonymity="no checkpoint args, local input paths, user/account/host identifiers or credentials copied"),
        started_utc=datetime.now(timezone.utc).isoformat())
    DEST.mkdir(parents=True)
    atomic_json(DEST / "manifest.json", metadata)
    all_rows, case_rows, record_hashes = [], [], {}
    started = time.perf_counter()
    for index, seed in enumerate(SEEDS):
        if source_hashes() != sources:
            raise ValueError("Current source changed during diagnostic; do not mix versions")
        task = (seed, payload["model"], payload["hidden"], 9114100+index, False, 256, None,
                "v3", architecture, distribution, schema)
        records, metrics, events, evidence, bound = captured_episode(task)
        if model_digest(train._worker_model.state_dict()) != before:
            raise ValueError("Frozen model changed during sampling")
        rows = credit_rows(records, metrics, events, seed)
        all_rows.extend(rows)
        lb = bound["physical_clairvoyant_lower_s"]
        case_row = dict(seed=seed, action_seed=9114100+index, case_sha256=evidence["row"]["case_sha256"],
            successful=evidence["row"]["successful"], all_cleared=evidence["row"]["all_cleared"],
            failed_clear_count=evidence["row"]["failed_clear_count"],
            source_total=evidence["row"]["source_total"], accepted_exit=True, physical_audit_pass=True,
            full_reward_accounting_pass=True, gae1_telescoping_pass=True, transitions=len(records),
            completion_reason=metrics["completion_reason"], virtual_time_s=metrics["virtual_time_s"],
            lower_bound_s=lb, time_over_lower_bound=metrics["virtual_time_s"]/lb,
            fixed_tail_s=metrics["diagnostic_completion_tail_s"],
            all_reported_fallback_s=metrics["learning"]["fallback_virtual_time_s"],
            **{k:v for k,v in stats(rows).items() if k not in ("transitions", "episodes")})
        case_rows.append(case_row)
        serializable = [{**r, "features":r["features"].tolist(), "context":r["context"].tolist()} for r in records]
        artifact = dict(case=case_row, bound=bound, evaluation_record=evidence,
            trajectory=serializable, credit_rows=rows, sampling_metrics=metrics,
            source_manifest_sha256=metadata["source_manifest_sha256"], checkpoint_sha256=WEIGHT_SHA)
        path = DEST / f"case-{seed}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(artifact, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        record_hashes[path.name] = sha(path)
        atomic_json(DEST / "progress.json", dict(completed_cases=len(case_rows), expected_cases=16, cases=case_rows))
        print(json.dumps({k:case_row[k] for k in ("seed", "successful", "transitions", "virtual_time_s", "lower_bound_s", "time_over_lower_bound", "sign_disagreement_fraction")}), flush=True)
    if source_hashes() != sources or sha(args.checkpoint) != WEIGHT_SHA or model_digest(payload["model"]) != before:
        raise ValueError("Source/checkpoint identity changed")
    summary = dict(complete=True, expected_cases=16, completed_cases=len(case_rows),
        successful_cases=sum(r["successful"] for r in case_rows),
        failed_clear_count=sum(r["failed_clear_count"] for r in case_rows),
        physical_audit_passes=sum(r["physical_audit_pass"] for r in case_rows),
        full_reward_accounting_passes=sum(r["full_reward_accounting_pass"] for r in case_rows),
        gae1_telescoping_passes=sum(r["gae1_telescoping_pass"] for r in case_rows),
        source_and_checkpoint_unchanged=True, total_time_s=sum(r["virtual_time_s"] for r in case_rows),
        mean_time_s=statistics.mean(r["virtual_time_s"] for r in case_rows),
        total_lower_bound_s=sum(r["lower_bound_s"] for r in case_rows),
        mean_lower_bound_s=statistics.mean(r["lower_bound_s"] for r in case_rows),
        aggregate_time_over_lower_bound=sum(r["virtual_time_s"] for r in case_rows)/sum(r["lower_bound_s"] for r in case_rows),
        pooled=stats(all_rows), by_action={kind:stats([r for r in all_rows if r["kind"]==kind]) for kind in KINDS},
        by_distance={group:stats([r for r in all_rows if r["distance_group"]==group]) for group in ("00-09", "10-29", "30-59", "60-119", "120+")},
        by_action_and_distance={kind:{group:stats([r for r in all_rows if r["kind"]==kind and r["distance_group"]==group])
            for group in ("00-09", "10-29", "30-59", "60-119", "120+")} for kind in KINDS},
        record_sha256=record_hashes, runtime_s=time.perf_counter()-started,
        scope="Stochastic original parent on fixed training benchmark cases; factual credit diagnostic, no improvement or action causal-value claim",
        note="A1 uses the same single realized continuation, not the unobserved counterfactual return. Groups and steps are correlated.")
    write_csv(DEST / "per_case.csv", case_rows)
    write_csv(DEST / "per_transition.csv", all_rows)
    atomic_json(DEST / "summary.json", summary)
    print(json.dumps({key:summary[key] for key in ("completed_cases", "successful_cases", "mean_time_s", "mean_lower_bound_s", "aggregate_time_over_lower_bound", "pooled")}), flush=True)


if __name__ == "__main__":
    main()

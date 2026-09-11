"""Transactional CPU SCST with two real synthetic rollouts per reserved scene.

Only the sampled trajectory contributes gradients. A complete batch receives
one optimizer step, averaging trajectory sums without length normalization.
Each returned leg is written once; a small append-only index retains attempts.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import signal
import statistics
import time

import torch
from torch.distributions import Categorical

from .scst_network import (ALGORITHM, ARCHITECTURE, CHECKPOINT_VERSION, CONTROLLER_ENTRYPOINT,
    OBJECTIVE, RolloutPolicy, configure_cpu, feature_schema, initialize_micro_warmstart,
    model_from_metadata, pack_observations, validate_checkpoint)
from .train import (TRAIN_START, TRAIN_END, DEFAULT_DEADLINE, TrainingStop, attach_returns,
    training_case_spec, parse_deadline, summarize_training_metrics, _write_batch, _write_json)

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite scalar accounting")
    return float(value)


def penalized_cost(metrics):
    actual = _finite(metrics["actual_time_s"])
    if actual < 0 or type(metrics.get("success")) is not bool:
        raise ValueError("Invalid actual time/success")
    expected = actual if metrics["success"] else max(actual, OBJECTIVE["failure_penalty_s"])
    if abs(_finite(metrics["penalized_time_s"]) - expected) > 2e-5:
        raise ValueError("SCST failure accounting mismatch")
    return expected


def pair_advantage(sample, greedy):
    digest = sample.get("policy_checkpoint_sha256")
    if (sample.get("leg") != "sample" or greedy.get("leg") != "greedy" or
            sample["seed"] != greedy["seed"] or
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest) or digest != greedy.get("policy_checkpoint_sha256")):
        raise ValueError("SCST legs must share one scene and frozen policy")
    if sample.get("administrative_skip") or greedy.get("administrative_skip"):
        raise TrainingStop("Incomplete administrative pair must be replayed")
    return (penalized_cost(greedy["metrics"]) - penalized_cost(sample["metrics"])) / OBJECTIVE["cost_unit_s"]


def verify_billing(actual, learning):
    """The controller keeps billed entry/terminal tails outside the two subtotals."""
    accounted = sum(_finite(learning[key]) for key in
                    ("decision_cost_s", "fallback_cost_s", "uncovered_cost_s"))
    if (abs(accounted - _finite(actual)) > 2e-5 or
            abs(_finite(learning["total_billed_cost_s"]) - actual) > 2e-5):
        raise ValueError("Full decision/fallback/entry-terminal cost does not conserve billing")


def scst_update(model, optimizer, pairs, *, minibatch_size=128, max_grad_norm=.5, stop_check=None):
    """One on-policy batch gradient; no greedy features, value loss or entropy."""
    if not pairs or minibatch_size < 1 or max_grad_norm <= 0:
        raise ValueError("Nonempty pairs and positive update parameters required")
    advantages = [pair_advantage(pair["sample"], pair["greedy"]) for pair in pairs]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    objective, steps = 0., 0
    for pair, advantage in zip(pairs, advantages):
        records = pair["sample"]["records"]
        attach_returns(records, actual_time_s=pair["sample"]["metrics"]["actual_time_s"],
                       success=pair["sample"]["metrics"]["success"])
        for offset in range(0, len(records), minibatch_size):
            if stop_check and stop_check():
                optimizer.zero_grad(set_to_none=True)
                raise TrainingStop("SCST interrupted before its sole optimizer step")
            rows = records[offset:offset + minibatch_size]
            logits, _ = model(*pack_observations(rows))
            actions = torch.tensor([row["action_index"] for row in rows], dtype=torch.long)
            log_prob = Categorical(logits=logits).log_prob(actions)
            logged = torch.tensor([row["log_prob"] for row in rows], dtype=torch.float32)
            if not torch.allclose(log_prob.detach(), logged, rtol=1e-5, atol=5e-5):
                raise ValueError("Saved sample is not on the frozen current policy")
            loss = -(advantage / len(pairs)) * log_prob.sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite SCST loss")
            loss.backward()
            objective += float(loss.detach())
            steps += len(rows)
    if stop_check and stop_check():
        optimizer.zero_grad(set_to_none=True)
        raise TrainingStop("SCST interrupted before its sole optimizer step")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
    if not math.isfinite(gradient_norm):
        raise ValueError("Nonfinite SCST gradient")
    optimizer.step()
    return {"updates": 1, "trajectory_pairs": len(pairs), "sampled_steps": steps,
            "loss": objective, "gradient_norm": gradient_norm,
            "mean_advantage": statistics.mean(advantages),
            "advantage_std": statistics.pstdev(advantages), "entropy_coefficient": 0.,
            "reduction": OBJECTIVE["reduction"]}


def rollout_leg(task):
    started, cpu_started = time.perf_counter(), time.process_time()
    configure_cpu()
    from simulation import LocalResearchSimulator
    from .micro_controller import run_q4_micro
    from .scenarios import build_case
    seed, leg = task["seed"], task["leg"]
    if leg not in ("sample", "greedy"):
        raise ValueError("Unknown SCST leg")
    identity = {"seed": seed, "leg": leg, "policy_checkpoint_sha256": task["policy_checkpoint_sha256"]}
    if time.time() >= task["deadline_epoch"]:
        return {**identity, "administrative_skip": "deadline_before_start", "records": []}
    spec = training_case_spec(seed)
    model = model_from_metadata(task["network"])
    model.load_state_dict(task["model"])
    model.eval()
    random.seed(task["action_seed"])
    torch.manual_seed(task["action_seed"])
    policy = RolloutPolicy(model, deterministic=leg == "greedy")
    simulator = LocalResearchSimulator(build_case(seed, split="train", **spec),
        max_real_duration_s=min(300., max(1., task["deadline_epoch"]-time.time())))
    policy_started, policy_cpu_started = time.perf_counter(), time.process_time()
    report = run_q4_micro(simulator.client(), policy=policy, max_decisions=512,
        record_transitions=leg == "sample", action_deadline_epoch=task["deadline_epoch"])
    evaluation = simulator.evaluation()
    learning = report.learning
    if learning["feature_schema"] != feature_schema():
        raise ValueError("SCST public feature semantics mismatch")
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model
                   and not report.error and not report.exit_error)
    actual = float(report.virtual_time_s)
    records = [dict(row) for row in learning["transitions"]]
    if leg == "sample":
        if len(records) != len(policy.records):
            raise ValueError("Sample controller/policy trace length mismatch")
        for record, sampled in zip(records, policy.records):
            if any(record[key] != sampled[key] for key in ("global_features", "candidate_features", "action_index")):
                raise ValueError("Sample policy/controller observation mismatch")
            record["log_prob"] = sampled["log_prob"]
        accounting = attach_returns(records, actual_time_s=actual, success=success)
    else:
        if records or policy.records:
            raise ValueError("Greedy leg must not retain training feature tensors")
        expected = actual if success else max(actual, OBJECTIVE["failure_penalty_s"])
        accounting = {"actual_time_s": actual, "penalized_time_s": expected,
                      "failure_penalty_adjustment_s": expected-actual, "reward_cost_s": expected}
    verify_billing(actual, learning)
    policy_wall, policy_cpu = time.perf_counter()-policy_started, time.process_time()-policy_cpu_started
    # Pair reward accounting is fixed above. Truth and lower bounds are log-only.
    from experiments.q4_comparison_bounds import common_bound
    bound_start, bound_cpu_start = time.perf_counter(), time.process_time()
    bound = common_bound(evaluation["ground_truth"])
    bound_wall, bound_cpu = time.perf_counter()-bound_start, time.process_time()-bound_cpu_start
    lower = float(bound["common_lower_bound_s"])
    metrics = {"seed": seed, "split": "train", **spec, "success": success, **accounting,
        "failed_clear_count": int(evaluation["failed_clear_count"]),
        "fallback_cost_s": float(learning["fallback_cost_s"]), "decisions": learning["decisions"],
        "wall_time_s": time.perf_counter()-started, "worker_cpu_s": time.process_time()-cpu_started,
        "policy_wall_s": policy_wall, "policy_cpu_s": policy_cpu,
        "posthoc_bound_wall_s": bound_wall, "posthoc_bound_cpu_s": bound_cpu,
        "common_lower_bound_s": lower, "actual_time_over_lower_bound": actual/lower if success else None,
        "penalized_time_over_lower_bound": accounting["penalized_time_s"]/lower,
        "completion_reason": report.completion_reason, "error": report.error, "exit_error": report.exit_error}
    result = {**identity, "records": records, "metrics": metrics,
        "algorithm": ALGORITHM, "controller_entrypoint": CONTROLLER_ENTRYPOINT,
        "evaluation_phase": "after_policy_termination", "action_history": report.action_history,
        "observation_history": simulator.observation_history(), "evaluation": evaluation,
        "controller_learning": {k: v for k, v in learning.items() if k != "transitions"},
        "strategy_parameters": report.strategy_parameters, "common_bound": bound}
    if report.completion_reason == "training_deadline" or (not success and time.time() >= task["deadline_epoch"]):
        result["administrative_skip"] = "training_deadline_during_episode"
    return result


def save_checkpoint(path, model, optimizer, state, config, initialization):
    if model.metadata().get("architecture") != ARCHITECTURE or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Cannot save a different model/device as SCST")
    payload = {"version": CHECKPOINT_VERSION, "algorithm": ALGORITHM, "device": "cpu",
        "controller_entrypoint": CONTROLLER_ENTRYPOINT, "feature_schema": feature_schema(),
        "network": model.metadata(), "objective": dict(OBJECTIVE), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "state": state, "config": config,
        "initialization": initialization, "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path):
    configure_cpu()
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=saved["config"]["learning_rate"])
    optimizer.load_state_dict(saved["optimizer"])
    random.setstate(saved["rng"]["python"])
    torch.set_rng_state(saved["rng"]["torch"])
    return model, optimizer, saved["state"], saved["config"], saved["initialization"]


def write_leg(path, row, index_path):
    """One large serialization per completed leg; append only small provenance."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing raw leg")
    _write_batch(path, row)
    raw = path.read_bytes()
    entry = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
             "seed": row["seed"], "leg": row["leg"], "administrative_skip": row.get("administrative_skip")}
    with Path(index_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return entry


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-micro-warmstart", type=Path)
    parser.add_argument("--initialize-sha256")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-budget", type=int, default=2)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--entropy-coefficient", type=float, default=0.)
    parser.add_argument("--random-seed", type=int, default=424343)
    parser.add_argument("--scenario-start", type=int, default=8006000)
    parser.add_argument("--scenario-end", type=int, default=TRAIN_END)
    parser.add_argument("--max-batches", type=int, default=100000)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.)
    parser.add_argument("--deadline", default=DEFAULT_DEADLINE)
    args = parser.parse_args(argv)
    cpu_valid = (args.workers == args.cpu_budget == 1 or 1 <= args.workers < args.cpu_budget <= 60)
    if (not cpu_valid or args.max_decisions != 512 or
            args.entropy_coefficient != 0 or min(args.batch_pairs, args.minibatch_size, args.max_batches) < 1 or
            not TRAIN_START <= args.scenario_start <= args.scenario_end <= TRAIN_END or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
        parser.error("Invalid CPU, fixed SCST objective, scene partition or training budget")
    if args.resume and (args.initialize_micro_warmstart or args.initialize_sha256):
        parser.error("Resume and initialization are distinct operations")
    if not args.resume and (not args.initialize_micro_warmstart or not args.initialize_sha256):
        parser.error("New SCST training needs an explicit frozen micro warmstart path and SHA-256")
    return parser, args


def main(argv=None):
    global _stop_requested
    _stop_requested = False
    parser, args = _arguments(argv)
    configure_cpu()
    deadline = min(parse_deadline(args.deadline), time.time()+args.max_wall_seconds)
    if deadline <= time.time():
        parser.error("Training deadline has passed")
    config = {key: getattr(args, key) for key in ("workers", "cpu_budget", "batch_pairs", "minibatch_size",
        "learning_rate", "max_decisions", "entropy_coefficient", "random_seed", "scenario_start", "scenario_end")}
    if args.resume:
        if args.resume.resolve() != (args.output/"latest.pt").resolve():
            parser.error("Resume requires the latest transaction in the same output directory")
        model, optimizer, state, previous, initialization = restore_checkpoint(args.resume)
        if previous != config:
            parser.error("Resume cannot silently change configuration")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("New SCST output must be empty")
        model, initialization = initialize_micro_warmstart(args.initialize_micro_warmstart, args.initialize_sha256)
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        state = {"next_seed": args.scenario_start, "pairs": 0, "rollouts": 0, "batches": 0,
            "attempted_pairs": 0, "attempted_rollouts": 0, "pending_batch": None, "next_attempt": 0,
            "wall_time_s": 0., "stop_reason": None}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"raw").mkdir(exist_ok=True)
    if not args.resume:
        save_checkpoint(args.output/"initial.pt", model, optimizer, state, config, initialization)
    save_checkpoint(args.output/"latest.pt", model, optimizer, state, config, initialization)
    started, prior_wall = time.perf_counter(), state["wall_time_s"]
    previous_signal = signal.signal(signal.SIGTERM, _request_stop)
    executor = None
    try:
        if args.workers > 1:
            executor = ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=configure_cpu)
        while state["batches"] < args.max_batches and time.time() < deadline-5 and not _stop_requested:
            batch_parent_cpu_start = time.process_time()
            if state["pending_batch"] is None:
                if state["next_seed"] > args.scenario_end:
                    state["stop_reason"] = "training_partition_exhausted"
                    break
                count = min(args.batch_pairs, args.scenario_end-state["next_seed"]+1)
                seeds = list(range(state["next_seed"], state["next_seed"]+count))
                state["next_seed"] += count
                state["attempted_pairs"] += count
                state["pending_batch"] = {"seeds": seeds, "action_seeds":
                    [random.randrange(2**31) for _ in seeds]}
            pending = state["pending_batch"]
            attempt = state["next_attempt"]
            state["next_attempt"] += 1
            state["attempted_rollouts"] += 2*len(pending["seeds"])
            state["wall_time_s"] = prior_wall+time.perf_counter()-started
            save_checkpoint(args.output/"latest.pt", model, optimizer, state, config, initialization)
            stem = f"batch-{state['batches']:06d}-attempt-{attempt:06d}"
            frozen_policy_path = args.output/(stem+"-policy.pt")
            if frozen_policy_path.exists():
                raise FileExistsError("Frozen sampling policy already exists")
            save_checkpoint(frozen_policy_path, model, optimizer, state, config, initialization)
            policy_hash = hashlib.sha256(frozen_policy_path.read_bytes()).hexdigest()
            index_path = args.output/"raw"/(stem+".jsonl")
            if index_path.exists():
                raise FileExistsError("Attempt index already exists")
            tasks = [dict(seed=seed, leg=leg, action_seed=action_seed, network=model.metadata(),
                model=model.state_dict(), policy_checkpoint_sha256=policy_hash, deadline_epoch=deadline)
                for seed, action_seed in zip(pending["seeds"], pending["action_seeds"])
                for leg in ("sample", "greedy")]
            _write_json(args.output/"raw"/(stem+"-manifest.json"), {
                "algorithm": ALGORITHM, "policy_checkpoint": frozen_policy_path.name,
                "policy_checkpoint_sha256": policy_hash, "reservation": pending,
                "legs": [{"seed": task["seed"], "leg": task["leg"], "file": f"{stem}-leg-{i:04d}.json.gz"}
                         for i, task in enumerate(tasks)]})
            legs, entries = [], []
            results = executor.map(rollout_leg, tasks) if executor else map(rollout_leg, tasks)
            for i, leg in enumerate(results):
                entry = write_leg(args.output/"raw"/f"{stem}-leg-{i:04d}.json.gz", leg, index_path)
                entries.append(entry)
                legs.append(leg)
            if _stop_requested or any(leg.get("administrative_skip") for leg in legs):
                state["stop_reason"] = "administrative_pair_replay_required"
                break
            pairs = [{"sample": legs[i], "greedy": legs[i+1]} for i in range(0, len(legs), 2)]
            update_start, update_cpu_start = time.perf_counter(), time.process_time()
            update = scst_update(model, optimizer, pairs, minibatch_size=args.minibatch_size,
                stop_check=lambda: _stop_requested or time.time() >= deadline)
            update.update(wall_time_s=time.perf_counter()-update_start, cpu_time_s=time.process_time()-update_cpu_start)
            state["batches"] += 1
            state["pairs"] += len(pairs)
            state["rollouts"] += len(legs)
            state["pending_batch"] = None
            state["wall_time_s"] = prior_wall+time.perf_counter()-started
            state["stop_reason"] = None
            progress = {"algorithm": ALGORITHM, "batch": state["batches"], "pairs": state["pairs"],
                "rollouts": state["rollouts"], "wall_time_s": state["wall_time_s"],
                "sample": summarize_training_metrics([pair["sample"]["metrics"] for pair in pairs]),
                "greedy": summarize_training_metrics([pair["greedy"]["metrics"] for pair in pairs]),
                "sum_both_legs_worker_cpu_s": sum(leg["metrics"]["worker_cpu_s"] for leg in legs),
                "sum_both_legs_worker_wall_s": sum(leg["metrics"]["wall_time_s"] for leg in legs),
                "update": update, "raw_index": "raw/"+index_path.name,
                "raw_legs": entries, "scope": "fresh synthetic SCST diagnostics; not independent validation"}
            progress["measured_parent_plus_legs_cpu_s"] = time.process_time()-batch_parent_cpu_start
            if executor is not None:  # Inline workers already belong to the parent CPU clock.
                progress["measured_parent_plus_legs_cpu_s"] += progress["sum_both_legs_worker_cpu_s"]
            progress["cpu_measurement_scope"] = "through batch summary; use supervisor for complete process-tree overhead"
            state["last_progress"] = progress
            save_checkpoint(args.output/"latest.pt", model, optimizer, state, config, initialization)
            save_checkpoint(args.output/f"checkpoint-{state['batches']:06d}.pt", model, optimizer, state, config, initialization)
            _write_json(args.output/"progress.json", progress)
            with (args.output/"progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, allow_nan=False)+"\n")
            brief = {key: progress[key] for key in ("algorithm", "batch", "pairs", "rollouts", "wall_time_s", "update",
                "sum_both_legs_worker_cpu_s", "sum_both_legs_worker_wall_s", "measured_parent_plus_legs_cpu_s", "scope")}
            for leg in ("sample", "greedy"):
                brief[leg] = {key: progress[leg][key] for key in ("full_clear", "batch_episodes", "failed_clear_count",
                    "mean_penalized_time_s", "p95_penalized_time_s", "mean_common_lower_bound_s",
                    "ratio_of_mean_penalized_time_to_mean_bound", "all_clear_actual_time_over_mean_bound")}
            print(json.dumps(brief), flush=True)
    except TrainingStop:
        model, optimizer, state, config, initialization = restore_checkpoint(args.output/"latest.pt")
        state["stop_reason"] = "administrative_update_rolled_back"
    except BaseException:
        model, optimizer, state, config, initialization = restore_checkpoint(args.output/"latest.pt")
        state["stop_reason"] = "interrupted_or_error"
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        state["wall_time_s"] = prior_wall+time.perf_counter()-started
        state["stop_reason"] = state["stop_reason"] or ("batch_limit" if state["batches"] >= args.max_batches else "wall_or_training_deadline")
        save_checkpoint(args.output/"latest.pt", model, optimizer, state, config, initialization)
        _write_json(args.output/"status.json", state)
        if state.get("last_progress"):
            _write_json(args.output/"progress.json", state["last_progress"])
        signal.signal(signal.SIGTERM, previous_signal)
    return state


if __name__ == "__main__":
    main()

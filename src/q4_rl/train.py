"""Resumable CPU-only PPO on disjoint synthetic Q4 training scenes.

Gamma=lambda=1: return is negative *total billed virtual time*. A fixed 1000 s
unit improves numerical conditioning and is not fitted to any data. Failure
penalties replace incomplete time by max(time, 360000); no run is filtered for
being unsuccessful. Every batch keeps raw transitions, including fallback cost.
The lower-bound evaluator runs only after rollout and reward construction, for
reported diagnostics. Its result cannot enter observations, returns or updates.
This module never imports the validation database.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
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
from torch import nn

from .network import (CandidateActorCritic, TorchPolicy, CHECKPOINT_VERSION,
                      configure_cpu, feature_schema, model_from_metadata, pack_observations)


COST_UNIT_S = 1000.0
FAILURE_PENALTY_S = 360000.0
TRAIN_START, TRAIN_END = 8000000, 8099999
DEFAULT_DEADLINE = "2026-09-12T10:00:00+08:00"
_stop_requested = False


class TrainingStop(Exception):
    """Administrative stop: preserve the last complete learner transaction."""


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def training_case_spec(seed):
    """A prespecified eight-family, two-source-mode balanced curriculum."""
    from .scenarios import FAMILIES
    if type(seed) is not int or not TRAIN_START <= seed <= TRAIN_END:
        raise ValueError("seed is outside the isolated synthetic training partition")
    offset = seed - TRAIN_START
    return {"family": FAMILIES[offset % len(FAMILIES)],
            "source_mode": "mixed" if (offset // len(FAMILIES)) % 2 == 0 else "all_directional"}


def undiscounted_returns(costs, *, penalty_adjustment_s=0.0):
    """Gamma is exactly one, including unequal-duration macro actions."""
    if any(not math.isfinite(float(x)) or x < 0 for x in costs):
        raise ValueError("actual billed step costs must be finite and nonnegative")
    if not math.isfinite(penalty_adjustment_s) or penalty_adjustment_s < 0:
        raise ValueError("invalid failure adjustment")
    total = penalty_adjustment_s / COST_UNIT_S
    returns = []
    for cost in reversed(costs):
        total += cost / COST_UNIT_S
        returns.append(-total)
    return list(reversed(returns))


def attach_returns(records, *, actual_time_s, success):
    """Keep actual costs unchanged and record a separate terminal failure cost."""
    actual_time_s = float(actual_time_s)
    if not math.isfinite(actual_time_s) or actual_time_s < 0:
        raise ValueError("invalid episode time")
    costs = [float(row["cost_s"]) for row in records]
    if not math.isclose(sum(costs), actual_time_s, abs_tol=2e-5, rel_tol=0):
        raise ValueError("transition costs do not account for the complete episode")
    penalized = actual_time_s if success else max(actual_time_s, FAILURE_PENALTY_S)
    adjustment = penalized - actual_time_s
    returns = undiscounted_returns(costs, penalty_adjustment_s=adjustment)
    for i, (row, value) in enumerate(zip(records, returns)):
        row["return"] = value
        row["terminal_penalty_s"] = adjustment if i == len(records) - 1 else 0.0
    return {"actual_time_s": actual_time_s, "penalized_time_s": penalized,
            "failure_penalty_adjustment_s": adjustment, "reward_cost_s": penalized}


def rollout(task):
    """Worker sees scene identity only in the environment constructor."""
    worker_started, worker_cpu_started = time.perf_counter(), time.process_time()
    configure_cpu()
    from simulation import LocalResearchSimulator
    from .controller import run_q4_rl
    from .scenarios import build_case

    seed, action_seed = task["seed"], task["action_seed"]
    spec = training_case_spec(seed)
    if time.time() >= task["deadline_epoch"]:
        return {"seed": seed, "administrative_skip": "deadline_before_start", "records": []}
    model = model_from_metadata(task["network"])
    model.load_state_dict(task["model"])
    model.eval()
    # Construction consumes random numbers; restore the per-episode stream last.
    random.seed(action_seed)
    torch.manual_seed(action_seed)
    policy = TorchPolicy(model, deterministic=False) if task["mode"] == "ppo" else None
    remaining = max(1.0, task["deadline_epoch"] - time.time())
    simulator = LocalResearchSimulator(build_case(seed, split="train", **spec),
                                       max_real_duration_s=min(300.0, remaining))
    started, policy_cpu_started = time.perf_counter(), time.process_time()
    report = run_q4_rl(simulator.client(), policy=policy, max_decisions=task["max_decisions"],
                      action_deadline_epoch=task["deadline_epoch"])
    evaluation = simulator.evaluation()  # Only after policy termination.
    learning = report.learning
    records = [dict(row) for row in learning["transitions"]]
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model
                   and not report.error and not report.exit_error)
    if policy is not None:
        if len(records) != len(policy.records):
            raise ValueError("controller/policy transition count mismatch")
        for row, sample in zip(records, policy.records):
            for field in ("global_features", "candidate_features", "action_index"):
                if row[field] != sample[field]:
                    raise ValueError("controller/policy public observation mismatch")
            row.update(log_prob=sample["log_prob"], value=sample["value"])
    accounting = attach_returns(records, actual_time_s=report.virtual_time_s, success=success)
    policy_wall_s, policy_cpu_s = time.perf_counter() - started, time.process_time() - policy_cpu_started
    # Returns have already been finalized above. This independent, hindsight
    # value belongs exclusively to logs and is never attached to training rows.
    from experiments.q4_comparison_bounds import common_bound
    bound_started, bound_cpu_started = time.perf_counter(), time.process_time()
    bound = common_bound(evaluation["ground_truth"])
    lower = float(bound["common_lower_bound_s"])
    bound_wall_s = time.perf_counter() - bound_started
    bound_cpu_s = time.process_time() - bound_cpu_started
    metrics = {"seed": seed, "split": "train", **spec, "success": success,
               "actual_time_s": float(report.virtual_time_s), **accounting,
               "failed_clear_count": int(evaluation["failed_clear_count"]),
               "fallback_cost_s": float(learning.get("fallback_cost_s", 0.0)),
               "decisions": len(records), "wall_time_s": time.perf_counter() - worker_started,
               "worker_cpu_s": time.process_time() - worker_cpu_started,
               "policy_wall_s": policy_wall_s, "policy_cpu_s": policy_cpu_s,
               "posthoc_bound_wall_s": bound_wall_s, "posthoc_bound_cpu_s": bound_cpu_s,
               "common_lower_bound_s": lower,
               "actual_time_over_lower_bound": float(report.virtual_time_s) / lower if success else None,
               "penalized_time_over_lower_bound": accounting["penalized_time_s"] / lower,
               "completion_reason": report.completion_reason,
               "error": report.error, "exit_error": report.exit_error}
    result = {"seed": seed, "records": records, "metrics": metrics,
              "evaluation_phase": "after_policy_termination",
              "action_history": report.action_history,
              "observation_history": simulator.observation_history(),
              "evaluation": evaluation,
              "scenario": evaluation["ground_truth"], "common_bound": bound}
    if report.completion_reason == "training_deadline" or (
            not success and time.time() >= task["deadline_epoch"]):
        # Keep the interrupted raw attempt but never perform a selected partial
        # batch update near an administrative deadline. All reserved seeds are
        # replayed on resume, including the interrupted ones.
        result["administrative_skip"] = "training_deadline_during_episode"
    return result


def _tensor_batch(model, records):
    return pack_observations(records, global_dim=model.global_dim, candidate_dim=model.candidate_dim)


def _percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def summarize_training_metrics(metrics):
    """All cases retained; ratios of means never silently become mean ratios."""
    if not metrics:
        raise ValueError("cannot summarize an empty training batch")
    actual = [row["actual_time_s"] for row in metrics]
    penalized = [row["penalized_time_s"] for row in metrics]
    lower = [row["common_lower_bound_s"] for row in metrics]
    if any(value <= 0 or not math.isfinite(value) for value in lower):
        raise ValueError("invalid common diagnostic lower bound")
    all_complete = all(row["success"] for row in metrics)
    # Independent local RNG: diagnostic resampling cannot move the training RNG.
    diagnostic_rng = random.Random(904203)
    means = [statistics.mean(diagnostic_rng.choices(penalized, k=len(penalized))) for _ in range(500)]
    result = {"full_clear": sum(row["success"] for row in metrics),
              "full_clear_rate": sum(row["success"] for row in metrics) / len(metrics),
              "batch_episodes": len(metrics),
              "mean_actual_time_s": statistics.mean(actual), "p95_actual_time_s": _percentile(actual, .95),
              "mean_penalized_time_s": statistics.mean(penalized), "p95_penalized_time_s": _percentile(penalized, .95),
              "mean_common_lower_bound_s": statistics.mean(lower),
              "ratio_of_mean_penalized_time_to_mean_bound": sum(penalized) / sum(lower),
              "all_clear_actual_time_over_mean_bound": sum(actual) / sum(lower) if all_complete else None,
              "mean_of_penalized_ratios": statistics.mean(t / lb for t, lb in zip(penalized, lower)),
              "failed_clear_count": sum(row["failed_clear_count"] for row in metrics),
              "mean_penalized_time_bootstrap95_s": [_percentile(means, .025), _percentile(means, .975)],
              "uncertainty_scope": "descriptive resampling only; synthetic curriculum batch is not an independent efficacy estimate",
              "case_ratios": [{"seed": row["seed"], "actual_time_over_lower_bound":
                  row["actual_time_s"] / row["common_lower_bound_s"] if row["success"] else None,
                  "penalized_time_over_lower_bound": row["penalized_time_s"] / row["common_lower_bound_s"]}
                  for row in metrics]}
    for name in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s",
                 "posthoc_bound_wall_s", "posthoc_bound_cpu_s"):
        result["sum_episode_" + name] = sum(row[name] for row in metrics)
        result["mean_episode_" + name] = statistics.mean(row[name] for row in metrics)
    return result


def ppo_update(model, optimizer, records, *, epochs=3, minibatch_size=128,
               clip_epsilon=0.2, entropy_coefficient=0.005, value_coefficient=0.5,
               stop_check=None, actor_advantages=None):
    if not records:
        return {"updates": 0, "records": 0}
    if epochs < 1 or minibatch_size < 1:
        raise ValueError("positive PPO epochs/minibatch size required")
    old_log = torch.tensor([r["log_prob"] for r in records], dtype=torch.float32)
    old_value = torch.tensor([r["value"] for r in records], dtype=torch.float32)
    returns = torch.tensor([r["return"] for r in records], dtype=torch.float32)
    if actor_advantages is None:
        advantages = returns - old_value
    else:
        if len(actor_advantages) != len(records):
            raise ValueError("actor advantage count must match the complete batch")
        advantages = torch.tensor(actor_advantages, dtype=torch.float32)
        if advantages.ndim != 1 or not bool(torch.isfinite(advantages).all()):
            raise ValueError("actor advantages must be finite scalars")
    # Training-batch advantage centering is allowed; no observation statistics
    # or validation samples are estimated or retained as normalization state.
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    summaries = []
    model.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(records)).split(minibatch_size):
            if stop_check is not None and stop_check():
                raise TrainingStop("administrative stop during PPO update")
            rows = [records[int(i)] for i in indices]
            logits, values = model(*_tensor_batch(model, rows))
            distribution = Categorical(logits=logits)
            action = torch.tensor([r["action_index"] for r in rows], dtype=torch.long)
            log_prob = distribution.log_prob(action)
            ratio = (log_prob - old_log[indices]).exp()
            surrogate = torch.minimum(ratio * advantages[indices],
                ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages[indices])
            policy_loss = -surrogate.mean()
            # Huber loss remains well conditioned when retained failures have
            # much larger time than successes; the target itself is not clipped.
            value_loss = nn.functional.smooth_l1_loss(values, returns[indices])
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            summaries.append({"policy_loss": float(policy_loss.detach()),
                              "value_loss": float(value_loss.detach()),
                              "entropy": float(entropy.detach()),
                              "gradient_norm": float(gradient_norm),
                              "approximate_kl": float((old_log[indices] - log_prob).mean().detach())})
    return {"updates": len(summaries), "records": len(records), **{
        name: statistics.mean(row[name] for row in summaries) for name in summaries[0]}}


def imitation_update(model, optimizer, records, *, epochs=3, minibatch_size=128, stop_check=None):
    """Optional warm start from fresh synthetic heuristic rollouts only."""
    if not records:
        return {"updates": 0, "records": 0}
    losses = []
    model.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(records)).split(minibatch_size):
            if stop_check is not None and stop_check():
                raise TrainingStop("administrative stop during imitation update")
            rows = [records[int(i)] for i in indices]
            logits, values = model(*_tensor_batch(model, rows))
            actions = torch.tensor([row["action_index"] for row in rows], dtype=torch.long)
            returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32)
            loss = nn.functional.cross_entropy(logits, actions) + 0.1 * nn.functional.smooth_l1_loss(values, returns)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {"updates": len(losses), "records": len(records), "imitation_loss": statistics.mean(losses)}


def save_checkpoint(path, model, optimizer, state, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CHECKPOINT_VERSION, "network": model.metadata(),
               "feature_schema": feature_schema(), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "state": state, "config": config,
               "rng": {"python": random.getstate(), "torch": torch.get_rng_state()},
               "objective": {"gamma": 1.0, "lambda": 1.0, "cost_unit_s": COST_UNIT_S,
                             "failure_penalty_s": FAILURE_PENALTY_S},
               "device": "cpu"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, *, learning_rate=None):
    configure_cpu()
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved.get("version") != CHECKPOINT_VERSION or saved.get("feature_schema") != feature_schema():
        raise ValueError("checkpoint public schema mismatch")
    if saved.get("device") != "cpu" or saved.get("objective", {}).get("gamma") != 1.0:
        raise ValueError("checkpoint resource/objective contract mismatch")
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=saved["config"]["learning_rate"])
    optimizer.load_state_dict(saved["optimizer"])
    if learning_rate is not None and learning_rate != saved["config"]["learning_rate"]:
        raise ValueError("resuming cannot silently change optimizer learning rate")
    random.setstate(saved["rng"]["python"])
    torch.set_rng_state(saved["rng"]["torch"])
    return model, optimizer, saved["state"], saved["config"]


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_batch(path, value):
    temporary = Path(str(path) + ".tmp")
    # dumps uses the native JSON encoder; dump emits millions of tiny writes.
    # The decoded bytes are identical, including float spelling and separators.
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with gzip.open(temporary, "wb", compresslevel=6) as stream:
        stream.write(payload)
    temporary.replace(path)


def parse_deadline(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("deadline requires an explicit time zone")
    return stamp.timestamp()


def _configuration(args):
    # Paths, credentials, machine/user names never enter uploaded checkpoints.
    return {name: getattr(args, name) for name in (
        "hidden", "learning_rate", "epochs", "minibatch_size", "batch_episodes",
        "warmstart_episodes", "max_decisions", "random_seed", "scenario_start",
        "scenario_end", "entropy_coefficient")}


def main(argv=None):
    global _stop_requested
    _stop_requested = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-budget", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--warmstart-episodes", type=int, default=32)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=424242)
    parser.add_argument("--scenario-start", type=int, default=TRAIN_START)
    parser.add_argument("--scenario-end", type=int, default=TRAIN_END)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--max-batches", type=int, default=100000)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--deadline", default=DEFAULT_DEADLINE)
    parser.add_argument("--progress-interval-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if not 1 <= args.workers < args.cpu_budget <= 60:
        parser.error("workers + 1 learner must fit CPU budget (maximum 60)")
    if not TRAIN_START <= args.scenario_start <= args.scenario_end <= TRAIN_END:
        parser.error("training seeds must remain inside 8000000..8099999")
    if (min(args.epochs, args.minibatch_size, args.batch_episodes, args.max_decisions, args.max_batches) < 1
            or args.warmstart_episodes < 0 or args.learning_rate <= 0 or args.max_wall_seconds <= 0):
        parser.error("invalid training budget")
    configure_cpu()
    old_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    deadline = min(parse_deadline(args.deadline), time.time() + args.max_wall_seconds)
    if deadline <= time.time():
        parser.error("training deadline has passed; a new explicit deadline is required")
    config = _configuration(args)
    if args.resume:
        model, optimizer, state, old_config = restore_checkpoint(args.resume)
        if old_config != config:
            parser.error("resume configuration differs; preserve frozen settings or create an explicit new experiment")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("new training output must be empty; use --resume for existing runs")
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        model = CandidateActorCritic(hidden=args.hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        state = {"next_seed": args.scenario_start, "episodes": 0, "attempted_episodes": 0,
                 "warmstart_completed": 0, "ppo_batches": 0, "batches": 0,
                 "wall_time_s": 0.0, "pending_batch": None, "stop_reason": None}
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_checkpoint(args.output / "random.pt", model, optimizer, state, config)
    save_checkpoint(args.output / "latest.pt", model, optimizer, state, config)
    started, prior_wall = time.perf_counter(), state["wall_time_s"]
    next_review = started + args.progress_interval_seconds
    executor = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=configure_cpu) if args.workers > 1 else None)
    try:
        while state["batches"] < args.max_batches and time.time() < deadline - 5 and not _stop_requested:
            if state["pending_batch"] is None:
                if state["next_seed"] > args.scenario_end:
                    state["stop_reason"] = "training_partition_exhausted"
                    break
                mode = "imitation" if state["warmstart_completed"] < args.warmstart_episodes else "ppo"
                count = min(args.batch_episodes, args.scenario_end - state["next_seed"] + 1)
                if mode == "imitation":
                    count = min(count, args.warmstart_episodes - state["warmstart_completed"])
                seeds = list(range(state["next_seed"], state["next_seed"] + count))
                state["next_seed"] += count
                state["attempted_episodes"] += count
                state["pending_batch"] = {"mode": mode, "seeds": seeds,
                                           "action_seeds": [random.randrange(2**31) for _ in seeds]}
                state["wall_time_s"] = prior_wall + time.perf_counter() - started
                save_checkpoint(args.output / "latest.pt", model, optimizer, state, config)
            pending = state["pending_batch"]
            tasks = [{"seed": seed, "action_seed": action_seed, "mode": pending["mode"],
                      "model": model.state_dict(), "network": model.metadata(),
                      "max_decisions": args.max_decisions, "deadline_epoch": deadline}
                     for seed, action_seed in zip(pending["seeds"], pending["action_seeds"])]
            batch = list(executor.map(rollout, tasks)) if executor else [rollout(task) for task in tasks]
            _write_batch(args.output / f"batch-{state['batches']:06d}.json.gz", batch)
            if _stop_requested or any(row.get("administrative_skip") for row in batch):
                state["stop_reason"] = "deadline_in_reserved_batch"
                # No biased partial update; checkpoint retains the reservation
                # so resumption reruns this entire batch from the same weights.
                break
            records = [record for row in batch for record in row["records"]]
            update_started, update_cpu_started = time.perf_counter(), time.process_time()
            if pending["mode"] == "imitation":
                update = imitation_update(model, optimizer, records, epochs=args.epochs,
                                          minibatch_size=args.minibatch_size,
                                          stop_check=lambda: _stop_requested or time.time() >= deadline)
                state["warmstart_completed"] += len(batch)
            else:
                update = ppo_update(model, optimizer, records, epochs=args.epochs,
                                    minibatch_size=args.minibatch_size,
                                    entropy_coefficient=args.entropy_coefficient,
                                    stop_check=lambda: _stop_requested or time.time() >= deadline)
                state["ppo_batches"] += 1
            update["wall_time_s"] = time.perf_counter() - update_started
            update["cpu_time_s"] = time.process_time() - update_cpu_started
            state["episodes"] += len(batch)
            state["batches"] += 1
            state["pending_batch"] = None
            state["wall_time_s"] = prior_wall + time.perf_counter() - started
            state["stop_reason"] = None
            metrics = [row["metrics"] for row in batch]
            progress = {"batch": state["batches"], "phase": pending["mode"], "episodes": state["episodes"],
                        "next_seed": state["next_seed"], "wall_time_s": state["wall_time_s"],
                        **summarize_training_metrics(metrics),
                        "update": update,
                        "scope": "synthetic training diagnostics with a post-termination historical lower bound; not independent validation"}
            _write_json(args.output / "progress.json", progress)
            with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, allow_nan=False) + "\n")
            save_checkpoint(args.output / "latest.pt", model, optimizer, state, config)
            save_checkpoint(args.output / f"checkpoint-{state['batches']:06d}.pt", model, optimizer, state, config)
            if pending["mode"] == "imitation" and state["warmstart_completed"] == args.warmstart_episodes:
                save_checkpoint(args.output / "warmstart.pt", model, optimizer, state, config)
            if time.perf_counter() >= next_review:
                _write_json(args.output / "review_due.json", {"batch": state["batches"],
                    "checkpoint": f"checkpoint-{state['batches']:06d}.pt", "review_requested": True,
                    "reason": "Run the separately frozen development evaluation before promotion."})
                next_review = time.perf_counter() + args.progress_interval_seconds
            print(json.dumps(progress, allow_nan=False), flush=True)
        if state["stop_reason"] is None:
            state["stop_reason"] = "batch_limit" if state["batches"] >= args.max_batches else "wall_or_training_deadline"
    except TrainingStop:
        # Roll back any incomplete gradient transaction, including its RNG.
        model, optimizer, state, config = restore_checkpoint(args.output / "latest.pt")
        state["stop_reason"] = "administrative_stop_reserved_batch_retained"
    except BaseException:
        model, optimizer, state, config = restore_checkpoint(args.output / "latest.pt")
        state["stop_reason"] = "interrupted_or_error"
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        state["wall_time_s"] = prior_wall + time.perf_counter() - started
        save_checkpoint(args.output / "latest.pt", model, optimizer, state, config)
        _write_json(args.output / "status.json", state)
        signal.signal(signal.SIGTERM, old_sigterm)
    return state


if __name__ == "__main__":
    main()

"""Bounded CPU PPO with full-episode BPTT; never shuffle individual RNN steps."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical

from simulation import LocalResearchSimulator, random_scenario
from .cpu_runtime import require_cpu
from .controller import feature_schema
from .portable_checkpoint import portable_paths
from .train import (TRAINING_SEED_RANGES, legal_training_seed, restore_sampling_budget,
                    sampling_stop_reason, source_manifest, git_version)
from .recurrent_network import (ALGORITHM, RecurrentCandidateActorCritic, RecurrentPolicy,
                                pack_episodes, initialize_from_mlp, model_from_checkpoint)
from .recurrent import run_recurrent_search

_worker_model = None


def episode(task):
    """One complete, independently reset trajectory from observation-only control."""
    require_cpu()
    seed = task["seed"]
    if not legal_training_seed(seed):
        raise ValueError("Training seed outside declared training-only partitions")
    if task.get("deadline") is not None and time.time() >= task["deadline"]:
        return [], dict(seed=seed, deadline_skipped=True)
    torch.set_num_threads(1)
    global _worker_model
    if (_worker_model is None or _worker_model.hidden != task["hidden"]
            or _worker_model.memory_hidden != task["memory_hidden"]):
        _worker_model = RecurrentCandidateActorCritic(task["hidden"], task["memory_hidden"])
    _worker_model.load_state_dict(task["weights"])
    _worker_model.eval()
    torch.manual_seed(task["action_seed"])
    np.random.seed(task["action_seed"] % 2**32)
    random.seed(task["action_seed"])
    records = []
    policy = RecurrentPolicy(_worker_model, deterministic=False)

    def record(features, context, action, target, selection, cost):
        records.append(dict(features=np.asarray(features, dtype=np.float32),
                            context=np.asarray(context, dtype=np.float32), action=action,
                            log_prob=selection[1], value=selection[2], reward=-cost / 1000,
                            episode_start=not bool(records), terminal=False))

    simulator = LocalResearchSimulator(random_scenario(3, seed), max_real_duration_s=300)
    started = time.perf_counter()
    report = run_recurrent_search(simulator.client(), policy=policy, recorder=record,
        max_decisions=task["max_decisions"], action_deadline_epoch=task.get("deadline"))
    evaluation = simulator.evaluation()  # Post-termination only, never a network input.
    if report.completion_reason == "training_deadline":
        # Whole-episode training never treats administrative cutoff as a failure
        # or as a terminal value target. No fragment from this episode is used.
        return [], dict(seed=seed, deadline_skipped=True, completion_reason=report.completion_reason)
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model)
    accounted = -sum(r["reward"] for r in records) * 1000
    initial = report.learning["initial_scan_virtual_time_s"]
    if records:
        records[-1]["reward"] -= max(0., report.virtual_time_s - initial - accounted) / 1000
        if not success:
            records[-1]["reward"] -= 360.0
        records[-1]["terminal"] = True
    return records, dict(seed=seed, success=success, records=len(records),
        virtual_time_s=report.virtual_time_s, failed_clear_count=evaluation["failed_clear_count"],
        reward_cost_s=-sum(r["reward"] for r in records) * 1000,
        initial_scan_virtual_time_s=initial, wall_time_s=time.perf_counter() - started,
        completion_reason=report.completion_reason, memory_steps=report.learning["memory_steps"],
        memory_reset_verified=policy.hidden is None,
        metric_scope="training mechanism/return accounting; not performance evaluation")


def compute_gae(records, gae_lambda=.95):
    """Gamma=1 terminal GAE within exactly one complete episode, never padding."""
    if not records or not records[0].get("episode_start") or not records[-1].get("terminal"):
        raise ValueError("GAE needs a complete episode, not a truncated administrative rollout")
    if any(r.get("episode_start") for r in records[1:]) or any(r.get("terminal") for r in records[:-1]):
        raise ValueError("GAE must not cross episode boundaries")
    advantage = next_value = 0.0
    for row in reversed(records):
        delta = row["reward"] + next_value - row["value"]
        advantage = delta + gae_lambda * advantage
        row["advantage"] = advantage
        row["return"] = advantage + row["value"]
        next_value = row["value"]
    return records


def update(model, optimizer, episodes, args, *, stop_at=None, on_optimizer_step=None):
    """Shuffle entire episodes; loss masks exclude both padding dimensions."""
    require_cpu()
    if not episodes or any(not trajectory for trajectory in episodes):
        raise ValueError("Recurrent PPO requires complete nonempty episode sequences")
    flat = [r for trajectory in episodes for r in trajectory]
    advantages = np.asarray([r["advantage"] for r in flat], dtype=np.float32)
    normalized = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
    # Copy records so repeated update diagnostics do not mutate original GAE.
    normalized_episodes = []
    offset = 0
    for trajectory in episodes:
        normalized_episodes.append([dict(r, advantage=float(normalized[offset + i])) for i, r in enumerate(trajectory)])
        offset += len(trajectory)
    metrics = []

    def summarize(**extra):
        means = {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]} if metrics else {}
        return dict(means, optimizer_steps=len(metrics), whole_episodes=len(episodes),
                    transitions=len(flat), bptt="whole-episode", **extra)

    model.train()
    for _ in range(args.epochs):
        ordering = np.random.permutation(len(episodes))
        for start in range(0, len(ordering), args.episodes_per_minibatch):
            if stop_at is not None and time.monotonic() >= stop_at:
                return summarize(deadline_reached=True)
            batch = pack_episodes([normalized_episodes[i] for i in ordering[start:start + args.episodes_per_minibatch]])
            logits, values, _ = model.forward_sequence(batch["features"], batch["context"],
                                                       batch["candidate_mask"], batch["time_mask"])
            valid = batch["time_mask"]
            distribution = Categorical(logits=logits[valid])
            log_ratio = distribution.log_prob(batch["action"][valid]) - batch["log_prob"][valid]
            ratio = log_ratio.exp()
            advantage = batch["advantage"][valid]
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - args.clip, 1 + args.clip) * advantage).mean()
            value_loss = torch.nn.functional.mse_loss(values[valid], batch["return"][valid])
            entropy = distribution.entropy().mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite recurrent PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if on_optimizer_step is not None:
                on_optimizer_step()
            kl = float(((ratio - 1) - log_ratio).mean().detach())
            metrics.append(dict(loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()), entropy=float(entropy.detach()),
                approx_kl=kl, grad_norm=float(grad), valid_steps=int(valid.sum())))
            if args.target_kl > 0 and kl > args.target_kl:
                return summarize(kl_stopped=True)
    return summarize()


def training_spec(args):
    return dict(gamma=1.0, gae_lambda=args.gae_lambda, epochs=args.epochs,
        episodes_per_update=args.episodes_per_update, episodes_per_minibatch=args.episodes_per_minibatch,
        lr=args.lr, clip=args.clip, value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        target_kl=args.target_kl, max_grad_norm=args.max_grad_norm, max_decisions=args.max_decisions,
        recurrent_training="whole-episode-BPTT-from-zero-on-every-PPO-minibatch",
        advantage_normalization="valid-transitions-of-current-on-policy-batch",
        truncation="discard-administratively-interrupted-episodes",
        reward="negative-full-virtual-time/1000; terminal-failure-penalty-360",
        dropout=0.0)


def save_checkpoint(path, model, optimizer, args, state):
    require_cpu()
    payload = dict(algorithm=ALGORITHM, hidden=model.hidden, memory_hidden=model.memory_hidden,
        memory_spec=model.memory_spec, architecture=model.architecture,
        action_distribution=model.action_distribution, action_schema=model.action_schema,
        feature_version="v3", feature_schema=feature_schema("v3"), training_spec=training_spec(args),
        model={k: v.detach().cpu() for k, v in model.state_dict().items()}, optimizer=optimizer.state_dict(),
        args=vars(args), state=state, source_manifest=source_manifest(), git_commit=git_version(),
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        saved_utc=datetime.now(timezone.utc).isoformat(), compute_policy="CPU-only",
        torch_version=torch.__version__, recurrent_state_checkpoint="episode-boundaries-only; no in-flight hidden carried on resume")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(portable_paths(payload), temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--memory-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9113101)
    parser.add_argument("--scenario-start", type=int)
    parser.add_argument("--scenario-end", type=int)
    parser.add_argument("--scenario-range", type=int, nargs=2, metavar=("START", "END"),
                        help="Inclusive training range; alternative to scenario-start/end")
    parser.add_argument("--max-attempted-episodes", type=int, default=199936)
    parser.add_argument("--updates", type=int, default=1562)
    parser.add_argument("--episodes-per-update", type=int, default=128)
    parser.add_argument("--episodes-per-minibatch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=.2)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--entropy-coef", type=float, default=.005)
    parser.add_argument("--value-coef", type=float, default=.5)
    parser.add_argument("--target-kl", type=float, default=.03)
    parser.add_argument("--max-grad-norm", type=float, default=.5)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--max-wall-s", "--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--checkpoint-seconds", "--checkpoint-interval", type=float, default=600,
                        help="Checkpoint interval in elapsed seconds")
    parser.add_argument("--deadline-utc")
    args = parser.parse_args(argv)
    if args.scenario_range is not None:
        if args.scenario_start is not None or args.scenario_end is not None:
            parser.error("Use scenario-range or scenario-start/end, not both")
        args.scenario_start, args.scenario_end = args.scenario_range
    else:
        args.scenario_start = 1600001 if args.scenario_start is None else args.scenario_start
        args.scenario_end = 1799999 if args.scenario_end is None else args.scenario_end
    require_cpu(args.device)
    if (args.workers < 0 or min(args.num_threads, args.episodes_per_update, args.episodes_per_minibatch,
                               args.epochs, args.max_decisions, args.hidden) < 1
            or args.max_decisions > 512 or args.max_attempted_episodes < 0 or args.updates < 0):
        parser.error("Invalid worker/sequence/episode budget; whole-episode horizon must be 1..512")
    if not any(first <= args.scenario_start <= args.scenario_end <= last for first, last in TRAINING_SEED_RANGES):
        parser.error("scenario-start/end must be a contiguous declared training-only interval")
    if (not 0 <= args.gae_lambda <= 1 or not math_finite_positive(args.max_wall_s)
            or not math_finite_positive(args.checkpoint_seconds)):
        parser.error("Invalid GAE or wall-clock budget")
    if (not all(np.isfinite(value) for value in (args.lr, args.clip, args.entropy_coef, args.value_coef,
                                                args.target_kl, args.max_grad_norm))
            or args.lr < 0 or not 0 < args.clip < 1 or min(args.entropy_coef, args.value_coef, args.target_kl) < 0
            or args.max_grad_norm <= 0):
        parser.error("Invalid PPO optimizer hyperparameters")
    if args.resume and args.initialize_from:
        parser.error("resume and initialize-from are mutually exclusive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Nonempty output requires explicit --resume; initial latest is written before sampling")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    model = RecurrentCandidateActorCritic(args.hidden, args.memory_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = dict(update=0, optimizer_steps=0, episodes=0, attempted_episodes=0,
                 next_seed=args.scenario_start, stop_reason=None, elapsed_training_s=0.0)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        restored = model_from_checkpoint(payload)
        if (restored.hidden != args.hidden or restored.memory_hidden != args.memory_hidden
                or payload.get("training_spec") != training_spec(args)
                or payload.get("source_manifest") != source_manifest()
                or payload.get("args", {}).get("seed") != args.seed):
            parser.error("Resume source, architecture, random seed or recurrent training semantics changed")
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(payload["optimizer"])
        state = dict(payload["state"])
        try:
            restore_sampling_budget(state, args, payload["args"])
        except ValueError as error:
            parser.error(str(error))
        torch.set_rng_state(payload["torch_rng"].cpu())
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    elif args.initialize_from:
        payload = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        initialize_from_mlp(model, payload)
        state["initialization"] = dict(path=str(args.initialize_from),
            sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
            algorithm=payload["algorithm"], source_manifest=payload.get("source_manifest"),
            preservation="same input history logits/value at zero residual; no post-training or cross-platform trajectory guarantee",
            optimizer="fresh", memory="fresh GRU; zero readout")
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
    if not args.resume:
        save_checkpoint(args.output / "initialized.pt", model, optimizer, args, state.copy())
    started = time.monotonic()
    elapsed_before = state["elapsed_training_s"]
    stop_at = started + max(0., args.max_wall_s - elapsed_before)
    global_deadline = None
    if args.deadline_utc:
        global_deadline = datetime.fromisoformat(args.deadline_utc)
        if global_deadline.tzinfo is None:
            parser.error("deadline-utc needs an explicit timezone")
        stop_at = min(stop_at, started + global_deadline.timestamp() - time.time())
    (args.output / "config.json").write_text(json.dumps(portable_paths(dict(vars(args), algorithm=ALGORITHM,
        memory_spec=model.memory_spec, training_spec=training_spec(args), source_manifest=source_manifest())),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) if args.workers else None
    last_checkpoint = started
    log = (args.output / "training.jsonl").open("a", encoding="utf-8")

    def optimizer_completed():
        # A caught interrupt may occur inside the sequence update. Keep the
        # counter aligned with optimizer tensors saved by the finalizer.
        state["optimizer_steps"] += 1

    try:
        while state["update"] < args.updates and time.monotonic() < stop_at and not sampling_stop_reason(state, args):
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            tasks = []
            for _ in range(args.episodes_per_update):
                if sampling_stop_reason(state, args):
                    break
                tasks.append(dict(seed=state["next_seed"], weights=weights, hidden=args.hidden,
                    memory_hidden=args.memory_hidden, action_seed=random.randrange(2**31),
                    max_decisions=args.max_decisions,
                    deadline=time.time() + max(0., stop_at - time.monotonic())))
                state["next_seed"] += 1
                state["attempted_episodes"] += 1
            state["elapsed_training_s"] = elapsed_before + time.monotonic() - started
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
            collect_start = time.monotonic()
            if pool:
                outcomes = list(pool.map(episode, tasks))
            else:
                # Inline sampling resets worker RNG. Preserve the learner RNG,
                # giving the same subsequent seed/sequence order as process mode.
                rngs = torch.get_rng_state(), np.random.get_state(), random.getstate()
                try:
                    outcomes = [episode(task) for task in tasks]
                finally:
                    torch.set_rng_state(rngs[0]); np.random.set_state(rngs[1]); random.setstate(rngs[2])
                    torch.set_num_threads(args.num_threads)
            collect_s = time.monotonic() - collect_start
            state["episodes"] += sum(not metrics.get("deadline_skipped", False) for _, metrics in outcomes)
            trajectories = [compute_gae(records, args.gae_lambda) for records, _ in outcomes if records]
            optimize_start = time.monotonic()
            losses = update(model, optimizer, trajectories, args, stop_at=stop_at,
                            on_optimizer_step=optimizer_completed) if trajectories else {"optimizer_steps": 0}
            if losses["optimizer_steps"]:
                state["update"] += 1
            entry = dict(stage="recurrent_ppo" if losses["optimizer_steps"] else "deadline_or_empty",
                update=state["update"], attempted_episodes=state["attempted_episodes"],
                completed_episodes=state["episodes"], next_seed=state["next_seed"],
                episodes=[metrics for _, metrics in outcomes], losses=losses,
                collect_wall_s=collect_s, optimize_wall_s=time.monotonic() - optimize_start)
            log.write(json.dumps(entry, allow_nan=False) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in entry.items() if k != "episodes"}), flush=True)
            state["elapsed_training_s"] = elapsed_before + time.monotonic() - started
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                save_checkpoint(args.output / f"ppo_{state['update']:06d}.pt", model, optimizer, args, state.copy())
                last_checkpoint = time.monotonic()
            if not losses["optimizer_steps"]:
                break
    except BaseException:
        state["stop_reason"] = "interrupted"
        raise
    finally:
        state["elapsed_training_s"] = elapsed_before + time.monotonic() - started
        state["stop_reason"] = state.get("stop_reason") or sampling_stop_reason(state, args) or (
            "update_limit" if state["update"] >= args.updates else
            "global_deadline" if global_deadline and time.time() >= global_deadline.timestamp() else
            "wall_time_limit" if time.monotonic() >= stop_at else "no_training_records")
        save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
        save_checkpoint(args.output / f"ppo_{state['update']:06d}.pt", model, optimizer, args, state.copy())
        log.close()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return 0


def math_finite_positive(value):
    import math
    return math.isfinite(value) and value > 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Observation-only, same-scenario REINFORCE with a greedy rollout baseline.

The independent baseline trajectory sees the same initial world, but no sampled
policy actions. It is used only as a scalar training control variate. One update
uses fresh trajectories and one accumulated gradient, averaged over episodes,
not over the variable number of decisions. No state-search policy is imported.
"""

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
from .controller import DeepRLSearch, FEATURE_DIMS
from .network import CandidateActorCritic, TorchPolicy, pack_observations
from .train import (git_version, initialize_from, save_checkpoint,
                    source_manifest, validate_resume)


TRAINER = "paired-greedy-reinforce-v1"
PAIRED_FEATURE_VERSIONS = ("v1", "v2", "v3")
_worker_model = None


def legal_seed(seed):
    return 100001 <= seed < 200000 or 2000 <= seed <= 5099


def penalized_cost(report, evaluation):
    success = bool(evaluation["all_cleared"]
                   and report.completion_certified_under_model
                   and not report.error and not report.exit_error)
    return report.virtual_time_s + (0 if success else 360000), success


def paired_episode(task):
    global _worker_model
    seed, weights, hidden, action_seed, version, max_decisions, deadline = task
    if version not in PAIRED_FEATURE_VERSIONS:
        raise ValueError("paired REINFORCE supports only feature versions v1, v2 and v3")
    if not legal_seed(seed):
        raise ValueError("seed outside declared training ranges")
    if time.time() >= deadline:
        return [], {"seed": seed, "deadline_skipped": True}
    torch.set_num_threads(1)
    if (_worker_model is None or _worker_model.hidden != hidden
            or _worker_model.feature_dim != FEATURE_DIMS[version]):
        _worker_model = CandidateActorCritic(hidden, FEATURE_DIMS[version])
    _worker_model.load_state_dict(weights)
    _worker_model.eval()
    # Construction consumes Torch RNG even though those initial weights are
    # replaced. Re-seed afterwards so fresh/resumed and reused workers agree.
    torch.manual_seed(action_seed)
    np.random.seed(action_seed % 2**32)
    random.seed(action_seed)
    records = []

    def record(features, context, action, teacher, selection, cost):
        records.append(dict(features=np.asarray(features, dtype=np.float32),
                            context=np.asarray(context, dtype=np.float32),
                            action=action, log_prob=selection[1]))

    started = time.perf_counter()
    outcomes = []
    # Generate a separate simulator for each trajectory. Their state, history,
    # coverage and rewards cannot leak into the other policy's observation.
    for deterministic in (False, True):
        if deterministic:
            # Keep even any future randomized geometry fallback independent of
            # the sampled trajectory's action count / consumed random numbers.
            torch.manual_seed(seed ^ 0x4A73)
            np.random.seed(seed ^ 0x4A73)
            random.seed(seed ^ 0x4A73)
        simulator = LocalResearchSimulator(random_scenario(3, seed), max_real_duration_s=300)
        if version == "v3":
            from .joint_scan import JointScanRLSearch
            controller_class = JointScanRLSearch
        else:
            controller_class = DeepRLSearch
        controller = controller_class(simulator.client(),
            TorchPolicy(_worker_model, deterministic=deterministic),
            recorder=None if deterministic else record, max_decisions=max_decisions,
            action_deadline_epoch=deadline, feature_version=version)
        before = time.perf_counter()
        report = controller.run()
        elapsed = time.perf_counter() - before
        if report.completion_reason == "training_deadline":
            return [], {"seed": seed, "deadline_skipped": True,
                        "completed_trajectories": len(outcomes)}
        evaluation = simulator.evaluation()  # Only after this trajectory exits.
        cost, success = penalized_cost(report, evaluation)
        outcomes.append(dict(cost_s=cost, virtual_time_s=report.virtual_time_s,
                             success=success, failed_clears=evaluation["failed_clear_count"],
                             decisions=report.learning["decisions"],
                             action_counts=report.learning["action_counts"],
                             fallback_time_s=report.learning["fallback_virtual_time_s"],
                             wall_time_s=elapsed))
    sampled, baseline = outcomes
    # Negative cost reward: positive advantage means this sample beat the
    # independently executed greedy policy on the same scenario.
    advantage = (baseline["cost_s"] - sampled["cost_s"]) / 1000
    for row in records:
        row["advantage"] = advantage
        row["trajectory_length"] = len(records)
    return records, dict(seed=seed, action_seed=action_seed, sampled=sampled,
                         baseline=baseline, advantage=advantage,
                         wall_time_s=time.perf_counter() - started)


def reinforce_loss(log_probs, advantages, entropies, lengths, episode_count, entropy_coef):
    """Sum likelihoods within each trajectory; average over trajectories.

    Dividing by transition count instead would reweight a variable-horizon
    task. Entropy is averaged within each trajectory as an explicit auxiliary
    heuristic; at coefficient zero the estimator targets expected total cost.
    """
    return (-torch.sum(log_probs * advantages)
            - entropy_coef * torch.sum(entropies / lengths)) / episode_count


def update(model, optimizer, records, episode_count, args, stop_at):
    if not records or episode_count <= 0:
        raise ValueError("no complete paired trajectories")
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_entropy, largest_logprob_gap = 0.0, 0.0, 0.0
    for start in range(0, len(records), args.minibatch):
        if time.monotonic() >= stop_at:
            optimizer.zero_grad(set_to_none=True)
            return dict(optimizer_steps=0, deadline_reached=True)
        batch = records[start:start + args.minibatch]
        logits, _ = model(*pack_observations(batch, device))
        distribution = Categorical(logits=logits)
        actions = torch.tensor([r["action"] for r in batch], device=device)
        advantages = torch.tensor([r["advantage"] for r in batch], device=device)
        lengths = torch.tensor([r["trajectory_length"] for r in batch], device=device)
        log_probs = distribution.log_prob(actions)
        old = torch.tensor([r["log_prob"] for r in batch], device=device)
        gap = float((log_probs.detach() - old).abs().max())
        largest_logprob_gap = max(largest_logprob_gap, gap)
        if gap > 1e-3:
            raise RuntimeError("collection/update policy mismatch before REINFORCE step")
        entropies = distribution.entropy()
        loss = reinforce_loss(log_probs, advantages, entropies, lengths,
                              episode_count, args.entropy_coef)
        loss.backward()
        total_loss += float(loss.detach())
        total_entropy += float((entropies.detach() / lengths).sum()) / episode_count
    if time.monotonic() >= stop_at:
        optimizer.zero_grad(set_to_none=True)
        return dict(optimizer_steps=0, deadline_reached=True)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    return dict(optimizer_steps=1, loss=total_loss, entropy=total_entropy,
                grad_norm=float(norm), largest_collection_logprob_gap=largest_logprob_gap)


def run_reinforce_search(client, *, checkpoint, **kwargs):
    """Common evaluation callback with explicit training-method attribution."""
    from . import run_rl_search
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("state", {}).get("trainer") != TRAINER:
        raise ValueError("not a paired REINFORCE checkpoint")
    report = run_rl_search(client, checkpoint=checkpoint, **kwargs)
    report.learning["controller_schema_algorithm"] = report.learning["algorithm"]
    report.learning["algorithm"] = TRAINER
    report.learning["training_method"] = TRAINER
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--feature-version", choices=PAIRED_FEATURE_VERSIONS, default="v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--seed", type=int, default=9112028)
    parser.add_argument("--scenario-start", type=int, default=100001)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--pairs-per-update", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--max-wall-s", type=float, default=1800)
    parser.add_argument("--checkpoint-seconds", type=float, default=600)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args(argv)
    if args.resume and args.initialize_from:
        parser.error("resume and initialize-from are mutually exclusive")
    if args.workers < 0 or min(args.pairs_per_update, args.minibatch, args.updates,
                              args.max_decisions, args.hidden) < 1:
        parser.error("invalid worker, batch or architecture setting")
    if min(args.lr, args.max_wall_s, args.checkpoint_seconds, args.max_grad_norm) <= 0 or args.entropy_coef < 0:
        parser.error("invalid optimizer or time setting")
    if not legal_seed(args.scenario_start):
        parser.error("scenario-start is outside training partitions")
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        parser.error("deadline must include a timezone")
    started = time.monotonic()
    stop_at = min(started + args.max_wall_s, started + deadline.timestamp() - time.time())
    if stop_at <= started:
        parser.error("training deadline has already elapsed")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("nonempty output; choose a new trial or explicitly resume")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    model = CandidateActorCritic(args.hidden, FEATURE_DIMS[args.feature_version]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = dict(trainer=TRAINER, update=0, optimizer_steps=0, episodes=0,
                 paired_baseline_episodes=0, next_seed=args.scenario_start)
    if args.resume:
        payload = torch.load(args.resume, map_location=args.device, weights_only=False)
        validate_resume(payload, args)
        if payload.get("state", {}).get("trainer") != TRAINER:
            parser.error("resume requires a paired REINFORCE training state")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        state = payload["state"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        if args.device.startswith("cuda") and "cuda_rng" in payload:
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_checkpoint(args.output / "random.pt", model, optimizer, args, state.copy())
    if args.initialize_from:
        payload = torch.load(args.initialize_from, map_location=args.device, weights_only=False)
        initialize_from(model, payload)
        state["initialization"] = dict(path=str(args.initialize_from),
            sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
            source_state=payload.get("state"), source_git_commit=payload.get("git_commit"))
        save_checkpoint(args.output / "initialized.pt", model, optimizer, args, state.copy())
    config = dict(**vars(args), trainer=TRAINER, git_commit=git_version(),
                  source_manifest=source_manifest(), reward_scale_s=1000, gamma=1,
                  gradient_normalization="episodes", baseline="current snapshot greedy, same scenario")
    config_path = args.output / f"config-u{state['update']:06d}-{time.time_ns()}.json"
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    executor = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) if args.workers else None
    log = (args.output / "training.jsonl").open("a", encoding="utf-8")
    last_checkpoint = time.monotonic()
    try:
        while state["update"] < args.updates and time.monotonic() < stop_at:
            collected_at = time.monotonic()
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            tasks = []
            for _ in range(args.pairs_per_update):
                seed = state["next_seed"]
                if not legal_seed(seed):
                    seed = 100001
                state["next_seed"] = seed + 1
                tasks.append((seed, weights, args.hidden, random.randrange(2**31),
                    args.feature_version, args.max_decisions,
                    time.time() + max(0, stop_at - time.monotonic())))
            results = list(executor.map(paired_episode, tasks)) if executor else [paired_episode(t) for t in tasks]
            completed = [(r, m) for r, m in results if not m.get("deadline_skipped")]
            state["episodes"] += len(completed)
            state["paired_baseline_episodes"] += len(completed)
            collect_time = time.monotonic() - collected_at
            if len(completed) != len(tasks) or time.monotonic() >= stop_at:
                log.write(json.dumps(dict(stage="deadline", episodes=[m for _, m in results])) + "\n")
                log.flush()
                break
            records = [r for trajectory, _ in completed for r in trajectory]
            optimized_at = time.monotonic()
            losses = update(model, optimizer, records, len(completed), args, stop_at)
            state["optimizer_steps"] += losses["optimizer_steps"]
            if losses["optimizer_steps"]:
                state["update"] += 1
            metrics = [m for _, m in completed]
            entry = dict(stage="reinforce" if losses["optimizer_steps"] else "deadline", update=state["update"],
                total_episodes=state["episodes"], total_baseline_episodes=state["paired_baseline_episodes"],
                mean_virtual_time_s=float(np.mean([m["sampled"]["virtual_time_s"] for m in metrics])),
                mean_baseline_time_s=float(np.mean([m["baseline"]["virtual_time_s"] for m in metrics])),
                mean_paired_saving_s=float(np.mean([m["advantage"] * 1000 for m in metrics])),
                success_count=sum(m["sampled"]["success"] for m in metrics),
                episodes=metrics, losses=losses, collect_wall_s=collect_time,
                optimize_wall_s=time.monotonic() - optimized_at,
                elapsed_wall_s=time.monotonic() - started)
            log.write(json.dumps(entry) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in entry.items() if k != "episodes"}), flush=True)
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                save_checkpoint(args.output / f"reinforce_{state['update']:06d}.pt", model, optimizer, args, state.copy())
                last_checkpoint = time.monotonic()
            if not losses["optimizer_steps"]:
                break
    finally:
        save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
        save_checkpoint(args.output / f"reinforce_{state['update']:06d}.pt", model, optimizer, args, state.copy())
        log.close()
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

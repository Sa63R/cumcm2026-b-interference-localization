"""Real BC warm-start + on-policy PPO, resumable and bounded by wall clock.

No official simulator is connected. Training reward is negative virtual time,
with a public-budget failure penalty; all automatic fallback time is charged.
Gamma=1 is fixed because macro-actions have different virtual durations and the
objective is undiscounted completion time, not a per-decision discounted proxy.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.distributions import Categorical

from simulation import LocalResearchSimulator, random_scenario
from .controller import ALGORITHM_VERSION, ALGORITHM_VERSIONS, FEATURE_DIMS, DeepRLSearch, feature_schema
from .network import CandidateActorCritic, TorchPolicy, pack_observations


_worker_model = None


def legal_training_seed(seed):
    return 100000 <= seed < 200000 or 2000 <= seed <= 5099


def episode(task):
    """Worker: CPU inference/environment; parent batches optimizer work on GPU."""
    global _worker_model
    seed, weights, hidden, action_seed, teacher, max_decisions = task[:6]
    deadline = task[6] if len(task) > 6 else None
    version = task[7] if len(task) > 7 else "v2"
    if not legal_training_seed(seed):
        raise ValueError("Training scenario seed is outside the declared training ranges")
    if deadline is not None and time.time() >= deadline:
        return [], dict(seed=seed, deadline_skipped=True)
    torch.set_num_threads(1)
    if (_worker_model is None or _worker_model.hidden != hidden
            or _worker_model.feature_dim != FEATURE_DIMS[version]):
        _worker_model = CandidateActorCritic(hidden, FEATURE_DIMS[version])
    _worker_model.load_state_dict(weights)
    _worker_model.eval()
    # Model construction consumes Torch RNG. Reset after constructing/loading,
    # so a first task in a fresh worker matches the same task in a reused worker.
    torch.manual_seed(action_seed)
    np.random.seed(action_seed % 2**32)
    random.seed(action_seed)
    records = []

    def record(features, context, action, target, selection, cost):
        records.append(dict(features=np.asarray(features, dtype=np.float32),
                            context=np.asarray(context, dtype=np.float32),
                            action=action, teacher=target, log_prob=selection[1],
                            value=selection[2], reward=-cost / 1000.0))

    simulator = LocalResearchSimulator(random_scenario(3, seed), max_real_duration_s=300)
    if version == "v3":
        from .joint_scan import JointScanRLSearch
        controller_class = JointScanRLSearch
    else:
        controller_class = DeepRLSearch
    controller = controller_class(simulator.client(), TorchPolicy(
        _worker_model, deterministic=False, teacher=teacher), recorder=record,
        max_decisions=max_decisions, action_deadline_epoch=deadline, feature_version=version)
    started = time.perf_counter()
    report = controller.run()
    # The evaluator may inspect success only after exit; it is not in features.
    evaluation = simulator.evaluation()
    if report.completion_reason == "training_deadline":
        # Administrative training interruption is not an environment failure
        # and must not bias on-policy returns toward the just-running scenes.
        return [], dict(seed=seed, deadline_skipped=True,
                        virtual_time_s=report.virtual_time_s,
                        completion_reason=report.completion_reason)
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model)
    accounted = -sum(r["reward"] for r in records) * 1000
    fixed_initial = report.learning["initial_scan_virtual_time_s"]
    extra_cost = max(0.0, report.virtual_time_s - fixed_initial - accounted)
    if records:
        # Decision-limit baseline tail is not a policy action but its full cost
        # is attributed to the final chosen action, preventing cheap early exit.
        records[-1]["reward"] -= extra_cost / 1000
        if not success:
            records[-1]["reward"] -= 360.0
    metrics = dict(seed=seed, success=success, virtual_time_s=report.virtual_time_s,
                   failure_penalty_s=0 if success else 360000,
                   failed_clear_count=evaluation["failed_clear_count"],
                   wall_time_s=time.perf_counter() - started,
                   records=len(records), learning=report.learning,
                   reward_cost_s=-sum(r["reward"] for r in records) * 1000,
                   initial_scan_virtual_time_s=fixed_initial,
                   completion_reason=report.completion_reason)
    return records, metrics


def compute_returns(records, gae_lambda=1.0):
    """Undiscounted terminal GAE; lambda=1 gives Monte Carlo policy returns."""
    advantage = 0.0
    next_value = 0.0
    for record in reversed(records):
        delta = record["reward"] + next_value - record["value"]
        advantage = delta + gae_lambda * advantage
        record["advantage"] = advantage
        record["return"] = advantage + record["value"]
        next_value = record["value"]
    return records


def update(model, optimizer, records, args, *, bc=False, stop_at=None):
    if not records:
        raise RuntimeError("No policy transitions collected")
    device = next(model.parameters()).device
    metrics = []
    def summarize(**extra):
        means = ({key: float(np.mean([r[key] for r in metrics])) for key in metrics[0]}
                 if metrics else {})
        return dict(means, optimizer_steps=len(metrics), **extra)
    advantages = np.asarray([r.get("advantage", 0.0) for r in records], dtype=np.float32)
    if not bc:
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
    for _ in range(args.bc_epochs if bc else args.epochs):
        ordering = np.random.permutation(len(records))
        for offset in range(0, len(ordering), args.minibatch):
            if stop_at is not None and time.monotonic() >= stop_at:
                return summarize(deadline_before_optimizer_step=not bool(metrics), deadline_reached=True)
            indices = ordering[offset:offset + args.minibatch]
            batch = [records[i] for i in indices]
            logits, values = model(*pack_observations(batch, device))
            distribution = Categorical(logits=logits)
            target = torch.tensor([r["teacher"] for r in batch], device=device)
            bc_loss = torch.nn.functional.cross_entropy(logits, target)
            if bc:
                returns = torch.tensor([r["return"] for r in batch], device=device)
                loss = bc_loss + args.value_coef * torch.nn.functional.mse_loss(values, returns)
                approx_kl = 0.0
            else:
                actions = torch.tensor([r["action"] for r in batch], device=device)
                old_log_prob = torch.tensor([r["log_prob"] for r in batch], device=device)
                returns = torch.tensor([r["return"] for r in batch], device=device)
                advantage = torch.as_tensor(advantages[indices], device=device)
                log_ratio = distribution.log_prob(actions) - old_log_prob
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(ratio * advantage,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * advantage).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                loss = (policy_loss + args.value_coef * value_loss
                        - args.entropy_coef * distribution.entropy().mean()
                        + args.aux_bc_coef * bc_loss)
                approx_kl = float(((ratio - 1) - log_ratio).mean().detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics.append(dict(loss=float(loss.detach()), bc_loss=float(bc_loss.detach()),
                                entropy=float(distribution.entropy().mean().detach()),
                                grad_norm=float(gradient), approx_kl=approx_kl))
            if not bc and args.target_kl > 0 and approx_kl > args.target_kl:
                return summarize(kl_stopped=True)
    return summarize()


def git_version():
    provided = os.environ.get("Q3_SOURCE_COMMIT")
    if provided:
        return provided
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


@lru_cache(maxsize=1)
def source_manifest():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py"))
    files += [root / "pyproject.toml", root / "research" / "v1_protocol.json"]
    entries = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files if path.is_file()}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=entries)


def save_checkpoint(path, model, optimizer, args, state):
    payload = dict(algorithm=ALGORITHM_VERSIONS[args.feature_version], hidden=args.hidden,
                   feature_version=args.feature_version, feature_schema=feature_schema(args.feature_version),
                   model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(), args=vars(args), state=state,
                   git_commit=git_version(), saved_utc=datetime.now(timezone.utc).isoformat(),
                   source_manifest=source_manifest(),
                   torch_version=torch.__version__, torch_rng=torch.get_rng_state(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate())
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_resume(payload, args):
    if (payload.get("algorithm") != ALGORITHM_VERSIONS[args.feature_version]
            or payload.get("hidden") != args.hidden):
        raise ValueError("checkpoint architecture/version does not match; use --initialize-from for explicit migration")
    if payload.get("feature_schema") != feature_schema(args.feature_version):
        raise ValueError("checkpoint feature semantics do not match")
    if payload.get("source_manifest") != source_manifest():
        raise ValueError("checkpoint source manifest differs; use --initialize-from for a separately logged new trial")


def initialize_from(model, payload):
    """Explicit fresh-trial transfer; new feature columns initially have zero weight.

    Preserves old logits before further learning if the v1 prefix and all other
    features/actions remain unchanged. Optimizer/RNG/state are intentionally new.
    """
    if payload.get("algorithm") not in ALGORITHM_VERSIONS.values():
        raise ValueError("unsupported transfer algorithm")
    previous_version = next(v for v, a in ALGORITHM_VERSIONS.items() if a == payload["algorithm"])
    target_version = next(v for v, dim in FEATURE_DIMS.items() if dim == model.feature_dim)
    if (feature_schema(previous_version)["action_semantics"]
            != feature_schema(target_version)["action_semantics"]):
        raise ValueError("transfer across different action semantics is prohibited; train a new BC/PPO policy")
    if ((previous_version != "v1" or "feature_schema" in payload)
            and payload.get("feature_schema") != feature_schema(previous_version)):
        raise ValueError("transfer checkpoint has unknown feature semantics")
    weights = dict(payload["model"])
    previous = weights["encoder.0.weight"]
    if previous.shape[0] != model.hidden or previous.shape[1] > model.feature_dim:
        raise ValueError("transfer requires equal hidden width and a nonshrinking feature prefix")
    expanded = torch.zeros_like(model.encoder[0].weight)
    expanded[:, :previous.shape[1]] = previous
    weights["encoder.0.weight"] = expanded
    model.load_state_dict(weights)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path, help="Explicit new-trial weight transfer; fresh optimizer/counters")
    parser.add_argument("--feature-version", choices=sorted(FEATURE_DIMS), default="v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--seed", type=int, default=9112026)
    parser.add_argument("--scenario-start", type=int, default=100001)
    parser.add_argument("--bc-episodes", type=int, default=128)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--aux-bc-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--max-wall-s", type=float, default=1800)
    parser.add_argument("--checkpoint-seconds", type=float, default=1200)
    parser.add_argument("--deadline-utc", help="ISO UTC hard deadline, e.g. 2026-09-11T06:00:00+00:00")
    args = parser.parse_args(argv)
    if args.workers < 0 or args.num_threads < 1 or args.episodes_per_update < 1 or args.max_decisions < 1:
        parser.error("workers must be >= 0; threads and episodes must be positive")
    if not legal_training_seed(args.scenario_start):
        parser.error("scenario-start must be in training-only ranges")
    if not 0 <= args.gae_lambda <= 1 or args.max_wall_s <= 0:
        parser.error("lambda must be in [0,1], wall time positive")
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("output directory is nonempty; choose a new trial directory or use --resume")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    model = CandidateActorCritic(args.hidden, FEATURE_DIMS[args.feature_version]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = dict(update=0, optimizer_steps=0, episodes=0, next_seed=args.scenario_start, bc_complete=False)
    if args.resume:
        payload = torch.load(args.resume, map_location=args.device, weights_only=False)
        try:
            validate_resume(payload, args)
        except ValueError as error:
            parser.error(str(error))
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
        state["bc_complete"] = True
        state["initialization"] = dict(path=str(args.initialize_from),
            sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
            source_algorithm=payload["algorithm"], source_manifest=payload.get("source_manifest"),
            source_state=payload.get("state"), source_git_commit=payload.get("git_commit"))
        save_checkpoint(args.output / "initialized.pt", model, optimizer, args, state.copy())
    started = time.monotonic()
    stop_at = started + args.max_wall_s
    if args.deadline_utc:
        deadline = datetime.fromisoformat(args.deadline_utc)
        if deadline.tzinfo is None:
            parser.error("deadline-utc must have an explicit timezone")
        stop_at = min(stop_at, started + deadline.timestamp() - time.time())
    config = {**vars(args), "algorithm": ALGORITHM_VERSIONS[args.feature_version], "git_commit": git_version(),
              "feature_schema": feature_schema(args.feature_version),
              "source_manifest": source_manifest(),
              "gamma": 1.0, "reward_scale_s": 1000, "failure_penalty_s": 360000,
              "training_seed_ranges": [[100000, 199999], [2000, 5099]],
              "validation_seeds_not_used_by_training": [6000, 6047]}
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    executor = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) if args.workers else None
    last_checkpoint = started
    log = (args.output / "training.jsonl").open("a", encoding="utf-8")

    def collect(count, teacher):
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        tasks = []
        for _ in range(count):
            seed = state["next_seed"]
            if not legal_training_seed(seed):
                seed = 100001
            state["next_seed"] = seed + 1
            epoch_deadline = time.time() + max(0.0, stop_at - time.monotonic())
            tasks.append((seed, weights, args.hidden, random.randrange(2**31), teacher,
                          args.max_decisions, epoch_deadline, args.feature_version))
        # Workers check the shared absolute deadline before every physical
        # action; already queued tasks skip instead of overrunning the cutoff.
        results = list(executor.map(episode, tasks)) if executor else [episode(t) for t in tasks]
        state["episodes"] += len(results)
        records = [r for trajectory, _ in results for r in compute_returns(trajectory, args.gae_lambda)]
        return records, [metrics for _, metrics in results]

    try:
        if not state["bc_complete"] and args.bc_episodes > 0 and time.monotonic() < stop_at:
            collect_started = time.monotonic()
            records, episodes = collect(args.bc_episodes, True)
            collected = time.monotonic()
            losses = update(model, optimizer, records, args, bc=True, stop_at=stop_at) if records else {}
            state["optimizer_steps"] += losses.get("optimizer_steps", 0)
            state["bc_complete"] = bool(records) and time.monotonic() < stop_at
            sha = save_checkpoint(args.output / "bc_only.pt", model, optimizer, args, state.copy())
            entry = dict(stage="bc", update=0, episodes=episodes, losses=losses,
                         collect_wall_s=collected - collect_started,
                         optimize_wall_s=time.monotonic() - collected, checkpoint_sha256=sha)
            log.write(json.dumps(entry) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in entry.items() if k != "episodes"}), flush=True)
        while state["update"] < args.updates and time.monotonic() < stop_at:
            collect_started = time.monotonic()
            records, episodes = collect(args.episodes_per_update, False)
            collected = time.monotonic()
            if not records or time.monotonic() >= stop_at:
                log.write(json.dumps(dict(stage="deadline", episodes=episodes)) + "\n")
                log.flush()
                break
            losses = update(model, optimizer, records, args, stop_at=stop_at)
            state["optimizer_steps"] += losses.get("optimizer_steps", 0)
            if losses.get("optimizer_steps", 0) == 0:
                log.write(json.dumps(dict(stage="deadline", episodes=episodes, losses=losses)) + "\n")
                log.flush()
                break
            state["update"] += 1
            entry = dict(stage="ppo", update=state["update"], total_episodes=state["episodes"],
                         mean_virtual_time_s=float(np.mean([e["virtual_time_s"] for e in episodes if not e.get("deadline_skipped")])),
                         success_count=sum(e.get("success", False) for e in episodes), episodes=episodes,
                         losses=losses, collect_wall_s=collected - collect_started,
                         optimize_wall_s=time.monotonic() - collected,
                         elapsed_wall_s=time.monotonic() - started)
            log.write(json.dumps(entry) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in entry.items() if k != "episodes"}), flush=True)
            # latest is always restartable; periodic immutable snapshots permit
            # BC/PPO comparison and validation selection without test leakage.
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                save_checkpoint(args.output / f"ppo_{state['update']:06d}.pt", model, optimizer, args, state.copy())
                last_checkpoint = time.monotonic()
    finally:
        save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
        save_checkpoint(args.output / f"ppo_{state['update']:06d}.pt", model, optimizer, args, state.copy())
        log.close()
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

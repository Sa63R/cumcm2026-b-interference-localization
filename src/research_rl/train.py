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

from .cpu_runtime import require_cpu

import numpy as np
import torch
from torch.distributions import Categorical

from simulation import LocalResearchSimulator, random_scenario
from .controller import ALGORITHM_VERSION, ALGORITHM_VERSIONS, FEATURE_DIMS, DeepRLSearch, feature_schema
from .network import (CandidateActorCritic, TorchPolicy, pack_observations,
                      architecture_from_args, checkpoint_architecture, validate_architecture)
from .distributions import (checkpoint_distribution, distribution_from_args,
                            validate_distribution, merge_probe_diagnostics)
from .action_sets import (ACTION_SCHEMA_NAMES, action_schema_from_args, checkpoint_action_schema,
                          validate_action_schema, controller_for)
from .portable_checkpoint import portable_paths


_worker_model = None
TRAINING_SEED_RANGES = ((2000, 5099), (100001, 199999), (1000001, 1999999), (3000001, 3999999))


def legal_training_seed(seed):
    return any(first <= seed <= last for first, last in TRAINING_SEED_RANGES)


def restore_sampling_budget(state, args, previous_args=None):
    """Preserve reserved attempts, including batches interrupted before results."""
    previous_args = previous_args or {}
    if previous_args.get("scenario_start", args.scenario_start) != args.scenario_start:
        raise ValueError("resume cannot change scenario-start")
    for name in ("scenario_end", "max_attempted_episodes"):
        old = previous_args.get(name)
        current = getattr(args, name)
        if old is not None:
            if current is None:
                setattr(args, name, old)
            elif current > old:
                raise ValueError(f"resume cannot enlarge {name.replace('_', '-')}")
    cursor = state["next_seed"]
    # Old checkpoints advanced the cursor before sampling but counted episodes
    # only after the entire batch returned. Recover that in-flight reservation.
    state.setdefault("attempted_episodes", max(state.get("episodes", 0), cursor - args.scenario_start))
    attempted = state["attempted_episodes"]
    if type(attempted) is not int or attempted < state.get("episodes", 0) or attempted < 0:
        raise ValueError("checkpoint has an invalid attempted episode count")
    if args.scenario_end is not None:
        if cursor != args.scenario_start + attempted or not args.scenario_start <= cursor <= args.scenario_end + 1:
            raise ValueError("checkpoint cursor is outside its monotonic scenario partition")
    if args.max_attempted_episodes is not None and attempted > args.max_attempted_episodes:
        raise ValueError("checkpoint has already exceeded max-attempted-episodes")
    state["stop_reason"] = None


def sampling_stop_reason(state, args):
    if args.max_attempted_episodes is not None and state["attempted_episodes"] >= args.max_attempted_episodes:
        return "attempted_episode_limit"
    if args.scenario_end is not None and state["next_seed"] > args.scenario_end:
        return "scenario_range_exhausted"
    if args.scenario_start >= 1000001 and not legal_training_seed(state["next_seed"]):
        return "scenario_range_exhausted"
    return None


def initial_recovery_checkpoint(output):
    """Recognize only a pristine initialization interrupted before latest.pt."""
    allowed = {"random.pt", "initialized.pt", "random.pt.tmp", "initialized.pt.tmp", "latest.pt.tmp"}
    names = {p.name for p in output.iterdir()}
    if not names <= allowed:
        return None
    for name in ("initialized.pt", "random.pt"):
        if (output / name).is_file():
            return output / name
    return None


def episode(task):
    """Worker and learner both run on CPU; each worker uses one Torch thread."""
    require_cpu()
    global _worker_model
    seed, weights, hidden, action_seed, teacher, max_decisions = task[:6]
    deadline = task[6] if len(task) > 6 else None
    version = task[7] if len(task) > 7 else "v2"
    architecture = validate_architecture(task[8] if len(task) > 8 else None)
    action_distribution = validate_distribution(task[9] if len(task) > 9 else None)
    candidate_schema = validate_action_schema(task[10] if len(task) > 10 else None)
    if not legal_training_seed(seed):
        raise ValueError("Training scenario seed is outside the declared training ranges")
    if deadline is not None and time.time() >= deadline:
        return [], dict(seed=seed, deadline_skipped=True)
    torch.set_num_threads(1)
    if (_worker_model is None or _worker_model.hidden != hidden
            or _worker_model.feature_dim != FEATURE_DIMS[version]
            or _worker_model.architecture != architecture
            or _worker_model.action_distribution != action_distribution
            or _worker_model.action_schema != candidate_schema):
        _worker_model = CandidateActorCritic(hidden, FEATURE_DIMS[version], architecture,
                                             action_distribution, candidate_schema)
    _worker_model.load_state_dict(weights)
    _worker_model.eval()
    # Model construction consumes Torch RNG. Reset after constructing/loading,
    # so a first task in a fresh worker matches the same task in a reused worker.
    torch.manual_seed(action_seed)
    np.random.seed(action_seed % 2**32)
    random.seed(action_seed)
    records = []
    diagnostic_records = []

    def record(features, context, action, target, selection, cost):
        records.append(dict(features=np.asarray(features, dtype=np.float32),
                            context=np.asarray(context, dtype=np.float32),
                            action=action, teacher=target, log_prob=selection[1],
                            value=selection[2], reward=-cost / 1000.0))
        if policy.last_probe_diagnostics is not None:
            diagnostic_records.append(policy.last_probe_diagnostics)

    simulator = LocalResearchSimulator(random_scenario(3, seed), max_real_duration_s=300)
    controller_class = controller_for(version, candidate_schema)
    policy = TorchPolicy(_worker_model, deterministic=False, teacher=teacher,
                         capture_diagnostics=version in {"v3", "v4"})
    controller = controller_class(simulator.client(), policy, recorder=record,
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
    if version in {"v3", "v4"}:
        metrics["sampling_probe_diagnostics"] = merge_probe_diagnostics(diagnostic_records)
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
    files += [root / "pyproject.toml", root / "research" / "v1_protocol.json",
              root / "research" / "cpu_v2_protocol.json"]
    entries = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files if path.is_file()}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=entries)


def save_checkpoint(path, model, optimizer, args, state):
    require_cpu(getattr(args, "device", "cpu"))
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("All model parameters must reside on CPU")
    payload = dict(algorithm=ALGORITHM_VERSIONS[args.feature_version], hidden=args.hidden,
                   architecture=model.architecture,
                   action_distribution=model.action_distribution,
                   action_schema=model.action_schema,
                   feature_version=args.feature_version, feature_schema=feature_schema(args.feature_version),
                   model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(), args=vars(args), state=state,
                   git_commit=git_version(), saved_utc=datetime.now(timezone.utc).isoformat(),
                   source_manifest=source_manifest(),
                   torch_version=torch.__version__, torch_rng=torch.get_rng_state(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate())
    payload["compute_policy"] = "CPU-only; no GPU backend queried or used"
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Concrete PosixPath/WindowsPath objects cannot be instantiated on the
    # other OS. Their provenance text is sufficient; tensors/RNG stay intact.
    torch.save(portable_paths(payload), temporary)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_resume(payload, args):
    if (payload.get("algorithm") != ALGORITHM_VERSIONS[args.feature_version]
            or payload.get("hidden") != args.hidden):
        raise ValueError("checkpoint architecture/version does not match; use --initialize-from for explicit migration")
    if payload.get("feature_schema") != feature_schema(args.feature_version):
        raise ValueError("checkpoint feature semantics do not match")
    if checkpoint_architecture(payload) != architecture_from_args(args):
        raise ValueError("checkpoint architecture differs; use --initialize-from for an explicit compatible expansion")
    if checkpoint_distribution(payload) != distribution_from_args(args):
        raise ValueError("checkpoint action distribution differs; use an explicit new-trial initialization")
    if checkpoint_action_schema(payload) != action_schema_from_args(args):
        raise ValueError("checkpoint action schema differs; use an explicit new-trial initialization")
    if payload.get("source_manifest") != source_manifest():
        raise ValueError("checkpoint source manifest differs; use --initialize-from for a separately logged new trial")


def preserves_initial_probabilities(model, payload):
    """Only established transfers on identical observation/candidate histories.

    Equal distribution/schema names alone do not establish equal controllers.
    In particular, a feature-width change is not automatically a neutral one.
    """
    previous = next((v for v, a in ALGORITHM_VERSIONS.items() if a == payload.get("algorithm")), None)
    target = next(v for v, dim in FEATURE_DIMS.items() if dim == model.feature_dim)
    return ((previous == target or (previous, target) == ("v3", "v4"))
            and checkpoint_distribution(payload) == model.action_distribution
            and checkpoint_action_schema(payload) == model.action_schema)


def initialize_from(model, payload, *, allow_distribution_change=False, allow_action_schema_change=False):
    """Explicit fresh-trial transfer; new feature columns initially have zero weight.

    Preserves old logits before further learning if the v1 prefix, other
    features/actions, and action distribution remain unchanged. Explicit
    distribution migration changes probabilities. Optimizer/RNG/state are new.
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
    if previous_version == "v4" and target_version != "v4":
        raise ValueError("cannot discard trained route-debt features")
    if previous_version != "v4" and any(key.startswith("route_adapter.") for key in weights):
        raise ValueError("route adapter contradicts source feature semantics")
    if checkpoint_action_schema(payload) != model.action_schema and not allow_action_schema_change:
        raise ValueError("action schema change requires an explicit supported new-trial migration")
    if checkpoint_distribution(payload) != model.action_distribution and not allow_distribution_change:
        # Legacy/paired callers must never silently load grouped weights into
        # a flat-policy trainer. PPO's explicit new-trial path opts in below.
        raise ValueError("action distribution change requires an explicit supported new-trial migration")
    previous_arch = checkpoint_architecture(payload)
    target_arch = model.architecture
    if previous_arch["name"] == "attention":
        if (target_arch["name"] != "attention"
                or previous_arch["heads"] != target_arch["heads"]
                or previous_arch["layers"] > target_arch["layers"]):
            raise ValueError("cannot silently discard/change trained attention blocks")
    previous = weights["encoder.0.weight"]
    previous_input_dim = 60 if previous_version == "v4" else FEATURE_DIMS[previous_version]
    if previous.shape[1] != previous_input_dim:
        raise ValueError("transfer input width contradicts checkpoint feature semantics")
    if previous.shape[0] != model.hidden or previous.shape[1] > model.encoder[0].in_features:
        raise ValueError("transfer requires equal hidden width and a nonshrinking feature prefix")
    expanded = torch.zeros_like(model.encoder[0].weight)
    expanded[:, :previous.shape[1]] = previous
    weights["encoder.0.weight"] = expanded
    # Only the old complete network is copied; genuinely new gated branches
    # retain initialization and zero gates. Existing attention gates are copied.
    expanded_state = model.state_dict()
    unexpected = set(weights) - set(expanded_state)
    if unexpected:
        raise ValueError(f"unexpected checkpoint parameters: {sorted(unexpected)}")
    for key in expanded_state:
        existed = (not key.startswith("relations.") or
                   (previous_arch["name"] == "attention"
                    and int(key.split(".")[1]) < previous_arch["layers"]))
        if key.startswith("route_adapter."):
            existed = previous_version == "v4"
        if key not in weights and existed:
            raise ValueError(f"checkpoint is missing existing parameter {key}")
    expanded_state.update(weights)
    model.load_state_dict(expanded_state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path, help="Explicit new-trial weight transfer; fresh optimizer/counters")
    parser.add_argument("--feature-version", choices=sorted(FEATURE_DIMS), default="v2")
    parser.add_argument("--architecture", choices=("mlp", "attention"), default="mlp")
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--group-alpha", type=int, choices=(0, 1), default=0,
                        help="0: original flat policy; 1: subtract log task-group size (v3 only)")
    parser.add_argument("--probe-candidates", choices=ACTION_SCHEMA_NAMES, default="base",
                        help="Explicit action-set identity; extensions require v3 and a new trial")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--seed", type=int, default=9112026)
    parser.add_argument("--scenario-start", type=int, default=100001)
    parser.add_argument("--scenario-end", type=int,
                        help="Inclusive final training seed; never recycle seeds when set")
    parser.add_argument("--max-attempted-episodes", type=int,
                        help="Lifetime sampling budget, including reserved/interrupted batches")
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
    require_cpu(args.device)
    if args.workers < 0 or args.num_threads < 1 or args.episodes_per_update < 1 or args.max_decisions < 1:
        parser.error("workers must be >= 0; threads and episodes must be positive")
    if not legal_training_seed(args.scenario_start):
        parser.error("scenario-start must be in training-only ranges")
    if args.scenario_end is not None and not any(
            first <= args.scenario_start <= args.scenario_end <= last for first, last in TRAINING_SEED_RANGES):
        parser.error("scenario-start/end must form one contiguous training-only range")
    if args.max_attempted_episodes is not None and args.max_attempted_episodes < 0:
        parser.error("max-attempted-episodes must be nonnegative")
    if not 0 <= args.gae_lambda <= 1 or args.max_wall_s <= 0:
        parser.error("lambda must be in [0,1], wall time positive")
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")
    recovery = None
    resume_path = args.resume
    if args.output.exists() and any(args.output.iterdir()):
        if not resume_path or (not resume_path.exists() and resume_path.resolve() == (args.output / "latest.pt").resolve()):
            recovery = initial_recovery_checkpoint(args.output)
            if recovery is None:
                parser.error("output directory is nonempty; choose a new trial directory or use a valid --resume")
            resume_path = recovery
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    model = CandidateActorCritic(args.hidden, FEATURE_DIMS[args.feature_version],
                                 architecture_from_args(args), distribution_from_args(args),
                                 action_schema_from_args(args)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = dict(update=0, optimizer_steps=0, episodes=0, attempted_episodes=0,
                 next_seed=args.scenario_start, bc_complete=False, stop_reason=None)
    if resume_path:
        try:
            payload = torch.load(resume_path, map_location=args.device, weights_only=False)
            validate_resume(payload, args)
            state = dict(payload["state"])
            restore_sampling_budget(state, args, payload.get("args"))
            if recovery and any(state.get(key, 0) for key in ("update", "optimizer_steps", "episodes", "attempted_episodes")):
                raise ValueError("initialization recovery checkpoint already contains training")
            if recovery and payload.get("args", {}).get("seed", args.seed) != args.seed:
                raise ValueError("initialization recovery cannot change its random seed")
            if recovery and recovery.name == "random.pt":
                pending = payload.get("args", {}).get("initialize_from")
                if pending:
                    if args.initialize_from and Path(pending).resolve() != args.initialize_from.resolve():
                        raise ValueError("initialization recovery cannot change its parent checkpoint")
                    args.initialize_from = Path(pending)
                elif args.initialize_from:
                    raise ValueError("initialization recovery cannot add a parent checkpoint")
            if recovery and recovery.name == "initialized.pt":
                if "initialization" not in state:
                    raise ValueError("initialized checkpoint has no transfer provenance")
                if (args.initialize_from and hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
                        != state["initialization"].get("sha256")):
                    raise ValueError("initialization recovery cannot change its parent checkpoint")
                args.initialize_from = None  # Its initialized weights are already committed.
        except Exception as error:
            parser.error(str(error))
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        torch.set_rng_state(payload["torch_rng"].cpu())
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    args.output.mkdir(parents=True, exist_ok=True)
    if not resume_path:
        save_checkpoint(args.output / "random.pt", model, optimizer, args, state.copy())
    if args.initialize_from:
        payload = torch.load(args.initialize_from, map_location=args.device, weights_only=False)
        initialize_from(model, payload, allow_distribution_change=True, allow_action_schema_change=True)
        state["bc_complete"] = True
        state["initialization"] = dict(path=str(args.initialize_from),
            sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
            source_algorithm=payload["algorithm"], source_manifest=payload.get("source_manifest"),
            source_architecture=checkpoint_architecture(payload), target_architecture=model.architecture,
            source_distribution=checkpoint_distribution(payload), target_distribution=model.action_distribution,
            source_action_schema=checkpoint_action_schema(payload), target_action_schema=model.action_schema,
            preserves_initial_probabilities=preserves_initial_probabilities(model, payload),
            probability_preservation_scope="identical legal observation/candidate histories; no guarantee after learning",
            source_state=payload.get("state"), source_git_commit=payload.get("git_commit"))
        save_checkpoint(args.output / "initialized.pt", model, optimizer, args, state.copy())
    # Establish a restart point before config/log/executor setup, even when no
    # episode fits the wall-clock budget or the process is killed immediately.
    save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
    started = time.monotonic()
    elapsed_before = state.get("elapsed_training_s", 0.0)
    stop_at = started + args.max_wall_s
    if args.deadline_utc:
        deadline = datetime.fromisoformat(args.deadline_utc)
        if deadline.tzinfo is None:
            parser.error("deadline-utc must have an explicit timezone")
        stop_at = min(stop_at, started + deadline.timestamp() - time.time())
    config = {**vars(args), "algorithm": ALGORITHM_VERSIONS[args.feature_version], "git_commit": git_version(),
              "architecture_spec": model.architecture,
              "action_distribution": model.action_distribution,
              "action_schema": model.action_schema,
              "feature_schema": feature_schema(args.feature_version),
              "source_manifest": source_manifest(),
              "gamma": 1.0, "reward_scale_s": 1000, "failure_penalty_s": 360000,
              "training_seed_ranges": [list(bounds) for bounds in TRAINING_SEED_RANGES],
              "validation_seeds_not_used_by_training": [6000, 6047]}
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    executor = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) if args.workers else None
    last_checkpoint = started
    log = (args.output / "training.jsonl").open("a", encoding="utf-8")

    def time_stop_reason():
        if time.monotonic() < stop_at:
            return None
        return "global_deadline" if args.deadline_utc and deadline.timestamp() <= time.time() else "wall_time_limit"

    def final_stop_reason():
        return (sampling_stop_reason(state, args) or time_stop_reason()
                or ("update_limit" if state["update"] >= args.updates else "no_training_records"))

    def collect(count, teacher):
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        tasks = []
        for _ in range(count):
            if sampling_stop_reason(state, args):
                break
            seed = state["next_seed"]
            if not legal_training_seed(seed):
                seed = 100001
            state["next_seed"] = seed + 1
            state["attempted_episodes"] += 1
            epoch_deadline = time.time() + max(0.0, stop_at - time.monotonic())
            tasks.append((seed, weights, args.hidden, random.randrange(2**31), teacher,
                          args.max_decisions, epoch_deadline, args.feature_version, model.architecture,
                          model.action_distribution, model.action_schema))
        if not tasks:
            return [], []
        # Reserve before dispatch: even an uncatchable process kill cannot
        # reuse queued seeds or obtain extra budget by restarting an update.
        # A killed batch consumes its reservation, including unstarted tasks.
        state["elapsed_training_s"] = elapsed_before + time.monotonic() - started
        save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
        # Workers check the shared absolute deadline before every physical
        # action; already queued tasks skip instead of overrunning the cutoff.
        results = list(executor.map(episode, tasks)) if executor else [episode(t) for t in tasks]
        state["episodes"] += len(results)
        records = [r for trajectory, _ in results for r in compute_returns(trajectory, args.gae_lambda)]
        return records, [metrics for _, metrics in results]

    try:
        if (not state["bc_complete"] and args.bc_episodes > 0 and time.monotonic() < stop_at
                and not sampling_stop_reason(state, args)):
            collect_started = time.monotonic()
            records, episodes = collect(args.bc_episodes, True)
            collected = time.monotonic()
            losses = update(model, optimizer, records, args, bc=True, stop_at=stop_at) if records else {}
            state["optimizer_steps"] += losses.get("optimizer_steps", 0)
            state["bc_complete"] = bool(records) and time.monotonic() < stop_at
            sha = save_checkpoint(args.output / "bc_only.pt", model, optimizer, args, state.copy())
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
            entry = dict(stage="bc", update=0, episodes=episodes, losses=losses,
                         sampling_probe_diagnostics=merge_probe_diagnostics(
                             [e.get("sampling_probe_diagnostics", {}) for e in episodes]),
                         collect_wall_s=collected - collect_started,
                         optimize_wall_s=time.monotonic() - collected, checkpoint_sha256=sha)
            log.write(json.dumps(entry) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in entry.items() if k != "episodes"}), flush=True)
        while (state["update"] < args.updates and time.monotonic() < stop_at
               and not sampling_stop_reason(state, args)):
            collect_started = time.monotonic()
            records, episodes = collect(args.episodes_per_update, False)
            collected = time.monotonic()
            if not records or time.monotonic() >= stop_at:
                state["stop_reason"] = time_stop_reason() or "no_training_records"
                log.write(json.dumps(dict(stage="deadline", stop_reason=state["stop_reason"], episodes=episodes)) + "\n")
                log.flush()
                break
            losses = update(model, optimizer, records, args, stop_at=stop_at)
            state["optimizer_steps"] += losses.get("optimizer_steps", 0)
            if losses.get("optimizer_steps", 0) == 0:
                state["stop_reason"] = time_stop_reason() or "no_training_records"
                log.write(json.dumps(dict(stage="deadline", stop_reason=state["stop_reason"], episodes=episodes, losses=losses)) + "\n")
                log.flush()
                break
            state["update"] += 1
            state["elapsed_training_s"] = elapsed_before + time.monotonic() - started
            entry = dict(stage="ppo", update=state["update"], total_episodes=state["episodes"],
                         attempted_episodes=state["attempted_episodes"], next_seed=state["next_seed"],
                         sampling_probe_diagnostics=merge_probe_diagnostics(
                             [e.get("sampling_probe_diagnostics", {}) for e in episodes]),
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
    except BaseException:
        state["stop_reason"] = "interrupted"
        raise
    finally:
        state["stop_reason"] = state.get("stop_reason") or final_stop_reason()
        state["elapsed_training_s"] = elapsed_before + time.monotonic() - started
        save_checkpoint(args.output / "latest.pt", model, optimizer, args, state.copy())
        save_checkpoint(args.output / f"ppo_{state['update']:06d}.pt", model, optimizer, args, state.copy())
        log.close()
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

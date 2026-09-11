"""CPU policy improvement from full continuations at matched public prefixes.

One group is one new training world: original greedy action and up to two
random legal alternatives, followed by the same frozen deterministic actor.
Only complete groups supply labels. Critic parameters are frozen; actor and
shared representation use weighted pairwise cost ranking plus full-action KL.
The inference checkpoint remains the original v3/base/flat MLP schema.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import time

from .cpu_runtime import require_cpu

import numpy as np
import torch
from torch.nn import functional as F

from .controller import ALGORITHM_VERSIONS, FEATURE_DIMS, feature_schema
from .network import CandidateActorCritic, pack_observations, checkpoint_architecture
from .distributions import checkpoint_distribution
from .action_sets import checkpoint_action_schema
from .portable_checkpoint import portable_paths, model_tensor_digest
from .train import git_version, legal_training_seed, initialize_from, source_manifest, restore_sampling_budget

TRAINING_ALGORITHM = "q3-prefix-full-continuation-pairwise-kl-v1"
SCHEMA = {"version": 1, "name": "base"}


def collect_group(task):
    # Lazy import allows --updates 0 and checkpoint validation without sampling.
    from .prefix_rollouts import collect_group as worker
    return worker(task)


def objective_config(args):
    return dict(algorithm=TRAINING_ALGORITHM, groups_per_update=args.groups_per_update,
        alternatives=args.alternatives, lr=args.lr, epochs=args.epochs, minibatch=args.minibatch,
        gap_scale_s=args.gap_scale_s, max_pair_weight=args.max_pair_weight, kl_coef=args.kl_coef,
        max_grad_norm=args.max_grad_norm, max_decisions=args.max_decisions,
        pair_normalization="mean all unordered evaluated-candidate pairs, then mean groups",
        cost="complete remaining physical cost plus collector-declared safety failure penalty",
        old_policy="the frozen actor that collected this entire batch",
        kl_direction="old_to_new over all legal candidates, including unevaluated candidates",
        critic="head frozen; no value loss; shared encoder/context trained by actor objective",
        failure_penalty_s=360000, ties="zero pair weight; no arbitrary better-action tie break")


def public_args(args):
    return {k: (v.name if isinstance(v, Path) else v) for k,v in vars(args).items()}


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_small_evidence(output, results, actor_hash, state):
    """First two reserved worlds only; never embed full histories in the log."""
    for group, metrics in results:
        evidence = metrics.pop("evidence", None)
        if not evidence:
            continue
        directory = output / "evidence"
        directory.mkdir(exist_ok=True)
        path = directory / f"group-{metrics['seed']}.json.gz"
        if path.exists():
            raise ValueError("Reserved evidence filename already exists; refuse overwrite")
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"Unsupported evidence value: {type(value).__name__}")
        temp = path.with_suffix(path.suffix+".tmp")
        with gzip.open(temp,"wt",encoding="utf-8") as stream:
            json.dump(dict(group=group,trajectory_evidence=evidence,collection_metrics=metrics,
                actor_tensor_sha256=actor_hash,scope="training-only exact prefix/continuation audit"),
                stream,default=convert,ensure_ascii=False,allow_nan=False,separators=(",",":"))
        temp.replace(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(output).as_posix()
        state["evidence_files"][relative] = checksum
        metrics["evidence_file"] = relative
        metrics["evidence_sha256"] = checksum


def freeze_critic(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("critic."))
    return [p for p in model.parameters() if p.requires_grad]


def save_checkpoint(path, model, optimizer, args, state):
    require_cpu(args.device)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("All parameters must remain on CPU")
    payload = dict(algorithm=ALGORITHM_VERSIONS["v3"], training_algorithm=TRAINING_ALGORITHM,
        feature_version="v3", feature_schema=feature_schema("v3"), hidden=args.hidden,
        architecture=model.architecture, action_distribution=model.action_distribution,
        action_schema=model.action_schema,
        model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), state=state, training_state=dict(state),
        args=public_args(args), objective=objective_config(args), source_manifest=source_manifest(),
        git_commit=git_version(), saved_utc=datetime.now(timezone.utc).isoformat(),
        torch_version=torch.__version__, torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        compute_policy="CPU only; world groups and full rollout trajectories counted separately")
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(portable_paths(payload), temp)
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_resume(payload, args):
    if payload.get("training_algorithm") != TRAINING_ALGORITHM:
        raise ValueError("Resume requires this trainer; use --initialize-from for a new trial")
    if payload.get("hidden") != args.hidden or payload.get("feature_schema") != feature_schema("v3"):
        raise ValueError("Checkpoint network semantics differ")
    if (checkpoint_architecture(payload)["name"] != "mlp"
            or checkpoint_distribution(payload)["name"] != "flat"
            or checkpoint_action_schema(payload) != SCHEMA):
        raise ValueError("Only the unchanged v3/base/flat MLP is supported")
    if payload.get("source_manifest") != source_manifest():
        raise ValueError("Checkpoint source differs; resume cannot mix implementations")
    if payload.get("objective") != objective_config(args):
        raise ValueError("Resume cannot change the predeclared objective or update schedule")
    if payload.get("state") != payload.get("training_state"):
        raise ValueError("Independent training state does not match public counters")
    if payload.get("args", {}).get("seed") != args.seed:
        raise ValueError("Resume cannot change the trial seed")


def validate_group(group):
    features = np.asarray(group["features"], dtype=np.float32)
    context = np.asarray(group["context"], dtype=np.float32)
    old = np.asarray(group["old_logits"], dtype=np.float32)
    if (features.ndim != 2 or features.shape[1] != 60 or len(features) < 2
            or context.shape != (12,) or old.shape != (len(features),)
            or not all(np.isfinite(x).all() for x in (features, context, old))):
        raise ValueError("Invalid public prefix features/context/old policy")
    original = group["original_index"]
    if isinstance(original, bool) or not isinstance(original, (int, np.integer)) or original != int(old.argmax()):
        raise ValueError("Original action differs from frozen greedy policy")
    candidates = group["evaluated_candidates"]
    indices = [c["index"] for c in candidates]
    if not 2 <= len(candidates) <= 3 or len(set(indices)) != len(indices) or original not in indices:
        raise ValueError("A complete group requires original plus distinct legal alternatives")
    prefix = float(group["prefix_cost_s"])
    if not math.isfinite(prefix) or prefix < 0:
        raise ValueError("Invalid accepted prefix cost")
    for candidate in candidates:
        index = candidate["index"]
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(features):
            raise ValueError("Evaluated candidate is not legal")
        values = [float(candidate[k]) for k in ("total_time_s", "remaining_cost_s", "cost_to_go_s", "failure_penalty_s")]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Incomplete/nonfinite continuation cannot become a reward")
        total, remaining, penalized, penalty = values
        if (not math.isclose(total, prefix+remaining, abs_tol=3e-6)
                or not math.isclose(penalized, remaining+penalty, abs_tol=3e-6)):
            raise ValueError("Full continuation cost ledger is inconsistent")
    return {**group, "features":features, "context":context, "old_logits":old}


def ranking_pairs(group, args):
    pairs = []
    candidates = group["evaluated_candidates"]
    for i, left in enumerate(candidates):
        for right in candidates[i+1:]:
            difference = float(left["cost_to_go_s"])-float(right["cost_to_go_s"])
            better, worse = (right, left) if difference > 0 else (left, right)
            pairs.append((int(better["index"]), int(worse["index"]),
                min(abs(difference)/args.gap_scale_s, args.max_pair_weight)))
    return pairs


def objective(model, groups, args):
    logits, _ = model(*pack_observations(groups, "cpu"))
    pair_losses, kls = [], []
    for row, group in enumerate(groups):
        new_log_probs = logits[row, :len(group["features"])].log_softmax(-1)
        old_logits = torch.as_tensor(group["old_logits"], dtype=logits.dtype, device=logits.device)
        old_log_probs = old_logits.log_softmax(-1).detach()
        kls.append((old_log_probs.exp()*(old_log_probs-new_log_probs)).sum())
        pairs = ranking_pairs(group, args)
        pair_losses.append(torch.stack([weight*F.softplus(logits[row, worse]-logits[row, better])
            for better,worse,weight in pairs]).mean())
    pair_loss, kl = torch.stack(pair_losses).mean(), torch.stack(kls).mean()
    return pair_loss + args.kl_coef*kl, pair_loss, kl


@torch.no_grad()
def policy_statistics(model, groups, args):
    kl_values, pair_values, regrets = [], [], []
    correct = weight_correct = total_weight = pair_count = evaluated_greedy = 0
    for start in range(0, len(groups), args.minibatch):
        batch = groups[start:start+args.minibatch]
        logits, _ = model(*pack_observations(batch, "cpu"))
        for row, group in enumerate(batch):
            scores = logits[row, :len(group["features"])].cpu()
            old = torch.as_tensor(group["old_logits"]).log_softmax(-1)
            kl_values.append(float((old.exp()*(old-scores.log_softmax(-1))).sum()))
            pairs = ranking_pairs(group, args)
            pair_values.append(sum(weight*float(F.softplus(scores[worse]-scores[better]))
                for better,worse,weight in pairs)/len(pairs))
            for better,worse,weight in pairs:
                if weight > 0:
                    win = float(scores[better] > scores[worse])
                    pair_count += 1; correct += win
                    total_weight += weight; weight_correct += win*weight
            candidates = group["evaluated_candidates"]
            selected = max(candidates, key=lambda c:float(scores[c["index"]]))
            regrets.append(float(selected["cost_to_go_s"])-min(float(c["cost_to_go_s"]) for c in candidates))
            evaluated_greedy += int(int(scores.argmax()) in {c["index"] for c in candidates})
    return dict(full_candidate_kl=float(np.mean(kl_values)), maximum_group_kl=max(kl_values),
        pairwise_loss=float(np.mean(pair_values)), informative_pairs=pair_count,
        pairwise_accuracy=correct/pair_count if pair_count else None,
        weighted_pairwise_accuracy=weight_correct/total_weight if total_weight else None,
        sampled_candidate_regret_s=float(np.mean(regrets)),
        sampled_candidate_regret_scope="only among evaluated actions; not global action value",
        full_greedy_in_evaluated_fraction=evaluated_greedy/len(groups), groups=len(groups))


def update_actor(model, optimizer, groups, args, *, stop_at=None, on_step=None):
    """The reference logits never change within an update, including epochs."""
    groups = [validate_group(g) for g in groups]
    if not groups:
        raise ValueError("No complete groups available")
    before = policy_statistics(model, groups, args)
    if not before["informative_pairs"]:
        return dict(optimizer_steps=0, groups_used_by_optimizer=0,
            skipped_equal_costs=True, before=before, after=before)
    if before["maximum_group_kl"] > 2e-6:
        raise ValueError("Collected old policy differs from the frozen batch actor")
    steps, norms, observed_post_kls, used_groups = 0, [], [], set()
    stop = False
    for _ in range(args.epochs):
        for start in range(0, len(groups), args.minibatch):
            # Each epoch uses one stable permutation. RNG is checkpointed.
            if start == 0:
                order = np.random.permutation(len(groups))
            if stop_at is not None and time.monotonic() >= stop_at:
                stop = True; break
            batch = [groups[i] for i in order[start:start+args.minibatch]]
            loss, _, _ = objective(model, batch, args)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite full-continuation objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            indices_used = set(int(i) for i in order[start:start+args.minibatch])
            new_groups_used = len(indices_used-used_groups)
            used_groups.update(indices_used)
            if on_step is not None:
                on_step(new_groups_used)
            norms.append(float(norm))
            with torch.no_grad():
                observed_post_kls.append(float(objective(model, batch, args)[2]))
        if stop:
            break
    after = policy_statistics(model, groups, args)  # Actual full-batch POST-step KL.
    return dict(optimizer_steps=steps, groups_used_by_optimizer=len(used_groups),
        deadline_reached=stop, before=before, after=after,
        mean_gradient_norm=float(np.mean(norms)) if norms else 0.,
        maximum_observed_post_minibatch_kl=max(observed_post_kls, default=0.),
        mean_observed_post_minibatch_kl=float(np.mean(observed_post_kls)) if observed_post_kls else 0.)


def guarded_local_collect(task):
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        return collect_group(task)
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])


def add_collection_counts(state, results):
    complete_groups = 0
    statuses = {}
    for group, metrics in results:
        status = metrics.get("status", "missing_status")
        statuses[status] = statuses.get(status, 0)+1
        for target, source in (("actual_rollout_trajectories", "trajectories_started"),
                ("completed_rollout_trajectories", "trajectories_completed"),
                ("interrupted_rollout_trajectories", "trajectories_interrupted"),
                ("simulator_actions", "simulator_action_count"),
                ("replay_prefix_actions", "replay_prefix_action_count"),
                ("successful_trajectories", "successful_trajectories"),
                ("failed_trajectories", "failed_trajectories"),
                ("failed_clear_count", "failed_clear_count")):
            state[target] += int(metrics.get(source, 0))
        if group is not None:
            complete_groups += 1
    state["episodes"] += complete_groups
    state["completed_groups"] += complete_groups
    state["returned_groups"] += len(results)
    for status, count in statuses.items():
        state["group_status_counts"][status] = state["group_status_counts"].get(status, 0)+count
    return statuses


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initialize-from", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--seed", type=int, default=9112026)
    p.add_argument("--scenario-start", type=int, default=3100001)
    p.add_argument("--scenario-end", type=int, default=3399999)
    p.add_argument("--max-attempted-episodes", type=int)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--max-wall-s", type=float, default=1800)
    p.add_argument("--deadline-utc")
    p.add_argument("--checkpoint-seconds", type=float, default=600)
    p.add_argument("--groups-per-update", type=int, default=128)
    p.add_argument("--alternatives", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=32)
    p.add_argument("--gap-scale-s", type=float, default=100)
    p.add_argument("--max-pair-weight", type=float, default=2)
    p.add_argument("--kl-coef", type=float, default=1.)
    p.add_argument("--max-grad-norm", type=float, default=.5)
    p.add_argument("--max-decisions", type=int, default=256)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    require_cpu(args.device)
    from .prefix_rollouts import TRAINING_RANGES
    if args.resume and args.initialize_from:
        p.error("--resume and --initialize-from are mutually exclusive")
    if not args.resume and not args.initialize_from:
        p.error("An explicit parent checkpoint or an existing same-trial resume is required")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("Output is nonempty; never overwrite an existing trial")
    if not any(first <= args.scenario_start <= args.scenario_end <= last for first,last in TRAINING_RANGES):
        p.error("Invalid training-only inclusive scenario range")
    if (not 0 <= args.workers <= 60 or not 1 <= args.num_threads <= 60 or args.updates < 0
            or args.hidden < 1 or args.groups_per_update < 1 or args.epochs < 1
            or args.minibatch < 1 or not 1 <= args.max_decisions <= 256 or args.alternatives != 2):
        p.error("Invalid CPU/update/group configuration; this protocol uses exactly two alternatives")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr,args.gap_scale_s,args.max_pair_weight,
            args.max_grad_norm,args.max_wall_s,args.checkpoint_seconds)) or not math.isfinite(args.kl_coef) or args.kl_coef < 0:
        p.error("Invalid objective or time budget")
    if args.max_attempted_episodes is not None and args.max_attempted_episodes < 0:
        p.error("Attempted world-group budget must be nonnegative")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed % 2**32); random.seed(args.seed)
    model = CandidateActorCritic(args.hidden, FEATURE_DIMS["v3"], action_schema=SCHEMA)
    optimizer = torch.optim.Adam(freeze_critic(model), lr=args.lr)
    state = dict(update=0, optimizer_steps=0, episodes=0, completed_groups=0, returned_groups=0,
        attempted_episodes=0, attempted_groups=0, next_seed=args.scenario_start,
        actual_rollout_trajectories=0, completed_rollout_trajectories=0,
        interrupted_rollout_trajectories=0, simulator_actions=0, replay_prefix_actions=0,
        successful_trajectories=0, failed_trajectories=0, failed_clear_count=0,
        learned_groups=0, group_status_counts={}, elapsed_training_s=0., active_batch=None,evidence_files={},
        abandoned_reserved_groups=0, unreported_work_batches=0,
        counters_are_lower_bounds_after_unreported_interruption=False,
        episode_unit="complete eligible world groups; NOT individual simulator trajectories",
        attempt_unit="new world groups reserved before dispatch", stop_reason=None)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(payload, args)
        state = dict(payload["training_state"])
        restore_sampling_budget(state, args, payload["args"])
        if state["attempted_groups"] != state["attempted_episodes"]:
            raise ValueError("Group reservation count disagrees with compatibility counter")
        # A crash after reservation consumes that batch; do not reuse its seeds.
        if state.get("active_batch") is not None:
            if state["active_batch"]["stage"] == "reserved_before_dispatch":
                state["abandoned_reserved_groups"] += state["active_batch"]["count"]
                state["unreported_work_batches"] += 1
                state["counters_are_lower_bounds_after_unreported_interruption"] = True
            state["last_abandoned_batch"] = state.pop("active_batch")
        state["active_batch"] = None
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        np.random.set_state(payload["numpy_rng"]); random.setstate(payload["python_rng"])
    else:
        payload = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        if (payload.get("algorithm") != ALGORITHM_VERSIONS["v3"]
                or checkpoint_architecture(payload)["name"] != "mlp"
                or checkpoint_distribution(payload)["name"] != "flat"
                or checkpoint_action_schema(payload) != SCHEMA):
            raise ValueError("Parent must be the unchanged v3/base/flat MLP")
        initialize_from(model, payload)
        state["initialization"] = dict(checkpoint_sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
            tensor_sha256=model_tensor_digest(payload["model"]), source_algorithm=payload["algorithm"],
            new_optimizer=True, new_counters=True)
    model.eval()  # MLP has no dropout; collection and optimization use same mode.
    critic_digest = model_tensor_digest({k:v for k,v in model.state_dict().items() if k.startswith("critic.")})
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output / "latest.pt", model, optimizer, args, state)
    if not args.resume:
        save_checkpoint(args.output / "initialized.pt", model, optimizer, args, state)
    atomic_json(args.output / "config.json", dict(**public_args(args), objective=objective_config(args),
        training_algorithm=TRAINING_ALGORITHM, source_manifest=source_manifest(),
        compute_policy="CPU only", episode_unit=state["episode_unit"], gamma=1., reward_scale_s=1.))
    started = time.monotonic(); elapsed_before = state["elapsed_training_s"]
    stop_at = started + args.max_wall_s
    if args.deadline_utc:
        deadline = datetime.fromisoformat(args.deadline_utc)
        if deadline.tzinfo is None:
            p.error("deadline-utc requires an explicit timezone")
        stop_at = min(stop_at, started + deadline.timestamp()-time.time())
    last_snapshot = started
    executor = None
    log = (args.output / "training.jsonl").open("a", encoding="utf-8")
    try:
        while state["update"] < args.updates:
            if time.monotonic() >= stop_at:
                state["stop_reason"] = "deadline"; break
            remaining = args.scenario_end-state["next_seed"]+1
            if args.max_attempted_episodes is not None:
                remaining = min(remaining,args.max_attempted_episodes-state["attempted_episodes"])
            count = min(args.groups_per_update,remaining)
            if count <= 0:
                state["stop_reason"] = "attempted_group_or_seed_limit"; break
            if executor is None and args.workers:
                executor = ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context("spawn"))
            weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            snapshot_hash = model_tensor_digest(weights)
            tasks = []
            for _ in range(count):
                seed = state["next_seed"]
                if not legal_training_seed(seed):
                    raise ValueError("No seed recycling or crossing a nontraining gap")
                tasks.append(dict(seed=seed, weights=weights, hidden=args.hidden,
                    action_seed=random.randrange(2**31), deadline_epoch=time.time()+max(0.,stop_at-time.monotonic()),
                    max_decisions=args.max_decisions, alternatives=args.alternatives,
                    capture_evidence=state["attempted_groups"] < 2))
                state["next_seed"] += 1
                state["attempted_episodes"] += 1; state["attempted_groups"] += 1
            state["active_batch"] = dict(first_seed=tasks[0]["seed"],count=len(tasks),actor_tensor_sha256=snapshot_hash,
                optimizer_steps_before=state["optimizer_steps"],stage="reserved_before_dispatch")
            state["elapsed_training_s"] = elapsed_before+time.monotonic()-started
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, state)
            collection_started = time.monotonic()
            results = list(executor.map(collect_group,tasks)) if executor else [guarded_local_collect(t) for t in tasks]
            statuses = add_collection_counts(state,results)
            state["active_batch"]["stage"] = "collected_results"
            save_small_evidence(args.output,results,snapshot_hash,state)
            groups = [validate_group(group) for group,_ in results if group is not None]
            collection_seconds = time.monotonic()-collection_started
            entry = dict(stage="full_continuation_improvement", next_update=state["update"]+1,
                actor_tensor_sha256=snapshot_hash, reserved_groups=len(tasks), complete_groups=len(groups),
                group_status_counts=statuses, collect_wall_s=collection_seconds,
                episode_unit=state["episode_unit"], collection_metrics=[m for _,m in results])
            if statuses.get("invalid_execution"):
                entry["invalid_batch_not_learned"] = True
                log.write(json.dumps(entry,default=str)+"\n"); log.flush()
                raise RuntimeError("Invalid prefix replay/internal execution; no group in this batch is learned")
            if statuses.get("administrative_timeout"):
                # Do not keep the fast worlds from a batch while dropping slow
                # interrupted worlds. This invocation ends without any update.
                entry["not_learned"] = "whole_batch_administratively_interrupted"
                log.write(json.dumps(entry,default=str)+"\n"); log.flush()
                state["active_batch"] = None
                state["stop_reason"] = "administrative_timeout_whole_batch_not_learned"
                break
            if not groups or time.monotonic() >= stop_at:
                entry["not_learned"] = "no_complete_groups_or_deadline"
                log.write(json.dumps(entry,default=str)+"\n"); log.flush()
                state["active_batch"] = None
                if time.monotonic() >= stop_at:
                    state["stop_reason"] = "deadline"; break
                continue
            torch.set_num_threads(args.num_threads)
            state["active_batch"]["stage"] = "optimizing_complete_groups"
            optimization_started = time.monotonic()
            def on_step(new_groups_used):
                state["optimizer_steps"] += 1
                state["learned_groups"] += new_groups_used
            losses = update_actor(model,optimizer,groups,args,stop_at=stop_at,on_step=on_step)
            state["update"] += 1
            state["active_batch"] = None
            state["elapsed_training_s"] = elapsed_before+time.monotonic()-started
            entry.update(update=state["update"],optimizer_steps=state["optimizer_steps"],losses=losses,
                optimize_wall_s=time.monotonic()-optimization_started,
                total_completed_groups=state["completed_groups"], attempted_groups=state["attempted_groups"],
                total_rollout_trajectories=state["actual_rollout_trajectories"],
                total_simulator_actions=state["simulator_actions"],elapsed_training_s=state["elapsed_training_s"])
            log.write(json.dumps(entry,default=str)+"\n"); log.flush()
            print(json.dumps({k:v for k,v in entry.items() if k!="collection_metrics"}),flush=True)
            if model_tensor_digest({k:v for k,v in model.state_dict().items() if k.startswith("critic.")}) != critic_digest:
                raise RuntimeError("Frozen critic head was modified")
            save_checkpoint(args.output / "latest.pt",model,optimizer,args,state)
            if time.monotonic()-last_snapshot >= args.checkpoint_seconds:
                save_checkpoint(args.output / f"rollout_{state['update']:06d}.pt",model,optimizer,args,state)
                last_snapshot = time.monotonic()
        state["stop_reason"] = state.get("stop_reason") or "update_limit"
    except BaseException:
        active = state.get("active_batch")
        if (active and active["stage"] == "optimizing_complete_groups"
                and state["optimizer_steps"] > active["optimizer_steps_before"]):
            state["update"] += 1
            active["partial_update_counted"] = True
        state["stop_reason"] = "interrupted_or_invalid_execution"
        raise
    finally:
        state["elapsed_training_s"] = elapsed_before+time.monotonic()-started
        save_checkpoint(args.output / "latest.pt",model,optimizer,args,state)
        save_checkpoint(args.output / f"rollout_{state['update']:06d}.pt",model,optimizer,args,state)
        log.close()
        if executor is not None:
            executor.shutdown(wait=True,cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Transactional CPU PPO for Q4 G1 micro decisions, fresh synthetic scenes only.

Shared PPO, returns, scene partition and descriptive statistics come from train;
the micro rollout, checkpoint schema and transaction driver are independent.
Every interrupted attempt is retained. No observation, reward or normalization
uses post-termination truth/lower bounds. Defaults start a new model from scratch.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import random
import signal
import time

import torch

from .micro_network import (MicroCandidateActorCritic, TorchPolicy, configure_cpu,
    feature_schema, model_from_metadata, validate_checkpoint, CHECKPOINT_VERSION,
    CONTROLLER_ENTRYPOINT, OBJECTIVE, architecture_name, make_model)
from .train import (TRAIN_START, TRAIN_END, DEFAULT_DEADLINE, TrainingStop,
    training_case_spec, attach_returns, ppo_update, imitation_update,
    summarize_training_metrics, parse_deadline, _configuration, _write_json, _write_batch)
from .training_journal import EpisodeJournal


_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def rollout(task):
    """Only public numeric features cross the controller-to-model boundary."""
    worker_started, worker_cpu_started = time.perf_counter(), time.process_time()
    configure_cpu()
    from simulation import LocalResearchSimulator
    from .micro_controller import run_q4_micro
    from .scenarios import build_case

    seed, action_seed = task["seed"], task["action_seed"]
    spec = training_case_spec(seed)
    if task["mode"] not in ("ppo", "imitation"):
        raise ValueError("unknown micro rollout mode")
    if time.time() >= task["deadline_epoch"]:
        return {"seed": seed, "administrative_skip": "deadline_before_start", "records": []}
    model = model_from_metadata(task["network"])
    model.load_state_dict(task["model"])
    model.eval()
    random.seed(action_seed)
    torch.manual_seed(action_seed)
    policy = TorchPolicy(model, deterministic=False) if task["mode"] == "ppo" else None
    remaining = max(1., task["deadline_epoch"]-time.time())
    simulator = LocalResearchSimulator(build_case(seed, split="train", **spec),
        max_real_duration_s=min(300., remaining))
    started, policy_cpu_started = time.perf_counter(), time.process_time()
    report = run_q4_micro(simulator.client(), policy=policy,
        max_decisions=task["max_decisions"], action_deadline_epoch=task["deadline_epoch"])
    evaluation = simulator.evaluation()  # Controller has terminated and exited.
    learning = report.learning
    records = [dict(row) for row in learning["transitions"]]
    success = bool(evaluation["all_cleared"] and report.completion_certified_under_model
                   and not report.error and not report.exit_error)
    if learning["feature_schema"] != feature_schema():
        raise ValueError("micro rollout public schema mismatch")
    if policy is not None:
        if len(records) != len(policy.records):
            raise ValueError("micro controller/policy transition count mismatch")
        for row, sample in zip(records, policy.records):
            for field in ("global_features", "candidate_features", "action_index"):
                if row[field] != sample[field]:
                    raise ValueError("micro controller/policy observation mismatch")
            row.update(log_prob=sample["log_prob"], value=sample["value"])
    accounting = attach_returns(records, actual_time_s=report.virtual_time_s, success=success)
    policy_wall_s = time.perf_counter()-started
    policy_cpu_s = time.process_time()-policy_cpu_started
    # The actual undiscounted returns are fixed before hindsight diagnostics.
    from experiments.q4_comparison_bounds import common_bound
    bound_started, bound_cpu_started = time.perf_counter(), time.process_time()
    bound = common_bound(evaluation["ground_truth"])
    lower = float(bound["common_lower_bound_s"])
    bound_wall_s, bound_cpu_s = time.perf_counter()-bound_started, time.process_time()-bound_cpu_started
    metrics = {"seed": seed, "split": "train", **spec, "success": success,
        **accounting, "failed_clear_count": int(evaluation["failed_clear_count"]),
        "fallback_cost_s": float(learning.get("fallback_cost_s", 0.)),
        "decisions": len(records), "action_counts": dict(learning["action_counts"]),
        "wall_time_s": time.perf_counter()-worker_started,
        "worker_cpu_s": time.process_time()-worker_cpu_started,
        "policy_wall_s": policy_wall_s, "policy_cpu_s": policy_cpu_s,
        "posthoc_bound_wall_s": bound_wall_s, "posthoc_bound_cpu_s": bound_cpu_s,
        "common_lower_bound_s": lower,
        "actual_time_over_lower_bound": report.virtual_time_s/lower if success else None,
        "penalized_time_over_lower_bound": accounting["penalized_time_s"]/lower,
        "completion_reason": report.completion_reason,
        "error": report.error, "exit_error": report.exit_error}
    result = {"seed": seed, "records": records, "metrics": metrics,
        "controller_entrypoint": CONTROLLER_ENTRYPOINT,
        "evaluation_phase": "after_policy_termination",
        "action_history": report.action_history,
        "observation_history": simulator.observation_history(),
        "controller_learning": {key: value for key, value in learning.items() if key != "transitions"},
        "strategy_parameters": report.strategy_parameters,
        "evaluation": evaluation, "common_bound": bound}
    if report.completion_reason == "training_deadline" or (not success and time.time() >= task["deadline_epoch"]):
        result["administrative_skip"] = "training_deadline_during_episode"
    return result


def save_checkpoint(path, model, optimizer, state, config):
    metadata = model.metadata()
    if (config.get("architecture", "mlp") != architecture_name(metadata)
            or any(p.device.type != "cpu" for p in model.parameters())):
        raise ValueError("cannot save a macro model as a micro checkpoint")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"version": CHECKPOINT_VERSION, "controller_entrypoint": CONTROLLER_ENTRYPOINT,
        "network": metadata, "feature_schema": feature_schema(),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "state": state, "config": config, "device": "cpu", "objective": dict(OBJECTIVE),
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}}
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def restore_checkpoint(path, *, learning_rate=None):
    configure_cpu()
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    rate = saved["config"]["learning_rate"]
    if learning_rate is not None and learning_rate != rate:
        raise ValueError("resuming cannot silently change optimizer learning rate")
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    optimizer.load_state_dict(saved["optimizer"])
    random.setstate(saved["rng"]["python"])
    torch.set_rng_state(saved["rng"]["torch"])
    return model, optimizer, saved["state"], saved["config"]


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-budget", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--architecture", choices=("mlp", "induced"), default="mlp")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--warmstart-episodes", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=424242)
    parser.add_argument("--scenario-start", type=int, default=TRAIN_START)
    parser.add_argument("--scenario-end", type=int, default=TRAIN_END)
    parser.add_argument("--entropy-coefficient", type=float, default=.005)
    parser.add_argument("--max-batches", type=int, default=100000)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.)
    parser.add_argument("--deadline", default=DEFAULT_DEADLINE)
    parser.add_argument("--progress-interval-seconds", type=float, default=1800.)
    args = parser.parse_args(argv)
    if not 1 <= args.workers < args.cpu_budget <= 60:
        parser.error("workers + 1 learner must fit CPU budget (maximum 60)")
    if not TRAIN_START <= args.scenario_start <= args.scenario_end <= TRAIN_END:
        parser.error("training seeds must remain inside 8000000..8099999")
    if (not all(math.isfinite(x) for x in (args.learning_rate, args.max_wall_seconds,
                                          args.entropy_coefficient, args.progress_interval_seconds))
            or min(args.hidden, args.epochs, args.minibatch_size, args.batch_episodes, args.max_decisions, args.max_batches) < 1
            or args.warmstart_episodes < 0 or args.learning_rate <= 0 or args.max_wall_seconds <= 0
            or args.entropy_coefficient < 0 or args.progress_interval_seconds <= 0):
        parser.error("invalid micro training budget")
    return parser, args


def configuration(args):
    result = {**_configuration(args), "controller_entrypoint": CONTROLLER_ENTRYPOINT}
    # Preserve the complete legacy MLP configuration shape for existing resume.
    if args.architecture != "mlp":
        result["architecture"] = args.architecture
    return result


def main(argv=None):
    global _stop_requested
    _stop_requested = False
    parser, args = _arguments(argv)
    configure_cpu()
    deadline = min(parse_deadline(args.deadline), time.time()+args.max_wall_seconds)
    if deadline <= time.time():
        parser.error("training deadline has passed; a new explicit deadline is required")
    config = configuration(args)
    if args.resume:
        if args.resume.resolve() != (args.output/"latest.pt").resolve():
            parser.error("resume must use this output directory's latest transaction checkpoint")
        model, optimizer, state, previous = restore_checkpoint(args.resume)
        if previous != config:
            parser.error("resume configuration differs; start an explicit new experiment")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("new training output must be empty; use --resume")
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        model = make_model(args.architecture, hidden=args.hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        state = dict(next_seed=args.scenario_start, episodes=0, attempted_episodes=0,
            warmstart_completed=0, ppo_batches=0, batches=0, wall_time_s=0.,
            pending_batch=None, next_attempt=0, stop_reason=None)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_checkpoint(args.output/"random.pt", model, optimizer, state, config)
    save_checkpoint(args.output/"latest.pt", model, optimizer, state, config)
    started, prior_wall = time.perf_counter(), state["wall_time_s"]
    next_review = started+args.progress_interval_seconds
    old_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    executor = None
    try:
        if args.workers > 1:
            executor = ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=configure_cpu)
        while state["batches"] < args.max_batches and time.time() < deadline-5 and not _stop_requested:
            if state["pending_batch"] is None:
                if state["next_seed"] > args.scenario_end:
                    state["stop_reason"] = "training_partition_exhausted"
                    break
                mode = "imitation" if state["warmstart_completed"] < args.warmstart_episodes else "ppo"
                count = min(args.batch_episodes, args.scenario_end-state["next_seed"]+1)
                if mode == "imitation":
                    count = min(count, args.warmstart_episodes-state["warmstart_completed"])
                seeds = list(range(state["next_seed"], state["next_seed"]+count))
                state["next_seed"] += count
                state["attempted_episodes"] += count
                state["pending_batch"] = dict(mode=mode, seeds=seeds,
                    action_seeds=[random.randrange(2**31) for _ in seeds])
            pending = state["pending_batch"]
            attempt = state["next_attempt"]
            state["next_attempt"] += 1
            state["wall_time_s"] = prior_wall+time.perf_counter()-started
            save_checkpoint(args.output/"latest.pt", model, optimizer, state, config)
            raw_name = f"batch-{state['batches']:06d}-attempt-{attempt:06d}.json.gz"
            raw_path = args.output/raw_name
            if raw_path.exists():
                raise FileExistsError("refusing to overwrite a recorded micro attempt")
            journal = EpisodeJournal(raw_path, _write_batch)
            tasks = [dict(seed=seed, action_seed=action_seed, mode=pending["mode"],
                model=model.state_dict(), network=model.metadata(), max_decisions=args.max_decisions,
                deadline_epoch=deadline) for seed, action_seed in zip(pending["seeds"], pending["action_seeds"])]
            batch = []
            rows = executor.map(rollout, tasks) if executor else map(rollout, tasks)
            for row in rows:
                batch.append(row)
                # Preserve every returned episode once. Rewriting only the small
                # hash index avoids repeatedly serializing previous trajectories.
                journal.append(row)
            if _stop_requested or any(row.get("administrative_skip") for row in batch):
                state["stop_reason"] = "deadline_in_reserved_batch"
                break
            records = [record for row in batch for record in row["records"]]
            update_started, update_cpu_started = time.perf_counter(), time.process_time()
            if pending["mode"] == "imitation":
                update = imitation_update(model, optimizer, records, epochs=args.epochs,
                    minibatch_size=args.minibatch_size, stop_check=lambda: _stop_requested or time.time() >= deadline)
                state["warmstart_completed"] += len(batch)
            else:
                update = ppo_update(model, optimizer, records, epochs=args.epochs,
                    minibatch_size=args.minibatch_size, entropy_coefficient=args.entropy_coefficient,
                    stop_check=lambda: _stop_requested or time.time() >= deadline)
                state["ppo_batches"] += 1
            update.update(wall_time_s=time.perf_counter()-update_started,
                          cpu_time_s=time.process_time()-update_cpu_started)
            state["episodes"] += len(batch)
            state["batches"] += 1
            state["pending_batch"] = None
            state["wall_time_s"] = prior_wall+time.perf_counter()-started
            state["stop_reason"] = None
            progress = dict(batch=state["batches"], phase=pending["mode"], episodes=state["episodes"],
                next_seed=state["next_seed"], wall_time_s=state["wall_time_s"], raw_attempt=raw_name,
                controller_entrypoint=CONTROLLER_ENTRYPOINT,
                **summarize_training_metrics([row["metrics"] for row in batch]), update=update,
                scope="Synthetic micro training diagnostics only; independent evaluation required before promotion")
            # The progress payload travels with the model transaction. A crash
            # after commit cannot lose its raw provenance or duplicate an update.
            state["last_progress"] = progress
            save_checkpoint(args.output/"latest.pt", model, optimizer, state, config)
            save_checkpoint(args.output/f"checkpoint-{state['batches']:06d}.pt", model, optimizer, state, config)
            _write_json(args.output/"progress.json", progress)
            _write_json(args.output/f"progress-{state['batches']:06d}.json", progress)
            if pending["mode"] == "imitation" and state["warmstart_completed"] == args.warmstart_episodes:
                save_checkpoint(args.output/"warmstart.pt", model, optimizer, state, config)
            if time.perf_counter() >= next_review:
                _write_json(args.output/"review_due.json", dict(batch=state["batches"],
                    checkpoint=f"checkpoint-{state['batches']:06d}.pt", review_requested=True,
                    reason="Run a separately frozen development evaluation before promotion."))
                next_review = time.perf_counter()+args.progress_interval_seconds
            print(json.dumps(progress, allow_nan=False), flush=True)
        if state["stop_reason"] is None:
            state["stop_reason"] = "batch_limit" if state["batches"] >= args.max_batches else "wall_or_training_deadline"
    except TrainingStop:
        model, optimizer, state, config = restore_checkpoint(args.output/"latest.pt")
        state["stop_reason"] = "administrative_stop_reserved_batch_retained"
    except BaseException:
        model, optimizer, state, config = restore_checkpoint(args.output/"latest.pt")
        state["stop_reason"] = "interrupted_or_error"
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        state["wall_time_s"] = prior_wall+time.perf_counter()-started
        save_checkpoint(args.output/"latest.pt", model, optimizer, state, config)
        _write_json(args.output/"status.json", state)
        if state.get("last_progress"):
            _write_json(args.output/"progress.json", state["last_progress"])
        signal.signal(signal.SIGTERM, old_sigterm)
    return state


if __name__ == "__main__":
    main()

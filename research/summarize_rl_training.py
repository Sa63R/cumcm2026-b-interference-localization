"""Read archived logs only: resource audit, training traces, fixed validation.

Never imports a strategy/simulator or loads a pickle checkpoint. Archives and
validation members are selected explicitly; final-test artifacts are refused.
"""

import argparse
from collections import Counter
import hashlib
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import re
import random
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ARCHIVES = ("milestone-0200.tar.gz", "milestone-0300.tar.gz", "cold-001-complete.tar.gz",
                    "milestone-0410.tar.gz", "milestone-0510.tar.gz",
                    "milestone-0550.tar.gz", "milestone-0605.tar.gz", "milestone-0715.tar.gz")
VALIDATION_SEEDS = list(range(6000, 6048))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact_ranges(values):
    result = []
    for value in sorted(set(values)):
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def legal_seed(seed):
    return 100001 <= seed <= 199999 or 2000 <= seed <= 5099


def summarize_log(stream, config):
    """Keep compact per-update scalars; discard repeated per-source ledgers."""
    sha = hashlib.sha256()
    byte_count = 0
    stages = Counter()
    stage_seeds = {stage: [] for stage in ("bc", "ppo", "reinforce", "deadline")}
    all_seen = []
    deadline_skipped = deadline_recorded = failures = 0
    baseline_used = baseline_failures = deadline_completed_trajectories = 0
    optimizer_steps = optimizer_unknown = 0
    sample_s = optimize_s = 0.0
    curve = []
    per_episode_wall = 0.0
    records_count = 0
    bc_checkpoint_sha = None
    last_stage = None
    for raw in stream:
        sha.update(raw)
        byte_count += len(raw)
        if not raw.strip():
            continue
        row = json.loads(raw)
        stage = row["stage"]
        if stage not in stage_seeds:
            raise ValueError(f"Unexpected stage: {stage}")
        stages[stage] += 1
        last_stage = stage
        episodes = row.get("episodes", [])
        used = []
        for episode in episodes:
            seed = episode["seed"]
            if not legal_seed(seed):
                raise ValueError(f"Training log contains non-protocol seed {seed}")
            all_seen.append(seed)
            if episode.get("deadline_skipped"):
                deadline_skipped += 1
                deadline_completed_trajectories += episode.get("completed_trajectories", 0)
                continue
            if stage == "deadline":
                deadline_recorded += 1
            else:
                stage_seeds[stage].append(seed)
                used.append(episode.get("sampled", episode))
                if "baseline" in episode:
                    baseline_used += 1
                    baseline_failures += int(not episode["baseline"]["success"])
            failures += int(not episode.get("sampled", episode).get("success", False))
            per_episode_wall += episode.get("wall_time_s", 0.0)
        if stage in ("bc", "ppo", "reinforce"):
            sample_s += row.get("collect_wall_s", 0.0)
            optimize_s += row.get("optimize_wall_s", 0.0)
            steps = row.get("losses", {}).get("optimizer_steps")
            if steps is None:
                optimizer_unknown += 1
            else:
                if steps <= 0 and stage in ("ppo", "reinforce"):
                    raise ValueError("A logged policy update performed no optimizer step")
                optimizer_steps += steps
        if stage == "bc":
            bc_checkpoint_sha = row.get("checkpoint_sha256")
        if stage in ("ppo", "reinforce"):
            if curve and row["update"] != curve[-1]["update"] + 1:
                raise ValueError("Nonconsecutive update log; a resume segment needs an explicit join")
            times = [e["virtual_time_s"] for e in used]
            if times and not math.isclose(np.mean(times), row["mean_virtual_time_s"], abs_tol=1e-6):
                raise ValueError("Stored training mean differs from its episode records")
            records_count += sum(e.get("records", e.get("decisions", 0)) for e in used)
            curve.append(dict(update=row["update"], stage=stage,
                new_policy_episodes=len(stage_seeds["ppo"])+len(stage_seeds["reinforce"]),
                mean_virtual_time_s=row["mean_virtual_time_s"],
                elapsed_wall_s=row.get("elapsed_wall_s"),
                collect_wall_s=row.get("collect_wall_s"), optimize_wall_s=row.get("optimize_wall_s"),
                full_action_entropy=row.get("losses", {}).get("entropy"),
                teacher_cross_entropy=row.get("losses", {}).get("bc_loss"),
                scan_site_changes=float(np.mean([e.get("learning", {}).get("scan_site_changes", 0) for e in used])),
                interrupted_scans=float(np.mean([e.get("learning", {}).get("interrupted_scans", 0) for e in used]))))
    if not curve:
        raise ValueError("No completed policy updates in selected training log")
    policy_seeds = stage_seeds["ppo"] + stage_seeds["reinforce"]
    used_seeds = stage_seeds["bc"] + policy_seeds
    if curve[-1]["update"] >= config["updates"]:
        stop = "requested_updates_reached"
    elif last_stage == "deadline":
        stop = "explicit_deadline_log"
    elif curve[-1]["elapsed_wall_s"] >= config["max_wall_s"]:
        stop = "configured_wall_budget_reached_in_log"
    else:
        stop = "snapshot_before_requested_updates_stop_unknown"
    summary = dict(log_sha256=sha.hexdigest(), log_bytes=byte_count,
        stage_counts=dict(stages), actual_bc_episodes=len(stage_seeds["bc"]),
        ppo_episodes_used=len(stage_seeds["ppo"]), reinforce_episodes_used=len(stage_seeds["reinforce"]),
        policy_episodes_used=len(policy_seeds), paired_baseline_episodes_used=baseline_used,
        paired_baseline_failures=baseline_failures, unique_trial_training_seeds=len(set(used_seeds)),
        repeated_seed_episodes_within_trial=len(used_seeds)-len(set(used_seeds)),
        actual_bc_seed_ranges=compact_ranges(stage_seeds["bc"]),
        actual_policy_seed_ranges=compact_ranges(policy_seeds),
        logged_seed_ranges_including_deadline=compact_ranges(all_seen),
        completed_updates=curve[-1]["update"], stop_evidence=stop,
        last_logged_elapsed_s=curve[-1]["elapsed_wall_s"],
        summed_collect_wall_s=sample_s, summed_optimize_wall_s=optimize_s,
        summed_episode_worker_wall_s=per_episode_wall,
        policy_decisions_recorded=records_count, completed_episode_failures=failures,
        deadline_unoptimized_completed_or_partial_episodes=deadline_recorded,
        deadline_skipped_tasks=deadline_skipped,
        deadline_completed_trajectories_before_skip=deadline_completed_trajectories,
        optimizer_steps_when_recorded=optimizer_steps,
        optimization_stage_rows_missing_step_count=optimizer_unknown,
        bc_checkpoint_sha256=bc_checkpoint_sha)
    return summary, curve, set(used_seeds)


def validation_from_rows(rows, manifest, summary, baseline):
    if manifest.get("split") != "validation" or manifest.get("seeds") != VALIDATION_SEEDS:
        raise ValueError("Only the fixed 6000..6047 validation set may be summarized")
    rows = sorted(rows, key=lambda row: row["seed"])
    if [r["seed"] for r in rows] != VALIDATION_SEEDS:
        raise ValueError("Validation rows have missing/duplicate/unexpected seeds")
    if not all(r["successful"] and r["failed_clear_count"] == 0 for r in rows):
        raise ValueError("Failed candidates cannot enter raw-time performance figures; report their penalties separately")
    if any(r["case_sha256"] != b["case_sha256"] for r, b in zip(rows, baseline)):
        raise ValueError("Paired validation case identities differ")
    cost = np.asarray([r["virtual_time_s"] for r in rows])
    saved = np.asarray([r["virtual_time_s"] for r in baseline]) - cost
    if not math.isclose(cost.mean(), summary["raw_mean_total_time_s"], abs_tol=1e-6):
        raise ValueError("Validation summary mean differs from row records")
    rng = np.random.default_rng(913)
    bootstrap = saved[rng.integers(0, len(saved), size=(10000, len(saved)))].mean(1)
    return dict(strategy=manifest["strategy"], runs=len(rows),
        successful_runs=sum(bool(r["successful"]) for r in rows),
        failed_clear_count=sum(r["failed_clear_count"] for r in rows),
        mean_virtual_time_s=float(cost.mean()), p95_virtual_time_s=float(np.quantile(cost, .95)),
        case_virtual_times_s=cost.tolist(),  # Fixed seed order for initialization/training contrasts.
        saved_vs_rollout_s=float(saved.mean()),
        saved_vs_rollout_ci95_s=np.quantile(bootstrap, [.025, .975]).tolist(),
        wins=int((saved > 1e-6).sum()), losses=int((saved < -1e-6).sum()),
        ties=int((np.abs(saved) <= 1e-6).sum()),
        worst_regression_s=float(-saved.min()),
        checkpoint_sha256=manifest.get("identity", {}).get("checkpoint_sha256"),
        source_spec_sha256=manifest.get("identity", {}).get("spec_sha256"))


def read_archive(path, baseline):
    if any(word in path.name.lower() for word in ("final", "extended")):
        raise ValueError("Final/extended evaluation archives are excluded from this development audit")
    archive_sha = digest(path.read_bytes())
    source = dict(archive=path.name, bytes=path.stat().st_size, sha256=archive_sha, members=[])
    trials, validation, comparisons, seed_sets = {}, {}, {}, {}
    with tarfile.open(path) as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}

        def read_json(name):
            data = archive.extractfile(members[name]).read()
            source["members"].append(dict(name=name, bytes=len(data), sha256=digest(data)))
            return json.loads(data)

        for name in sorted(members):
            basename = PurePosixPath(name).name
            if (basename == "config.json" or (basename.startswith("config-u") and basename.endswith(".json"))) and "/results/rl/" in name:
                config = read_json(name)
                directory = str(PurePosixPath(name).parent)
                log_name = directory + "/training.jsonl"
                if log_name not in members:
                    continue
                trial = PurePosixPath(directory).name
                if trial in trials:
                    raise ValueError("Multiple resume config segments require an explicit audited join")
                detail, curve, seeds = summarize_log(archive.extractfile(members[log_name]), config)
                source["members"].append(dict(name=log_name, bytes=detail["log_bytes"], sha256=detail["log_sha256"]))
                retained_config = {key: value for key, value in config.items()
                                   if key not in ("feature_schema", "source_manifest")}
                retained_config["source_manifest_sha256"] = config.get("source_manifest", {}).get("sha256")
                trials[trial] = dict(config=retained_config, summary=detail, curve=curve,
                                     archive=path.name, config_member=name, log_member=log_name)
                seed_sets[trial] = seeds
            if name.endswith("/manifest.json") and "/validation-" in name:
                directory = str(PurePosixPath(name).parent)
                manifest = read_json(name)
                if manifest.get("strategy", "").startswith(("state", "geo")):
                    continue
                rows = read_json(directory + "/rows.json")
                summary = read_json(directory + "/summary.json")
                detail = validation_from_rows(rows, manifest, summary, baseline)
                detail.update(archive=path.name, manifest_member=name)
                validation[manifest["strategy"]] = detail
            if name.startswith("comparisons/validation-") and name.endswith(".json"):
                comparisons[name] = read_json(name)
    if digest(path.read_bytes()) != archive_sha:
        raise ValueError("Archive changed while being read; use a completed immutable export")
    return source, trials, validation, comparisons, seed_sets


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None}, facecolor="white")
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def checkpoint_update(strategy, trial):
    if not strategy.startswith(trial + "-"):
        return None
    if strategy == trial + "-initialized":
        return 0
    match = re.search(r"(?:-u|-ppo_|-reinforce_)(\d+)$", strategy)
    return int(match.group(1)) if match else None


def paired_endpoint(source, target):
    """Same already checked 48-case identity/order; positive means target faster."""
    savings = np.asarray(source["case_virtual_times_s"]) - np.asarray(target["case_virtual_times_s"])
    # Match the frozen shared report's Python random.choices stream, not the
    # older audit-only numpy bootstrap used for each endpoint versus rollout.
    rng = random.Random(20260911)
    values = savings.tolist()
    means = [sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000)]
    return dict(mean_seconds_saved=float(savings.mean()),
        saving_ci95_s=np.quantile(means, [.025,.975]).tolist(),
        bootstrap=dict(seed=20260911,resamples=10000,rng="Python random.choices",method="paired_percentile"),
        wins=int((savings>1e-6).sum()), losses=int((savings < -1e-6).sum()))


def audit_selected_designs(report):
    """Verify claimed inheritance and matched budgets from archived configs."""
    trials = report["trials"]
    chain = ("joint-cold-mlp-001", "cold-finetune-ppo-001", "cold-finetune-ppo-002")
    designs = {}
    if all(name in trials for name in chain):
        cumulative = 0
        links = []
        for index, name in enumerate(chain):
            run = trials[name]
            source = run["config"].get("initialize_from")
            if index == 0:
                if source is not None:
                    raise ValueError("Selected cold chain is not initialized from random")
            else:
                predecessor = trials[chain[index-1]]
                expected_file = f"ppo_{predecessor['summary']['completed_updates']:06d}.pt"
                if (not source or PurePosixPath(source).parent.name != chain[index-1]
                        or PurePosixPath(source).name != expected_file):
                    raise ValueError("Selected model chain does not match recorded initialization")
            if run["summary"]["actual_bc_episodes"] != 0:
                raise ValueError("Selected no-BC model chain contains BC")
            cumulative += run["summary"]["policy_episodes_used"]
            links.append(dict(trial=name, cumulative_policy_episodes=cumulative))
        designs["selected_ppo_chain"] = links
    group = ("joint-cold-group-alpha0-002", "joint-cold-group-alpha1-002")
    if all(name in trials for name in group):
        left, right = (trials[name] for name in group)
        keys = ("git_commit", "feature_version", "architecture", "hidden", "seed", "scenario_start",
                "updates", "episodes_per_update", "bc_episodes", "initialize_from", "epochs",
                "minibatch", "lr", "gae_lambda", "clip", "value_coef", "entropy_coef",
                "aux_bc_coef", "target_kl", "max_decisions", "workers", "num_threads", "max_wall_s")
        if any(left["config"].get(k) != right["config"].get(k) for k in keys):
            raise ValueError("Group comparison has an undeclared matched-config difference")
        if [left["config"].get("group_alpha"), right["config"].get("group_alpha")] != [0, 1]:
            raise ValueError("Expected an alpha0/alpha1 ablation")
        for key in ("policy_episodes_used", "completed_updates", "actual_policy_seed_ranges", "actual_bc_episodes"):
            if left["summary"][key] != right["summary"][key]:
                raise ValueError("Group endpoint is not matched in observed training data")
        designs["matched_group_endpoint"] = dict(trials=list(group), config_keys_checked=list(keys),
            completed_updates=left["summary"]["completed_updates"],
            sampled_episodes_each=left["summary"]["policy_episodes_used"],
            scope="same source/config and completed scenario sequence; intermediate wallclock checkpoints not matched")
    probes = ("cold-probes-base-001", "cold-probes-axis_quantiles-001")
    if all(name in trials for name in probes):
        left,right = (trials[name] for name in probes)
        keys = ("git_commit", "feature_version", "architecture", "hidden", "seed", "scenario_start",
                "updates", "episodes_per_update", "bc_episodes", "initialize_from", "epochs",
                "minibatch", "lr", "gae_lambda", "clip", "value_coef", "entropy_coef", "group_alpha",
                "aux_bc_coef", "target_kl", "max_decisions", "workers", "num_threads", "max_wall_s")
        if any(left["config"].get(key)!=right["config"].get(key) for key in keys):
            raise ValueError("Base/axis comparison has an undeclared matched-config difference")
        if [left["config"]["probe_candidates"],right["config"]["probe_candidates"]] != ["base","axis_quantiles"]:
            raise ValueError("Expected an explicit base/axis action-schema comparison")
        for key in ("policy_episodes_used","completed_updates","actual_policy_seed_ranges","actual_bc_episodes"):
            if left["summary"][key]!=right["summary"][key]:
                raise ValueError("Base/axis completed training budgets do not match")
        validation = report["validation"]
        initial = {name:validation[name+"-initialized"] for name in probes}
        old = validation["cold-finetune-ppo-002-ppo_000384"]
        if initial[probes[0]]["case_virtual_times_s"] != old["case_virtual_times_s"]:
            raise ValueError("Base initialization does not preserve the recorded best002 episode costs")
        deltas = {}
        for name in probes:
            for strategy, record in validation.items():
                update = checkpoint_update(strategy,name)
                if update is not None and update>0:
                    deltas[strategy]=paired_endpoint(initial[name],record)
        designs["matched_probe_endpoint"] = dict(trials=list(probes),config_keys_checked=list(keys),
            sampled_episodes_each=left["summary"]["policy_episodes_used"],
            completed_updates_each=left["summary"]["completed_updates"],
            base_initialization_matches_best002=True,
            axis_zero_training_change=paired_endpoint(initial[probes[0]],initial[probes[1]]),
            training_vs_own_initialization=deltas,
            final_axis_vs_base=paired_endpoint(validation[probes[0]+"-ppo_000512"],validation[probes[1]+"-ppo_000512"]),
            scope="Shared inherited weights and matched final sample count; action-set change already affects initialized policy. Early wallclock checkpoints have different updates.")
    routes = ("cold-route-v3-001", "cold-route-v4-001")
    if all(name in trials for name in routes):
        left,right = (trials[name] for name in routes)
        keys=("git_commit","architecture","hidden","seed","scenario_start","updates","episodes_per_update",
              "bc_episodes","initialize_from","epochs","minibatch","lr","gae_lambda","clip","value_coef",
              "entropy_coef","group_alpha","probe_candidates","aux_bc_coef","target_kl","max_decisions",
              "workers","num_threads","max_wall_s")
        if any(left["config"].get(k)!=right["config"].get(k) for k in keys):
            raise ValueError("Route representation comparison has an undeclared config difference")
        if [left["config"]["feature_version"],right["config"]["feature_version"]] != ["v3","v4"]:
            raise ValueError("Expected a v3/v4 representation comparison")
        for key in ("policy_episodes_used","completed_updates","actual_policy_seed_ranges","actual_bc_episodes"):
            if left["summary"][key]!=right["summary"][key]:
                raise ValueError("Route representation training endpoint budgets differ")
        validation=report["validation"]
        parent=validation["cold-finetune-ppo-002-ppo_000384"]
        if any(validation[name+"-initialized"]["case_virtual_times_s"]!=parent["case_virtual_times_s"] for name in routes):
            raise ValueError("Route initialized costs do not match the parent model")
        deltas={strategy:paired_endpoint(parent,record) for name in routes for strategy,record in validation.items()
                if checkpoint_update(strategy,name) is not None and checkpoint_update(strategy,name)>0}
        designs["matched_route_endpoint"]=dict(trials=list(routes),config_keys_checked=list(keys),
            sampled_episodes_each=left["summary"]["policy_episodes_used"],completed_updates_each=left["summary"]["completed_updates"],
            both_initializations_match_parent_virtual_times=True,training_vs_parent=deltas,
            final_v4_vs_v3=paired_endpoint(validation[routes[0]+"-ppo_000512"],validation[routes[1]+"-ppo_000512"]),
            scope="Same initial episode costs, actions/schema and completed samples; only v4 observation representation changes. Early wallclock checkpoints have unequal update counts.")
    return designs


def figures(report, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "q3-rl-training-audit-v1"})
    trials, validation = report["trials"], report["validation"]
    # Separate the later, explicitly matched distribution experiment below.
    cold = [name for name in trials if name.startswith("joint-cold-mlp-")
            and not trials[name]["config"].get("initialize_from")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    colors = ["#087e8b", "#b44c30", "#5b4db2", "#666666"]
    for index, name in enumerate(cold):
        run = trials[name]
        curve = run["curve"]
        x = np.asarray([r["update"] for r in curve])
        y = np.asarray([r["mean_virtual_time_s"] for r in curve])
        color = colors[index % len(colors)]
        label = f"seed {run['config']['seed']}"
        axes[0].plot(x, y, alpha=.16, linewidth=.65, color=color)
        ends = list(range(20, len(y)+1, 20))
        if not ends or ends[-1] != len(y): ends.append(len(y))
        starts = [0] + ends[:-1]
        axes[0].plot([x[e-1] for e in ends], [y[s:e].mean() for s,e in zip(starts,ends)],
                     linewidth=1.8, color=color, label=label)
        points = []
        for strategy, record in validation.items():
            update = checkpoint_update(strategy, name)
            if update is not None:
                points.append((update, record["mean_virtual_time_s"]))
        if points:
            points.sort()
            axes[1].plot(*zip(*points), "o--", linewidth=1.2, markersize=6, color=color, label=label)
            for update, mean in points:
                axes[1].annotate(f"{mean:.1f}", (update, mean), xytext=(0,8), textcoords="offset points", ha="center", color=color)
    axes[0].set(title="A  Stochastic training batches (changing worlds)", xlabel="PPO update", ylabel="Virtual time per episode (s; log scale)", yscale="log")
    axes[0].text(.98, .97, "Thin: each 32-episode batch\nThick: non-overlapping 20-update means\nNot a validation learning curve", transform=axes[0].transAxes, ha="right", va="top", fontsize=9)
    axes[1].axhline(report["baseline_mean_s"], color="#555555", linestyle=":", label="Frozen rollout baseline")
    axes[1].set(title="B  Fixed validation: 48 identical cases", xlabel="PPO update", ylabel="Greedy mean virtual time (s)")
    axes[1].margins(x=.1, y=.15)
    axes[1].text(.02, .03, "Markers are measured checkpoints only.\nSeeds may follow different scenario sequences.", transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="upper right" if ax is axes[1] else "lower left", fontsize=8)
    save_figure(fig, output / "cold_learning")

    selected = ["ppo_trial001_u2465", "ppo_transfer_v1_001_u1000", "ppo_transfer_v2_001_u1000",
                "joint-mlp-matched-001-u256", "joint-attention-matched-001-u256",
                "joint-cold-mlp-001-u213", "joint-cold-mlp-001-u512"]
    for name in trials:
        if name == "joint-cold-mlp-001" or "cold" not in name:
            continue
        measured = [(checkpoint_update(s, name), s) for s in validation if checkpoint_update(s, name) is not None]
        if measured: selected.append(max(measured)[1])
    selected = [s for s in selected if s in validation]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.8, .62*len(selected))), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    labels = []
    for index, strategy in enumerate(selected):
        record = validation[strategy]
        mean = record["saved_vs_rollout_s"]
        low, high = record["saved_vs_rollout_ci95_s"]
        axes[0].errorbar(mean, index, xerr=[[mean-low], [high-mean]], fmt="o", capsize=3,
                         color="#087e8b" if low > 0 else "#6b7280")
        labels.append(strategy.replace("joint-", "").replace("-matched-001", "").replace("-001", "").replace("ppo_transfer_", "").replace("_001", "").replace("-ppo_000", "-u").replace("-reinforce_000", "-u"))
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, xlabel="Seconds saved vs rollout (positive is faster)",
                title="A  Fixed validation endpoints; paired 95% CI")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#555555", linestyle=":")
    names = list(trials)
    collect = np.asarray([trials[n]["summary"]["summed_collect_wall_s"] / 60 for n in names])
    optimize = np.asarray([trials[n]["summary"]["summed_optimize_wall_s"] / 60 for n in names])
    elapsed = np.asarray([trials[n]["summary"]["last_logged_elapsed_s"] / 60 for n in names])
    remaining = np.maximum(0, elapsed-collect-optimize)
    for values, left, label, color in ((collect, np.zeros(len(names)), "Collection", "#3592a0"),
        (optimize, collect, "Optimization", "#df9a51"), (remaining, collect+optimize, "Other recorded elapsed", "#c5cbd3")):
        axes[1].barh(range(len(names)), values, left=left, color=color, label=label)
    axes[1].set(yticks=range(len(names)), yticklabels=[n.replace("joint-", "").replace("-matched-001", "").replace("-001", "") for n in names],
                xlabel="Minutes through last logged policy update", title="B  Process elapsed time (not exclusive GPU time)")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5,-.08), ncol=3)
    for ax in axes: ax.grid(axis="x", alpha=.15)
    save_figure(fig, output / "endpoints_and_cost")

    chain_names = ("joint-cold-mlp-001", "cold-finetune-ppo-001", "cold-finetune-ppo-002")
    group_names = ("joint-cold-group-alpha0-002", "joint-cold-group-alpha1-002")
    if not all(name in trials for name in chain_names + group_names):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), layout="constrained")
    offset = 0
    for index, name in enumerate(chain_names):
        run = trials[name]
        count = run["config"]["episodes_per_update"]
        points = sorted((offset + checkpoint_update(s, name)*count, r["mean_virtual_time_s"])
                        for s, r in validation.items() if checkpoint_update(s, name) is not None)
        if points:
            axes[0].plot(*zip(*points), "o--", color=colors[index],
                         label=("Cold start", "Fine-tune 1", "Fine-tune 2")[index])
            for x, y in points:
                axes[0].annotate(f"{y:.1f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        offset += run["summary"]["policy_episodes_used"]
    axes[0].set(title="A  Selected PPO chain: fixed 48-case validation",
                xlabel="Cumulative sampled episodes in this model's ancestry", ylabel="Greedy mean virtual time (s)")
    for index, name in enumerate(group_names):
        points = sorted((checkpoint_update(s, name), r["mean_virtual_time_s"])
                        for s, r in validation.items() if checkpoint_update(s, name) is not None)
        axes[1].plot(*zip(*points), "o--", color=colors[index], label=f"alpha {index}; seed 9112033")
        for x, y in points:
            axes[1].annotate(f"{y:.1f}", (x, y), xytext=(-5 if index else 5, 9),
                             textcoords="offset points", ha="center", fontsize=8)
    axes[1].set(title="B  Group correction: matched endpoint at u512",
                xlabel="PPO update (32 sampled episodes each)", ylabel="Greedy mean virtual time (s)")
    axes[1].text(.02, .03, "Early checkpoints: u184 vs u190 (unequal budget).\nFinal alpha 1 - alpha 0 benefit is not established.",
                 transform=axes[1].transAxes, fontsize=8)
    for ax in axes:
        ax.axhline(report["baseline_mean_s"], linestyle=":", color="#555555", label="Frozen rollout")
        ax.grid(axis="y", alpha=.18)
        ax.margins(x=.08, y=.22)
        ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, output / "chain_and_group")

    if "matched_probe_endpoint" in report["audited_designs"]:
        names = ("cold-probes-base-001", "cold-probes-axis_quantiles-001")
        fig,axes = plt.subplots(1,2,figsize=(12.8,4.9),layout="constrained")
        contrast_rows=[]
        for index,name in enumerate(names):
            points=sorted((checkpoint_update(s,name),r["mean_virtual_time_s"])
                          for s,r in validation.items() if checkpoint_update(s,name) is not None)
            axes[0].plot(*zip(*points),"o--",color=colors[index],label=("Base","Axis candidates")[index])
            axes[0].annotate(f"{points[0][1]:.1f}",(0,points[0][1]),xytext=(5,7),
                             textcoords="offset points",fontsize=8,color=colors[index])
            for strategy,delta in report["audited_designs"]["matched_probe_endpoint"]["training_vs_own_initialization"].items():
                if strategy.startswith(name+"-"):
                    contrast_rows.append((strategy.replace(name,("base","axis")[index]).replace("-ppo_000"," u"),delta,colors[index]))
        axes[0].set(title="A  Base / axis: fixed 48-case validation",xlabel="Additional PPO updates after initialization",
                    ylabel="Greedy mean virtual time (s)")
        axes[0].text(.02,.96,"u0 differs only through the candidate set.\nEarly checkpoints have unequal sample counts.",
                     transform=axes[0].transAxes,va="top",fontsize=8)
        axes[0].legend(fontsize=8,loc="lower right")
        axes[0].margins(y=.30)
        for i,(name,delta,color) in enumerate(contrast_rows):
            mean=delta["mean_seconds_saved"]; low,high=delta["saving_ci95_s"]
            axes[1].errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt="o",color=color,capsize=3)
        axes[1].set(title="B  Training benefit relative to own u0",xlabel="Paired seconds saved (95% CI)",
                    yticks=range(len(contrast_rows)),yticklabels=[row[0] for row in contrast_rows])
        axes[1].invert_yaxis(); axes[1].axvline(0,color="#555555",linestyle=":")
        for ax in axes: ax.grid(alpha=.15)
        save_figure(fig,output/"probe_initialization_and_learning")


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in rows) + "\n"


def write_report(report, output):
    trials = report["trials"]
    rows, configs = [], []
    for name, run in trials.items():
        c, s = run["config"], run["summary"]
        architecture = c.get("architecture", "mlp")
        version = c.get("feature_version", c.get("algorithm", "unknown").rsplit("-",1)[-1])
        trainer = "REINFORCE" if "reinforce" in c.get("trainer", "") else "PPO"
        init = PurePosixPath(c["initialize_from"]).parent.name + "/" + PurePosixPath(c["initialize_from"]).name if c.get("initialize_from") else "random"
        distribution = c.get("group_alpha", 0)
        rows.append([name, version+" / "+architecture+" / "+trainer+f" / alpha{distribution}",
            f"{s['completed_updates']}/{c['updates']}", s["actual_bc_episodes"], s["policy_episodes_used"], s["paired_baseline_episodes_used"],
            f"{s['last_logged_elapsed_s']/60:.2f}", s["stop_evidence"]])
        configs.append([name, c["seed"], init, str(s["actual_policy_seed_ranges"]),
                        c["git_commit"][:8], c["workers"], c["hidden"]])
    validations = []
    for name, row in report["validation"].items():
        low, high = row["saved_vs_rollout_ci95_s"]
        validations.append([name, f"{row['mean_virtual_time_s']:.3f}",
            f"{row['saved_vs_rollout_s']:.2f} [{low:.2f}, {high:.2f}]",
            f"{row['wins']}/{row['losses']}/{row['ties']}", f"{row['successful_runs']}/48", row["failed_clear_count"],
            f"{row['worst_regression_s']:.3f}"])
    text = """# Q3 RL 训练试验与固定验证记录

这份记录只读取指定归档中的真实日志和已经完成的验证结果，不运行仿真、不加载模型、不写论文正文。训练曲线与固定验证严格分开。全部输入、成员文件和分析脚本的SHA256见 `audit.json`；原始大日志保留在归档中，仓库只保存 `training_curves.json.gz` 中的精简每更新标量。此表覆盖已提供完整训练日志的试验；只有验证产物而缺少训练日志的旧试验只列验证端点，不补造训练成本。

## 训练试验总表

实际BC局数由日志阶段计算，不按config中的默认参数推测。继承检查点的试验不会重新进行BC；下面的策略采样局数只计当前试验参与更新的采样，**不包含继承模型的历史训练成本**。采样可以重复已经训练过的场景，局数不等同于全新世界数量。REINFORCE额外执行同场景greedy baseline，单列该列，不能按与PPO相同局数声称使用了相同仿真预算。

""" + table(["试验", "控制/网络/训练/分布", "完成/计划更新", "BC局", "策略采样局", "额外baseline局", "末更新时长/min", "终止证据"], rows)
    text += "\n" + table(["试验", "训练随机种子", "初始化来源", "实际策略采样场景段（闭区间）", "源码", "CPU workers", "hidden"], configs)
    text += """
末更新时长来自 `elapsed_wall_s`，包括该训练进程当时的等待和保存等开销，不代表GPU独占时间；并发试验不能直接用它推算GPU工作小时。达到更新数只是已完成预设预算，不代表收敛或达到最优。deadline末尾采样可能没有进入更新，这些任务单列在audit中。早期日志没有optimizer_steps字段，不能补造优化器更新次数。

## 两种曲线不能混用

![冷启动训练与固定验证](cold_learning.png)

左图是随机行为策略在不断变化场景上的训练批均值；浅色线为每批32局，深色线为20个更新的不重叠均值。右图才是同一48个场景（6000—6047）上的greedy验证；虚线仅连接已经测过的检查点，不表示中间所有权重都被评估。不同随机种子对应不同初始权重、采样随机性，并可能对应不同训练场景序列，所以没有把它们平均成一条平滑的“算法性能曲线”。

![验证端点与训练成本](endpoints_and_cost.png)

误差条是同48场景相对冻结rollout的配对均值差，bootstrap 10000次、随机种子913、百分位95%区间；它衡量场景样本波动，**不是不同训练随机种子的不确定性**。因为这些验证结果用于反复开发、选择检查点，不能把该区间当成独立最终测试的保证。

![最佳模型继承链与分组概率对照](chain_and_group.png)

左图将cold001及两次PPO微调按真实继承关系连接到累计采样量，终点共40960局（16384+12288+12288），没有BC。图中只列测过的固定验证端点，所用场景段、初始化和学习率见表；这是开发中选出的模型链，不是预先保证单调改善的算法曲线。右图单独保留同种子、同训练场景序列、同512×32采样量的alpha0/1对照；两个早期检查点分别是u184/u190，不能将它们当作同训练量对照。

## 全部已归档RL验证端点

""" + table(["检查点", "平均虚拟秒", "比rollout节省秒 [95% CI]", "胜/负/平", "全清成功", "失败清除", "最坏配对退化秒"], validations)
    text += """
## 当前证据的边界

1. 冷启动001（训练种子9112032）的u213/u512固定验证为3265.859/3228.646秒；第二种子9112033的u202/u497为3409.558/3342.352秒，后者末端对rollout的配对区间仍跨0。不能把第一条好曲线当作所有种子的保证，也不能将更新数不同的两个端点当成严格等预算的随机种子效应估计。
2. 冷启动与BC热启动试验的初始化、训练场景、训练局数及继承成本不同。因此当前跨试验结果不能单独证明“BC导致坏局”或“从零训练必然更好”。BC阶段teacher交叉熵与PPO阶段同名诊断的含义也不同；aux_bc_coef=0时PPO不按该交叉熵学习。
3. v1/v2同起点同更新数的特征对照，以及MLP/attention同起点同更新数的网络对照，比普通跨试验比较更有解释力。但各只有一组训练随机种子，仍不能证明额外几何特征或注意力在所有预算、初始化下无用。
4. 训练总虚拟时间下降可以来自减少远距离来回扫描，不能仅凭全动作熵或teacher交叉熵判断源内探测是否充分探索。历史快照没有每源条件熵日志，新的分组概率版本才记录它；没有对旧日志虚构这种指标。
5. 全清成功来自当前合法动作和保守兜底系统整体。现有记录不支持把全部成功率、全部时间收益归功于神经网络，更没有达到理论下界或不可再优化的证明。未使用官方正式测试，也未查看封存最终测试种子。
6. 从cold001/u512出发，后续PPO384端点3178.339秒，paired REINFORCE374端点3224.109秒。但后者多运行11968条baseline轨迹，且时间截止使两者未完成相同更新数；这组结果不能被写成“在完全相同算力/仿真预算下PPO优于REINFORCE”。
7. 05:10新增的第二次PPO微调u384为3162.245秒，比冻结rollout均值3373.253秒节省211.008秒（6.255%）；同48局41胜7负、全部清除、零失败清除，最坏局仍慢199.794秒。这条继承链累计40960条参与更新的策略轨迹，不能只报告最后一轮12288局。相对上一轮再快16.094秒，不据此断言该增量已经显著。
8. alpha0/alpha1同预算u512分别3294.830/3257.778秒；alpha1相对alpha0节省37.052秒，原归档配对95%区间[-12.447,86.859]秒、24胜24负，额外收益尚未确立。这一区间来自归档比较文件，表中各策略对rollout的区间由本脚本独立bootstrap；两者对象和随机种子不同，不能混写。
9. 06:05完成的base/axis训练每组512×32=16384条，使用同best002权重和seed9112036。base初始化逐局重现best002；axis初始化3134.115秒属于零训练动作集迁移效应。base/axis末端分别3174.712/3149.841秒，两个末端及其余四个中间端点相对各自初始化的增量区间跨0；axis u157例外，节省-56.903秒、区间[-110.221,-5.418]全负，显示该开发集上的早期退化。没有任何已测训练端点确立额外收益，不能误写成所有区间均跨0。base u512最坏局相对rollout多764.699秒，不能只看其均值。中间墙钟检查点的更新数不同，不构成同预算对照。
10. 07:15完成的v3/v4覆盖负债表示对照每组512×32=16384条，两个initialized逐局总时均等于best002，末端3176.419/3160.230秒。v4相对共同parent仅快2.015秒，95%区间[-40.678,44.600]跨0，尚无新特征额外收益的证据。源代码、动作集合、随机种子9112037、新场景180001..196384及完整采样量已核对；中间墙钟端点仍不是相同更新数。强起点attention仍不在本次归档范围。

## 复现

```powershell
..\\cumcm2026-b-interference-localization\\.venv-win\\Scripts\\python.exe research/summarize_rl_training.py --artifacts ../q3-v1-artifacts --output research/rl_training_audit
```

脚本仅需numpy和matplotlib，不需要Torch。`--archives`可显式指定新的完整归档名；不会自动扫描新增最终测试文件。图和表由同一次归档解析生成，精简日志每个更新都有记录，没有只保留表现好的训练批。
"""
    if "matched_group_endpoint" not in report["audited_designs"] or "selected_ppo_chain" not in report["audited_designs"]:
        start, end = text.index("![最佳模型继承链"), text.index("## 全部已归档RL验证端点")
        text = text[:start] + text[end:]
    if not set(DEFAULT_ARCHIVES).issubset({entry["archive"] for entry in report["inputs"]}):
        start, end = text.index("## 当前证据的边界"), text.index("## 复现")
        text = (text[:start] + "## 当前证据的边界\n\n本次显式输入只是历史归档的子集，故不复述需要完整07:15归档才能支持的数值结论。以上表格只列实际读取并通过场景身份及成功条件核验的数据；配对区间不能代表训练种子不确定性，也不能作为全局最优或最终测试保证。\n\n" + text[end:])
    if "matched_probe_endpoint" in report["audited_designs"]:
        design=report["audited_designs"]["matched_probe_endpoint"]
        probe_rows=[]
        for name,delta in design["training_vs_own_initialization"].items():
            low,high=delta["saving_ci95_s"]
            probe_rows.append([name,f"{report['validation'][name]['mean_virtual_time_s']:.3f}",
                f"{delta['mean_seconds_saved']:.3f}",f"[{low:.3f}, {high:.3f}]"])
        zero=design["axis_zero_training_change"]
        low,high=zero["saving_ci95_s"]
        probe_text=("## 动作集初始化与后续学习分开核算\n\n"
            "![初始化与追加PPO训练](probe_initialization_and_learning.png)\n\n"
            f"相同权重下，axis初始化相对base节省{zero['mean_seconds_saved']:.3f}秒，"
            f"本脚本配对95%区间[{low:.3f}, {high:.3f}]秒。这是零训练动作集合改变的效果，不计作PPO学习增量。"
            "下表改用各方法自己的initialized作参照，正值才表示后续训练更快；这些增量区间使用原协议bootstrap seed20260911、10000次Python random.choices，与下方各端点对rollout的历史审计seed913区分。\n\n"
            +table(["训练端点","平均虚拟秒","比自己初始化节省秒","95% CI"],probe_rows)+"\n")
        text=text.replace("## 全部已归档RL验证端点",probe_text+"## 全部已归档RL验证端点")
    if "matched_route_endpoint" in report["audited_designs"]:
        design=report["audited_designs"]["matched_route_endpoint"]
        route_rows=[]
        for name,delta in design["training_vs_parent"].items():
            low,high=delta["saving_ci95_s"]
            route_rows.append([name,f"{report['validation'][name]['mean_virtual_time_s']:.3f}",
                f"{delta['mean_seconds_saved']:.3f}",f"[{low:.3f}, {high:.3f}]"])
        pair=design["final_v4_vs_v3"];low,high=pair["saving_ci95_s"]
        route_text=("## v3/v4覆盖负债表示：共同初始化与额外训练\n\n"
            "两组initialized在48局逐例总时都与parent best002相同。下表正值表示比共同parent更快；采用原协议seed20260911/10000次配对bootstrap。\n\n"
            +table(["训练端点","平均虚拟秒","比共同parent节省秒","95% CI"],route_rows)
            +f"\n相同512更新末端，v4相对v3节省{pair['mean_seconds_saved']:.3f}秒，区间[{low:.3f}, {high:.3f}]。"
            "逐局均成功不等价于改进已经成立。\n\n")
        text=text.replace("## 全部已归档RL验证端点",route_text+"## 全部已归档RL验证端点")
    (output / "README.md").write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("../q3-v1-artifacts"))
    parser.add_argument("--archives", nargs="+", default=list(DEFAULT_ARCHIVES))
    parser.add_argument("--output", type=Path, default=Path("research/rl_training_audit"))
    args = parser.parse_args(argv)
    baseline_path = args.artifacts / "baseline-validation/validation-rollout/rows.json"
    baseline_data = baseline_path.read_bytes()
    baseline = sorted(json.loads(baseline_data), key=lambda row: row["seed"])
    if [r["seed"] for r in baseline] != VALIDATION_SEEDS or not all(r["successful"] for r in baseline):
        raise ValueError("Expected a complete successful frozen rollout validation baseline")
    report = dict(schema=1, analysis_source_sha256=digest(Path(__file__).read_bytes()),
        validation_seeds=VALIDATION_SEEDS, bootstrap=dict(replicates=10000, seed=913, method="paired_percentile"),
        baseline=dict(relative_path=str(baseline_path.relative_to(args.artifacts)), sha256=digest(baseline_data)),
        baseline_mean_s=float(np.mean([r["virtual_time_s"] for r in baseline])),
        inputs=[], trials={}, validation={}, archived_comparisons={}, cross_trial_seed_overlap={})
    seed_sets = {}
    for name in args.archives:
        source, trials, validation, comparisons, seeds = read_archive(args.artifacts/name, baseline)
        if set(trials) & set(report["trials"]):
            raise ValueError("Duplicate trial exports must be resolved explicitly")
        report["inputs"].append(source)
        report["trials"].update(trials)
        report["validation"].update(validation)
        report["archived_comparisons"][name] = comparisons
        seed_sets.update(seeds)
    names = list(seed_sets)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            overlap = seed_sets[first] & seed_sets[second]
            if overlap:
                report["cross_trial_seed_overlap"][first+" vs "+second] = dict(count=len(overlap), ranges=compact_ranges(overlap))
    report["audited_designs"] = audit_selected_designs(report)
    args.output.mkdir(parents=True, exist_ok=True)
    figures(report, args.output)
    write_report(report, args.output)
    curves = {name: run.pop("curve") for name, run in report["trials"].items()}
    curve_data = gzip.compress(json.dumps(curves, separators=(",", ":")).encode(), mtime=0)
    (args.output/"training_curves.json.gz").write_bytes(curve_data)
    report["training_curves"] = dict(file="training_curves.json.gz", sha256=digest(curve_data),
                                    scope="all completed update scalar records; no per-source ledgers")
    (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(trials=len(report["trials"]), validation_endpoints=len(report["validation"]),
        policy_episodes=sum(t["summary"]["policy_episodes_used"] for t in report["trials"].values()),
        paired_baseline_episodes=sum(t["summary"]["paired_baseline_episodes_used"] for t in report["trials"].values()), output=str(args.output))))


if __name__ == "__main__":
    main()

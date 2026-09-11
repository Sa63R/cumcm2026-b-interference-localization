"""Read-only classification of explicit synthetic TRAIN teacher batches.

This evaluates recorded teacher inputs, never counterfactual trajectories.
Set PYTHONPATH to the source tree that provides the selected strict loader.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import collections
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import torch

LOADERS = {"q4_rl.micro_network:load_policy": (13, 50),
           "q4_rl.memory_network:load_policy": (13, 58)}
INDEX_FORMAT = "q4-training-episode-index-v1"


def checked_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError("Expected an explicit SHA256")
    return value.lower()


def gzip_path(path):
    path = Path(path).resolve()
    if not path.name.endswith(".json.gz") or "practice_training.sqlite3" in path.as_posix().lower():
        raise ValueError("Only explicit gzip JSON training evidence is accepted; no database inputs")
    return path


def read_gzip(path, evidence, *, input_number, expected_sha=None):
    path = gzip_path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != checked_hash(expected_sha):
        raise ValueError("Episode evidence SHA256 mismatch")
    evidence.append({"input_number": input_number, "file": path.name,
                     "bytes": len(raw), "sha256": digest})
    return json.loads(gzip.decompress(raw))


def iter_episodes(path, evidence, *, input_number):
    """Legacy lists and immutable EpisodeJournal indices, with no path escape."""
    path = gzip_path(path)
    value = read_gzip(path, evidence, input_number=input_number)
    if isinstance(value, list):
        yield from value
        return
    if not isinstance(value, dict) or value.get("format") != INDEX_FORMAT or not isinstance(value.get("episodes"), list):
        raise ValueError("Unsupported training batch format")
    seen = set()
    for entry in value["episodes"]:
        name = entry["path"]
        if (not isinstance(name, str) or any(c in name for c in "/\\:") or name in seen
                or Path(name).name != name):
            raise ValueError("Invalid or duplicate journal episode path")
        episode = gzip_path(path.with_name(name))
        if episode.parent != path.parent:
            raise ValueError("Journal episode escapes its batch directory")
        rows = read_gzip(episode, evidence, input_number=input_number, expected_sha=entry["sha256"])
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("seed") != entry["seed"]:
            raise ValueError("Journal episode identity mismatch")
        seen.add(name)
        yield rows[0]


def load_frozen(checkpoint, digest, loader):
    digest = checked_hash(digest)
    if loader not in LOADERS:
        raise ValueError("Choose a supported strict G1 or G3 loader")
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix != ".pt":
        raise ValueError("Expected a frozen tensor checkpoint")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("Checkpoint SHA256 mismatch")
    module_name, function = loader.split(":")
    module = importlib.import_module(module_name)
    module.configure_cpu()
    policy = getattr(module, function)(checkpoint, deterministic=True)
    model = policy.model
    if ((model.global_dim, model.candidate_dim) != LOADERS[loader]
            or any(parameter.device.type != "cpu" for parameter in model.parameters())):
        raise ValueError("Loader/model CPU or dimensional contract mismatch")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen checkpoint changed during loading")
    model.eval()
    return model, module


def score_chunk(model, records, tags, groups):
    from q4_rl.network import pack_observations
    with torch.no_grad():
        global_features, features, mask = pack_observations(records, global_dim=model.global_dim,
                                                           candidate_dim=model.candidate_dim)
        logits, _ = model(global_features, features, mask)
        actions = torch.tensor([row["action_index"] for row in records], dtype=torch.long)
        indices = torch.arange(len(records))
        if any(type(row["action_index"]) is not int or not 0 <= row["action_index"] < len(row["candidate_features"]) for row in records):
            raise ValueError("Recorded teacher action is not a legal candidate index")
        predictions, log_probs = logits.argmax(-1), logits.log_softmax(-1)
        teacher_logp = log_probs[indices, actions]
        equivalent = (features == features[indices, actions, None, :]).all(-1) & mask
        multiplicity = equivalent.sum(-1)
        class_probability = (log_probs.exp()*equivalent).sum(-1)
        for j, (kind, role, batch) in enumerate(tags):
            values = {"n": 1, "correct": int(predictions[j] == actions[j]),
                "equivalent_top1": int(equivalent[j, predictions[j]]),
                "cross_entropy_sum": float(-teacher_logp[j]), "teacher_probability_sum": float(teacher_logp[j].exp()),
                "teacher_class_probability_sum": float(class_probability[j]),
                "indistinguishable_teacher_count": int(multiplicity[j] > 1), "multiplicity_sum": int(multiplicity[j]),
                "equal_features_ce_floor_sum": math.log(int(multiplicity[j]))}
            for key in ("all", f"kind:{kind}", f"role:{role}", f"kind_role:{kind}/{role}", f"batch:{batch}"):
                groups[key].update(values)


def summarize(groups):
    result = {}
    for key, sums in groups.items():
        n = sums["n"]
        result[key] = {"records": n, "top1": sums["correct"]/n,
            "float32_feature_equivalent_top1": sums["equivalent_top1"]/n,
            "cross_entropy": sums["cross_entropy_sum"]/n, "mean_teacher_probability": sums["teacher_probability_sum"]/n,
            "mean_teacher_equivalence_class_probability": sums["teacher_class_probability_sum"]/n,
            "fraction_teacher_has_indistinguishable_alternative": sums["indistinguishable_teacher_count"]/n,
            "mean_teacher_class_size": sums["multiplicity_sum"]/n,
            "mean_exact_feature_ce_floor": sums["equal_features_ce_floor_sum"]/n,
            "ce_excess_over_exact_feature_floor": (sums["cross_entropy_sum"]-sums["equal_features_ce_floor_sum"])/n}
    return result


def diagnose(*, batches, checkpoint, checkpoint_sha256, policy_loader, output, minibatch_size=128):
    if minibatch_size < 1 or not batches:
        raise ValueError("Explicit batches and a positive minibatch size are required")
    paths = [gzip_path(path) for path in batches]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate explicit batch path")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite any old evidence.
    started, cpu_started = time.perf_counter(), time.process_time()
    model, module = load_frozen(checkpoint, checkpoint_sha256, policy_loader)
    groups, evidence = collections.defaultdict(collections.Counter), []
    seeds, administrative, teacher_metrics = [], [], []
    for number, path in enumerate(paths, 1):
        records, tags = [], []
        for episode in iter_episodes(path, evidence, input_number=number):
            seed = episode.get("seed")
            if type(seed) is not int or not 8000000 <= seed <= 8099999:
                raise ValueError("Only the isolated synthetic TRAIN seed namespace is accepted")
            if seed in seeds or seed in administrative:
                raise ValueError("Repeated scene across explicit inputs; choose complete attempts explicitly")
            if episode.get("administrative_skip"):
                administrative.append(seed)
                continue
            learning, metrics = episode["controller_learning"], episode["metrics"]
            if metrics.get("split") != "train" or metrics.get("seed") != seed:
                raise ValueError("Episode does not identify synthetic TRAIN evidence")
            if (type(metrics.get("success")) is not bool or
                    any(not math.isfinite(metrics[key]) or metrics[key] < 0 for key in
                        ("actual_time_s", "penalized_time_s", "common_lower_bound_s")) or
                    metrics["common_lower_bound_s"] <= 0):
                raise ValueError("Invalid recorded teacher billing/bound context")
            expected_penalty = metrics["actual_time_s"] if metrics["success"] else max(metrics["actual_time_s"], 360000.)
            if abs(metrics["penalized_time_s"]-expected_penalty) > 2e-5:
                raise ValueError("Recorded teacher penalty accounting mismatch")
            if (episode.get("controller_entrypoint") != module.CONTROLLER_ENTRYPOINT
                    or learning.get("feature_schema") != module.feature_schema()):
                raise ValueError("Recorded observations and selected checkpoint feature/controller schema differ")
            steps = learning["micro_steps"]
            if len(steps) != len(episode["records"]):
                raise ValueError("Teacher action metadata length mismatch")
            seeds.append(seed)
            teacher_metrics.append(metrics)
            for record, step in zip(episode["records"], steps):
                if "log_prob" in record or "value" in record:
                    raise ValueError("PPO sampled actions are not BC teacher labels")
                if record.get("action_kind") != step["kind"]:
                    raise ValueError("Teacher action kind mismatch")
                records.append(record)
                tags.append((step["kind"], step["role"], number))
                if len(records) == minibatch_size:
                    score_chunk(model, records, tags, groups)
                    records, tags = [], []
        if records:
            score_chunk(model, records, tags, groups)
        print(json.dumps({"input_number": number, "classified_episodes_so_far": len(seeds),
                          "records_so_far": groups["all"]["n"]}), flush=True)
    total_t = sum(m["actual_time_s"] for m in teacher_metrics)
    total_penalized = sum(m["penalized_time_s"] for m in teacher_metrics)
    total_l = sum(m["common_lower_bound_s"] for m in teacher_metrics)
    if not teacher_metrics or total_l <= 0:
        raise ValueError("No complete teacher evidence with a positive recorded common bound")
    result = {"scope": "fixed synthetic-TRAIN teacher-input supervised classification; no counterfactual or independent efficacy claim",
        "checkpoint_sha256": checked_hash(checkpoint_sha256), "policy_loader": policy_loader,
        "feature_schema": module.feature_schema(), "network": model.metadata(), "inputs": evidence,
        "episodes": len(seeds), "seeds": seeds, "administrative_episodes": administrative,
        "groups": summarize(groups), "elapsed_s": time.perf_counter()-started,
        "cpu_time_s": time.process_time()-cpu_started,
        "equivalence_definition": f"exact equality of all {model.candidate_dim} float32 candidate inputs within one decision",
        "teacher_rollout_context": {"scope": "original teacher rollouts only; not performance of the classified model",
            "episodes": len(seeds), "all_cleared": sum(m["success"] for m in teacher_metrics),
            "failed_clear_count": sum(m["failed_clear_count"] for m in teacher_metrics),
            "mean_actual_time_s": total_t/len(seeds), "mean_common_lower_bound_s": total_l/len(seeds),
            "mean_penalized_time_s": total_penalized/len(seeds),
            "sum_penalized_time_over_sum_bound": total_penalized/total_l,
            "all_clear_actual_time_over_sum_bound": total_t/total_l if all(m["success"] for m in teacher_metrics) else None}}
    (output/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--policy-loader", choices=LOADERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minibatch-size", type=int, default=128)
    args = vars(parser.parse_args(argv))
    try:
        result = diagnose(**args)
    except Exception as exc:
        print(json.dumps({"diagnostic_failed": True, "error_type": type(exc).__name__}), flush=True)
        return 1
    print(json.dumps({"episodes": result["episodes"], "all": result["groups"]["all"],
                      "elapsed_s": result["elapsed_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the first complete transactions; these are not selected endpoints."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch
from q4_rl.micro_network import configure_cpu, validate_checkpoint as validate_g1
from q4_rl.memory_network import validate_checkpoint as validate_g3

configure_cpu()
EXPECTED = {
    "g1_h128_mc": "2879d49db6343ee8c7ce296e6bd85e52f6f905272298b4c6ab8e16bc5bbe90cf",
    "g1_h128_gae097": "12b9044874ee6ca377d8a97c3670584d6e5d4d6b03791ee6cb48c9933b329f42",
    "g3_h128_mc": "d0be8cd6f1496c7317d2e1bdae234a9397461c7cd24acdf6b8aeebb1129dbb22",
    "g3_h128_gae097": "5ca0ae59df7234a8ffcd416b90996adca3f62db6efebfd55a6f87391597b856a",
}
rows, progress = [], {}
for job, expected in EXPECTED.items():
    path = BASE / job / "checkpoint-000001.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    saved = torch.load(path, map_location="cpu", weights_only=True)
    schema = job.split("_")[0]
    (validate_g1 if schema == "g1" else validate_g3)(saved)
    state, config = saved["state"], saved["config"]
    assert (state["episodes"], state["batches"], state["ppo_batches"], state["warmstart_completed"]) == (16, 1, 1, 0)
    assert state["pending_batch"] is None and state["next_seed"] == 8022016
    assert config["learner_threads"] == 4 and config["random_seed"] == 424555
    assert config.get("gae_lambda", 1.) == (1. if job.endswith("_mc") else .97)
    initial_path = ROOT / "models/initialization" / (schema + "_h128_bc256.pt")
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == config["initialization"]["sha256"]
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    assert all(torch.isfinite(t).all() for t in saved["model"].values())
    changed = sum(not torch.equal(t, initial["model"][name]) for name, t in saved["model"].items())
    assert changed > 0
    item = state["last_progress"]
    assert item["phase"] == "ppo" and item["batch"] == 1 and item["episodes"] == 16
    steps = {int(s["step"].item()) for s in saved["optimizer"]["state"].values()}
    assert steps == {item["update"]["updates"]}
    progress[job] = item
    rows.append(dict(job=job, checkpoint_sha256=expected, episodes=16, ppo_batches=1,
        changed_parameter_tensors=changed, optimizer_steps=item["update"]["updates"],
        initialization_sha256=config["initialization"]["sha256"],
        actor_lambda=config.get("gae_lambda", 1.), pending_batch=None))
for schema in ("g1", "g3"):
    first, second = progress[schema + "_h128_mc"], progress[schema + "_h128_gae097"]
    # Same initial policy, seed stream and training cases. This checks recorded
    # first-batch outcomes, not identity of every internal trajectory byte.
    for key in ("case_ratios", "full_clear", "failed_clear_count", "mean_actual_time_s"):
        assert first[key] == second[key], (schema, key)
result = dict(all_four_first_transactions_verified=True, models=rows,
    first_batch_outcomes_match_within_schema=True,
    scope="Startup/update integrity only; no policy efficacy or endpoint selection",
    task="q4-rl-gae-v5-20260912", training_source_commit="df36f0ce",
    supervisor_pid=1725668, wrapper_pid=1725669,
    learner_pids=[1725677, 1725678, 1725679, 1725680],
    observed_related_thread_affinities="66 threads all CPU0-41; includes idle/runtime helper threads")
output = BASE / "VERIFIED.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({"verified_models": len(rows), "first_batch_outcomes_match_within_schema": True}))

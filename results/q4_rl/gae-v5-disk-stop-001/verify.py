"""Verify preserved complete updates and administrative resume checkpoints."""
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
    "g1_h128_mc": (13, "2870173ae856d8ed85963612a36bb66598d73dae6f33cecf45f6831f64fbbea6", "ba253fe13d5b9ea98b9db8d96d38f1c65d9b3c26c9df83a59ff802de635ca657"),
    "g1_h128_gae097": (13, "f236acabd5e62bbcd2ed80dbf3588d35913eaa830005df9702a4d66aa604ca49", "e8693bb863cf559ec4299255a9de630b07b2b47cec3edd01dee0d64be7c489ab"),
    "g3_h128_mc": (12, "46022d77b009dcf550ff7a385cedfddb67853c1da6d168513d3a2414713d4fb9", "ce4b39f779b12bfee48accb25d23c6e8c87edd3bb567f16c05445c35e05be215"),
    "g3_h128_gae097": (13, "ff1c306182684392772f10896cc4202252aaa745ee0e64ccfdbf9bd2c1eb7f1c", "85569cdd5682c90cf1752e1d5a9b0d206f9e9acd0e7902c1693221ef883fa54b"),
}

def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right

rows = []
for job, (batches, complete_sha, latest_sha) in EXPECTED.items():
    loaded = []
    for filename, expected in (("checkpoint.pt", complete_sha), ("latest.pt", latest_sha)):
        path = BASE / job / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        saved = torch.load(path, map_location="cpu", weights_only=True)
        (validate_g1 if job.startswith("g1_") else validate_g3)(saved)
        assert saved["state"]["batches"] == saved["state"]["ppo_batches"] == batches
        assert saved["state"]["episodes"] == 16 * batches
        assert saved["state"]["warmstart_completed"] == 0
        assert saved["config"].get("gae_lambda", 1.) == (1. if job.endswith("_mc") else .97)
        assert saved["config"]["learner_threads"] == 4
        assert all(torch.isfinite(t).all() for t in saved["model"].values())
        loaded.append(saved)
    complete, latest = loaded
    equal_tree(complete["model"], latest["model"])
    equal_tree(complete["optimizer"], latest["optimizer"])
    equal_tree(complete["config"], latest["config"])
    assert complete["state"]["pending_batch"] is None
    pending = latest["state"]["pending_batch"]
    assert pending["mode"] == "ppo"
    assert pending["seeds"] == list(range(8022000 + 16 * batches, 8022000 + 16 * (batches + 1)))
    assert len(pending["action_seeds"]) == 16
    assert latest["state"]["next_seed"] == 8022000 + 16 * (batches + 1)
    status_path = BASE / job / "status.json"
    status = json.loads(status_path.read_bytes())
    equal_tree(status, latest["state"])
    rows.append(dict(job=job, completed_batches=batches, trained_episodes=16 * batches,
        checkpoint_sha256=complete_sha, resume_sha256=latest_sha,
        status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(),
        model_and_adam_equal_to_last_complete=True, reserved_unlearned_episodes=16,
        stop_reason=status["stop_reason"], cumulative_wall_s=status["wall_time_s"]))

result = dict(scope="Checkpoint transaction integrity after disk guard; not policy efficacy or a completed 32-batch experiment",
    source_commit="df36f0ce43a8b1e23d6ebf1cb1b0bba6f16ef01d",
    task="q4-rl-gae-v5-20260912", all_verified=True, models=rows,
    unequal_update_exposure=True, resume_requires_original_pending_journals=True,
    training_raw_not_fully_downloaded_in_this_audit=True)
with (BASE / "VERIFIED.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({"verified_pairs": len(rows), "complete_batches": [r["completed_batches"] for r in rows],
                  "latest_model_and_adam_unchanged": True, "pending_preserved": True}))

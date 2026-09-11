"""Offline resume checks. No rollout, optimizer update, server or transfer."""
import copy
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
spec = importlib.util.spec_from_file_location("gae_resume_sidecar", BASE / "resume.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)
from q4_rl.micro_network import configure_cpu
configure_cpu()
NOW = datetime.fromisoformat(resume.DEADLINE).timestamp() - 60
CONFIG = json.loads((ROOT / "results/q4_rl/gae-v5-deployment-001/train_gae_v5.json").read_bytes())
STOP = ROOT / "results/q4_rl/gae-v5-disk-stop-001"


@pytest.fixture(scope="module")
def saved():
    result = {}
    for job in CONFIG["jobs"]:
        name = job["name"]
        raw = resume.exact_bytes(STOP / name / "latest.pt", resume.EXPECTED[name][1])
        import io
        result[name] = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    return result


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_exact_original_config_and_all_four_resume_entry_configurations(saved):
    path = ROOT / "results/q4_rl/gae-v5-deployment-001/train_gae_v5.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == resume.CONFIG_SHA
    expected_remaining = [2763, 2776, 2757, 2745]
    jobs = []
    for source, remaining in zip(CONFIG["jobs"], expected_remaining):
        prior = copy.deepcopy(saved[source["name"]])
        job = resume.resume_job(source, prior, ROOT, NOW)
        flags = resume.flag_map(job["argv"])
        assert int(flags["--max-wall-seconds"]) == remaining
        assert flags["--resume"] == flags["--output"] + "/latest.pt"
        assert flags["--max-batches"] == "32"
        assert flags["--scenario-start"] == "8022000"
        assert ("--initialize-micro-warmstart" in flags) == source["name"].startswith("g1")
        assert "--initialize-memory-warmstart" not in flags
        equal(prior, saved[source["name"]])
        jobs.append(job)
    assert resume.validate_budget(jobs)["combined_compute"] == 47
    with pytest.raises(ValueError, match="exceeds 50"):
        resume.validate_budget(jobs, other_compute=15)


@pytest.mark.parametrize("wall", [float("nan"), float("inf"), -1, 3600, 3600.1])
def test_wall_budget_never_resets(wall):
    with pytest.raises(ValueError):
        resume.remaining_seconds(wall, NOW)


def test_deadline_is_original_and_not_extended():
    with pytest.raises(ValueError, match="expired"):
        resume.remaining_seconds(800, NOW + 60)
    assert resume.remaining_seconds(800.2, NOW) == 2799


@pytest.mark.parametrize("change", ["module", "output", "lambda", "learning_rate", "pending", "cursor"])
def test_changed_resume_contract_rejected(saved, change):
    job = copy.deepcopy(CONFIG["jobs"][0])
    state = copy.deepcopy(saved[job["name"]])
    if change == "module":
        job["module"] = "q4_rl.memory_train"
    elif change in ("output", "lambda"):
        flag = "--output" if change == "output" else "--gae-lambda"
        job["argv"][job["argv"].index(flag) + 1] = "runs/other/job/training" if change == "output" else ".97"
    elif change == "learning_rate":
        state["config"]["learning_rate"] = .1
    elif change == "pending":
        state["state"]["pending_batch"] = None
    else:
        state["state"]["next_seed"] += 16
    with pytest.raises(ValueError):
        resume.resume_job(job, state, ROOT, NOW)


def test_path_sha_and_attempt_collisions(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        resume.inside(tmp_path, "../other/latest.pt")
    artifact = tmp_path / "latest.pt"
    artifact.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256"):
        resume.exact_bytes(artifact, resume.EXPECTED["g1_h128_mc"][1])
    import gzip
    index = {"format": "q4-training-episode-index-v1", "episodes": []}
    with gzip.open(tmp_path / "committed.json.gz", "wb") as stream:
        stream.write(json.dumps(index).encode())
    state = {"last_progress": {"raw_attempt": "committed.json.gz"}, "batches": 13, "next_attempt": 14}
    resume.check_old_journals(tmp_path, state)  # Missing zero-result pending index is allowed.
    (tmp_path / "batch-000013-attempt-000014-episode-0000.json.gz").write_bytes(b"retained")
    with pytest.raises(ValueError, match="next attempt"):
        resume.check_old_journals(tmp_path, state)


@pytest.mark.parametrize("group", ["g1", "g3"])
def test_actual_main_restores_latest_model_adam_rng_without_rollout(saved, tmp_path, monkeypatch, group):
    source = next(j for j in CONFIG["jobs"] if j["name"] == group + "_h128_mc")
    prior = saved[source["name"]]
    trainer = importlib.import_module(source["module"])
    flags = resume.flag_map(resume.resume_job(source, prior, ROOT, NOW)["argv"])
    latest = tmp_path / "latest.pt"
    latest.write_bytes((STOP / source["name"] / "latest.pt").read_bytes())
    flags.update({"--output": str(tmp_path), "--resume": str(latest), "--workers": "1",
        "--max-batches": str(prior["state"]["batches"])})  # Execute entry/restore only; loop cannot run.
    if group == "g1":
        flags["--initialize-micro-warmstart"] = str(ROOT / "models/initialization/g1_h128_bc256.pt")
    calls = []
    def capture(path, model, optimizer, state, config):
        equal(model.state_dict(), prior["model"])
        equal(optimizer.state_dict(), prior["optimizer"])
        equal(config, prior["config"])
        equal(torch.get_rng_state(), prior["rng"]["torch"])
        equal(random.getstate(), prior["rng"]["python"])
        equal(state["pending_batch"], prior["state"]["pending_batch"])
        calls.append(str(path))
    monkeypatch.setattr(trainer, "save_checkpoint", capture)
    monkeypatch.setattr(trainer, "_write_json", lambda *a, **k: None)
    monkeypatch.setattr(trainer.time, "time", lambda: NOW)
    monkeypatch.setattr(trainer, "rollout", lambda *a, **k: pytest.fail("rollout forbidden in offline resume test"))
    init_calls = []
    if group == "g1":
        original = trainer.load_micro_warmstart
        def init_check(*a, **k):
            initial = original(*a, **k)
            assert any(not torch.equal(initial["model"][key], prior["model"][key]) for key in prior["model"])
            init_calls.append(True)
            return initial
        monkeypatch.setattr(trainer, "load_micro_warmstart", init_check)
    result = trainer.main([v for pair in flags.items() for v in pair])
    assert calls and result["batches"] == prior["state"]["batches"]
    assert len(init_calls) == (1 if group == "g1" else 0)
    assert hashlib.sha256(latest.read_bytes()).hexdigest() == resume.EXPECTED[source["name"]][1]


def test_g1_omitted_initialization_binding_rejects_resume(saved):
    source = CONFIG["jobs"][0]
    job = resume.resume_job(source, saved[source["name"]], ROOT, NOW)
    flags = resume.flag_map(job["argv"])
    del flags["--initialize-micro-warmstart"]
    del flags["--initialize-sha256"]
    trainer = importlib.import_module(source["module"])
    _, args = trainer._arguments([v for pair in flags.items() for v in pair])
    assert trainer.configuration(args) != saved[source["name"]]["config"]


def test_g3_init_plus_resume_cli_rejected():
    source = CONFIG["jobs"][2]
    trainer = importlib.import_module(source["module"])
    with pytest.raises(SystemExit):
        trainer._arguments(source["argv"] + ["--resume", "runs/train-gae-v5/g3_h128_mc/training/latest.pt"])


def test_fresh_supervisor_directory_and_exact_budget_command(tmp_path):
    run = resume.fresh_run(tmp_path)
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        resume.fresh_run(tmp_path)
    command = resume.supervisor_command(tmp_path, tmp_path / "resume.py")
    assert command[command.index("--cpu-budget") + 1] == "50"
    assert command[command.index("--deadline") + 1] == resume.DEADLINE
    assert command[command.index("--minimum-free-gib") + 1] == "20"
    assert command[-1] == "--worker"


def test_original_release_artifacts_and_changed_source_rejection(tmp_path):
    import shutil
    import tarfile
    deployment = ROOT / "results/q4_rl/gae-v5-deployment-001"
    archive = deployment / "q4-gae-v5-source-20260912-r2.tar.gz"
    resume.exact_bytes(archive, resume.ARCHIVE_SHA)
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            assert member.isfile()
            destination = resume.inside(tmp_path, member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(stream.extractfile(member).read())
    copies = [(archive, "launch/train-gae-v5/" + archive.name),
        (deployment / "train_gae_v5.json", "research/q4_rl/train_gae_v5.json")]
    for group in ("g1", "g3"):
        relative = f"models/initialization/{group}_h128_bc256.pt"
        copies.append((ROOT / relative, relative))
    for source, relative in copies:
        target = resume.inside(tmp_path, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    assert resume.verify_source(tmp_path) == CONFIG
    with (tmp_path / "src/q4_rl/micro_train.py").open("ab") as stream:
        stream.write(b"\n# changed only in disposable test fixture\n")
    with pytest.raises(ValueError, match="SHA256"):
        resume.verify_source(tmp_path)

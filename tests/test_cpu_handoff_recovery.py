"""Short regression tests for interrupted CPU delivery runs; no training/network."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name+"_recovery", ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


handoff = load_script("cpu_handoff")
builder = load_script("build_cpu_handoff")


@pytest.fixture
def status_args():
    return dict(scenario=1000001, scenario_end=1199999, max_attempted=199936,
                max_updates=1562, trial_seconds=3600, deadline_epoch=100000, now=1000)


def state(**changes):
    return dict(update=10, optimizer_steps=40, attempted_episodes=1280,
                next_seed=1001281, elapsed_training_s=100, **changes)


def test_dirty_package_cannot_be_run_as_training():
    with pytest.raises(ValueError, match="restricted to --smoke"):
        handoff.enforce_package_mode({"dirty_source_smoke_only": True}, False)
    handoff.enforce_package_mode({"dirty_source_smoke_only": True}, True)
    handoff.enforce_package_mode({"dirty_source_smoke_only": False}, False)


def test_trial_bounds_and_absolute_deadline_are_passed(tmp_path):
    args = handoff.training_command(tmp_path, seed=9112101, scenario=1000001,
        scenario_end=1199999, max_attempted_episodes=199936,
        workers=16, threads=4, episodes=128, updates=1562, seconds=60,
        checkpoint=tmp_path/"parent.pt", deadline_epoch=100000)
    assert args[args.index("--scenario-end")+1] == "1199999"
    assert args[args.index("--max-attempted-episodes")+1] == "199936"
    assert handoff.datetime.fromisoformat(args[args.index("--deadline-utc")+1]).timestamp() == 100000


def test_atomic_gzip_keeps_last_complete_case_on_interruption(tmp_path, monkeypatch):
    case = tmp_path/"case.json.gz"
    handoff.write_gzip_json(case, {"completed": 1})
    original = case.read_bytes()

    def interrupted(value, stream, **kwargs):
        stream.write('{"partial":')
        raise KeyboardInterrupt()

    monkeypatch.setattr(handoff.json, "dump", interrupted)
    with pytest.raises(KeyboardInterrupt):
        handoff.write_gzip_json(case, {"completed": 2})
    assert case.read_bytes() == original
    with gzip.open(case, "rt") as stream:
        assert json.load(stream) == {"completed": 1}


def test_interrupted_collection_cannot_resume_into_another_trial(status_args):
    checkpoint = state()
    checkpoint.update(update=1561, attempted_episodes=199936, next_seed=1199937)
    result = handoff.trial_status(checkpoint, **status_args)
    assert result["training_budget_complete"]
    assert not result["can_resume"]
    assert result["stop_reason"] == "attempted_episode_limit"


def test_out_of_partition_checkpoint_is_invalid(status_args):
    checkpoint = state()
    checkpoint.update(attempted_episodes=200064, next_seed=1200065)
    result = handoff.trial_status(checkpoint, **status_args)
    assert result["status"] == "invalid"
    assert not result["training_budget_complete"] and not result["can_resume"]


def test_completion_is_rebuilt_without_a_finished_file(status_args):
    checkpoint = state()
    checkpoint.update(update=1562, attempted_episodes=199936, next_seed=1199937)
    result = handoff.trial_status(checkpoint, **status_args)
    assert result["training_budget_complete"]
    assert result["stop_reason"] == "update_limit"


def test_zero_optimizer_run_is_not_complete(status_args):
    checkpoint = state()
    checkpoint.update(update=0, optimizer_steps=0, elapsed_training_s=3601)
    result = handoff.trial_status(checkpoint, **status_args)
    assert not result["training_budget_complete"]
    assert not result["can_resume"]
    assert result["stop_reason"] == "no_optimizer_progress"


def test_global_cutoff_does_not_count_as_completed_trial(status_args):
    checkpoint = state(stop_reason="global_deadline")
    result = handoff.trial_status(checkpoint, **status_args)
    assert not result["training_budget_complete"]
    assert not result["can_resume"]
    assert result["stop_reason"] == "global_deadline"


def test_elapsed_trial_budget_is_reconstructed(status_args):
    checkpoint = state()
    checkpoint["elapsed_training_s"] = 3600.1
    assert handoff.trial_status(checkpoint, **status_args)["training_budget_complete"]


def test_failed_process_cannot_be_reported_as_complete(status_args):
    checkpoint = state()
    checkpoint["elapsed_training_s"] = 3600.1
    result = handoff.trial_status(checkpoint, returncode=124, **status_args)
    assert not result["training_budget_complete"]
    assert result["stop_reason"] == "process_timeout"


@pytest.mark.parametrize("info", [
    {"available_cpu_slots": 8, "available_memory_gib": 128},
    {"available_cpu_slots": 96, "available_memory_gib": 15},
])
def test_cached_workers_cannot_exceed_current_resources(info):
    cached = {"selected_workers": 96, "learner_threads": 4, "episodes_per_update": 128}
    with pytest.raises(ValueError, match="Cached benchmark"):
        handoff.validate_cached_benchmark(cached, info, 96, False)


def test_too_little_memory_does_not_force_one_worker():
    assert handoff.choose_worker_counts({"available_cpu_slots": 96, "available_memory_gib": 2}) == []


def test_timeout_stops_descendants_after_the_leader_exits(tmp_path, monkeypatch):
    signals, launches = [], []
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)

    class Process:
        pid = 12345

        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("learner", timeout)
            return -15  # Parent exits on TERM; descendants can still remain.

    def launch(*args, **kwargs):
        launches.append(kwargs)
        return Process()

    monkeypatch.setattr(handoff, "os", SimpleNamespace(name="posix", environ={},
                        killpg=lambda pid, sig: signals.append((pid, sig))))
    monkeypatch.setattr(handoff.subprocess, "Popen", launch)
    code = handoff.run_training(["learner"], tmp_path/"learner.log", deadline_epoch=time.time()+1)
    assert code == 124 and launches[0]["start_new_session"] is True
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_package_inventory_and_dirty_check_cover_every_bundled_input(tmp_path, monkeypatch):
    names = ["scripts/build_cpu_handoff.py", "scripts/cpu_handoff.py", "scripts/object_exchange.py",
             "scripts/START_CPU.sh", "scripts/sync_cpu_results.py", "scripts/RUN_WITH_SYNC.sh",
             "research/cpu_v2_protocol.json", "research/v1_protocol.json",
             "research/cpu_v2/README_OPERATOR.md", "research/cpu_v2/extra.md", "pyproject.toml",
             "experiments/__init__.py", "experiments/research_v1_eval.py",
             "experiments/run_q3_comparison.py", "src/package/__init__.py"]
    for name in names:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test\r\n")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    calls = []

    def git_output(*args):
        calls.append(args)
        if args[0] == "status":
            return " M experiments/research_v1_eval.py\n"
        if args[0] == "ls-files":
            return "\0".join(names)+"\0"
        return "abc123\n"

    monkeypatch.setattr(builder, "git_output", git_output)
    checkpoint = tmp_path/"parent.pt"
    checkpoint.write_bytes(b"frozen-parent")
    monkeypatch.setattr(builder, "PARENT_SHA", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Commit the source"):
        builder.build(checkpoint, tmp_path/"blocked.tar.gz")
    assert set(calls[0][calls[0].index("--")+1:]) == set(names)
    archive = tmp_path/"smoke.tar.gz"
    builder.build(checkpoint, archive, allow_dirty=True)
    with tarfile.open(archive) as stream:
        manifest = json.load(stream.extractfile("q3-cpu-handoff/PACKAGE_MANIFEST.json"))
        assert manifest["dirty_source_smoke_only"]
        assert set(manifest["source_paths"]) == set(names)
        assert stream.extractfile("q3-cpu-handoff/RUN_WITH_SYNC.sh").read() == b"test\n"
        assert "scripts/sync_cpu_results.py" in manifest["files"]
        assert "research/cpu_v2/extra.md" in manifest["files"]


def test_ignored_untracked_sources_are_dirty(tmp_path, monkeypatch):
    source = tmp_path/"src/ignored.py"
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "git_output", lambda *args: "")
    dirty, untracked = builder.source_status([source])
    assert not dirty and untracked == ["src/ignored.py"]


def test_interrupt_records_incomplete_status_and_exports(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    (tmp_path/"research").mkdir()
    (tmp_path/"PACKAGE_MANIFEST.json").write_text("{}")
    (tmp_path/"research/cpu_v2_protocol.json").write_text("{}")
    output = tmp_path/"cpu_runs/run01"
    deadline = handoff.datetime.fromtimestamp(time.time()+3600, handoff.timezone.utc).isoformat()
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "verify_package", lambda: {"dirty_source_smoke_only": False})
    monkeypatch.setattr(handoff, "hardware", lambda: {"available_cpu_slots": 4, "available_memory_gib": 8})
    monkeypatch.setattr(sys, "argv", ["cpu_handoff.py", "run", "--smoke", "--output", str(output),
                                     "--deadline", deadline])

    def interrupted(*args):
        handoff.request_shutdown(signal.SIGTERM, None)

    monkeypatch.setattr(handoff, "benchmark", interrupted)
    exports = []
    monkeypatch.setattr(handoff, "export_results", lambda path: exports.append(path))
    previous = signal.getsignal(signal.SIGTERM)
    assert handoff.main() == 130
    assert signal.getsignal(signal.SIGTERM) == previous
    assert exports == [output]
    summary = json.loads((output/"summary.json").read_text())
    assert not summary["training_complete"] and not summary["evaluation_complete"]
    assert "KeyboardInterrupt" in summary["error"]

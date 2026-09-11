"""Hard sampling limits remain valid across an actual abrupt process exit."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from research_rl import train
from research_rl.controller import CONTEXT_DIM, FEATURE_DIMS


def arguments(output, **changes):
    options = dict(output=str(output), device="cpu", hidden=16, bc_episodes=0,
                   updates=20, episodes_per_update=3, workers=0, epochs=1,
                   max_wall_s=60, scenario_start=1000001, scenario_end=1000010,
                   max_attempted_episodes=5)
    options.update(changes)
    return [item for key, value in options.items() for item in ("--" + key.replace("_", "-"), str(value))]


def checkpoint(output):
    return torch.load(output / "latest.pt", map_location="cpu", weights_only=False)


def stub_episodes(monkeypatch):
    seen = []

    def episode(task):
        seen.append(task[0])
        return [dict(features=[[0.0] * FEATURE_DIMS["v2"], [0.1] * FEATURE_DIMS["v2"]],
                     context=[0.0] * CONTEXT_DIM, action=0, teacher=0,
                     log_prob=-0.693147, value=0.0, reward=-0.1)], dict(
                         seed=task[0], virtual_time_s=100.0, success=True)

    monkeypatch.setattr(train, "episode", episode)
    return seen


def test_attempt_budget_truncates_final_batch_and_survives_resume(tmp_path, monkeypatch):
    seen = stub_episodes(monkeypatch)
    args = arguments(tmp_path)
    assert train.main(args) == 0
    state = checkpoint(tmp_path)["state"]
    assert seen == list(range(1000001, 1000006))
    assert (state["attempted_episodes"], state["episodes"], state["update"]) == (5, 5, 2)
    assert state["stop_reason"] == "attempted_episode_limit"
    assert train.main(args + ["--resume", str(tmp_path / "latest.pt")]) == 0
    assert seen == list(range(1000001, 1000006))


def test_scenario_end_is_inclusive_and_never_recycles(tmp_path, monkeypatch):
    seen = stub_episodes(monkeypatch)
    args = arguments(tmp_path, scenario_start=1199997, scenario_end=1199999,
                     max_attempted_episodes=10, episodes_per_update=2)
    assert train.main(args) == 0
    assert seen == [1199997, 1199998, 1199999]
    state = checkpoint(tmp_path)["state"]
    assert state["next_seed"] == 1200000
    assert state["stop_reason"] == "scenario_range_exhausted"
    assert train.main(args + ["--resume", str(tmp_path / "latest.pt")]) == 0
    assert seen == [1199997, 1199998, 1199999]


def test_abrupt_exit_reserves_batch_before_dispatch_and_resume_cannot_cross_partition(tmp_path, monkeypatch):
    args = arguments(tmp_path, scenario_end=1000004, max_attempted_episodes=4)
    code = ("import os; from research_rl import train; "
            "train.episode = lambda task: os._exit(17); "
            f"train.main({args!r})")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(train.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 17, result.stderr
    state = checkpoint(tmp_path)["state"]
    assert (state["attempted_episodes"], state["episodes"], state["next_seed"]) == (3, 0, 1000004)
    seen = stub_episodes(monkeypatch)
    assert train.main(args + ["--resume", str(tmp_path / "latest.pt")]) == 0
    assert seen == [1000004]
    resumed = checkpoint(tmp_path)["state"]
    assert (resumed["attempted_episodes"], resumed["episodes"], resumed["next_seed"]) == (4, 1, 1000005)
    assert resumed["stop_reason"] == "attempted_episode_limit"


@pytest.mark.parametrize("artifact,transfer,resume_latest", [
    ("random.pt", False, False), ("random.pt", True, True), ("initialized.pt", True, False)])
def test_initial_checkpoint_window_recovers_without_replacing_artifacts(tmp_path, monkeypatch, artifact, transfer, resume_latest):
    original_save = train.save_checkpoint
    args = arguments(tmp_path / "trial", updates=0)
    if transfer:
        parent = tmp_path / "parent"
        train.main(arguments(parent, updates=0))
        args += ["--initialize-from", str(parent / "latest.pt")]
    output = tmp_path / "trial"

    def interrupt_after_artifact(path, *values):
        result = original_save(path, *values)
        if path.name == artifact:
            raise KeyboardInterrupt("simulated kill before latest")
        return result

    monkeypatch.setattr(train, "save_checkpoint", interrupt_after_artifact)
    with pytest.raises(KeyboardInterrupt):
        train.main(args)
    assert not (output / "latest.pt").exists()
    original = hashlib.sha256((output / artifact).read_bytes()).hexdigest()
    monkeypatch.setattr(train, "save_checkpoint", original_save)
    if resume_latest:
        index = args.index("--initialize-from")
        del args[index:index + 2]
        args += ["--resume", str(output / "latest.pt")]
    assert train.main(args) == 0
    assert hashlib.sha256((output / artifact).read_bytes()).hexdigest() == original
    state = checkpoint(output)["state"]
    assert state["attempted_episodes"] == 0
    assert state["stop_reason"] == "update_limit"
    assert ("initialization" in state) == transfer


def test_unknown_nonempty_directory_is_preserved(tmp_path, monkeypatch):
    stub_episodes(monkeypatch)
    args = arguments(tmp_path, updates=0)
    train.main(args)
    (tmp_path / "latest.pt").unlink()
    # Even a valid random checkpoint does not authorize replacing other data.
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(SystemExit):
        train.main(args)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_resume_recovers_legacy_inflight_cursor_and_keeps_saved_caps(tmp_path, monkeypatch):
    seen = stub_episodes(monkeypatch)
    args = arguments(tmp_path, updates=0)
    train.main(args)
    payload = checkpoint(tmp_path)
    payload["state"].pop("attempted_episodes")
    payload["state"].update(next_seed=1000004, episodes=0)
    torch.save(payload, tmp_path / "latest.pt")
    resume_args = arguments(tmp_path)
    for option in ("--scenario-end", "--max-attempted-episodes"):
        at = resume_args.index(option)
        del resume_args[at:at + 2]
    assert train.main(resume_args + ["--resume", str(tmp_path / "latest.pt")]) == 0
    assert seen == [1000004, 1000005]
    assert checkpoint(tmp_path)["state"]["attempted_episodes"] == 5


def test_resume_cannot_enlarge_saved_partition_or_budget(tmp_path):
    train.main(arguments(tmp_path, updates=0))
    before = (tmp_path / "latest.pt").read_bytes()
    for option, value in (("scenario_end", 1200001), ("max_attempted_episodes", 6)):
        with pytest.raises(SystemExit):
            train.main(arguments(tmp_path, **{option: value}) + ["--resume", str(tmp_path / "latest.pt")])
        assert (tmp_path / "latest.pt").read_bytes() == before


def test_invalid_cross_range_is_rejected_before_creating_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        train.main(arguments(output, scenario_start=5000, scenario_end=100002))
    assert not output.exists()


def test_zero_attempt_budget_writes_restartable_initial_checkpoint(tmp_path, monkeypatch):
    seen = stub_episodes(monkeypatch)
    assert train.main(arguments(tmp_path, max_attempted_episodes=0)) == 0
    assert seen == []
    assert checkpoint(tmp_path)["state"]["stop_reason"] == "attempted_episode_limit"

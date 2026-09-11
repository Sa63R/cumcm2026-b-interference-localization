"""Profile preflight tests use synthetic files and never connect to a simulator."""

import json
import sys

import pytest

from practice_control import branch_worker as worker


def profile_fixture(tmp_path, monkeypatch, *, entrypoint="strategies:run_search"):
    delivery = tmp_path / "delivery"
    restored = tmp_path / "restored"
    config = delivery / "configs/baseline"
    source = restored / "baseline"
    config.mkdir(parents=True)
    (source / "src/strategies").mkdir(parents=True)
    code = b"# frozen fixture\n"
    (source / "src/strategies/__init__.py").write_bytes(code)
    spec = {"name": "fixture", "entrypoint": entrypoint, "kwargs": {"variant": "rollout"}}
    freeze = {"git_commit": "fixture-commit", "identity": {
        "spec_sha256": worker.canonical_digest(spec),
        "source_sha256": {"src/strategies/__init__.py": worker.digest(code)},
        "checkpoint_sha256": {},
    }}
    freeze_bytes = json.dumps(freeze).encode()
    (config / "spec.json").write_text(json.dumps(spec))
    (config / "freeze.json").write_bytes(freeze_bytes)
    monkeypatch.setitem(worker.PROFILES, "baseline", {
        "entrypoint": "strategies:run_search", "commit": "fixture-commit",
        "freeze_sha256": worker.digest(freeze_bytes),
    })
    return delivery, restored, source, config


def test_profile_preflight_verifies_without_importing_solver(tmp_path, monkeypatch):
    delivery, restored, source, _ = profile_fixture(tmp_path, monkeypatch)
    before = set(sys.modules)
    actual_source, spec, kwargs, metadata = worker.verify_profile("baseline", delivery, restored)
    assert actual_source == source
    assert kwargs == {"variant": "rollout"}
    assert metadata["source_commit"] == "fixture-commit"
    assert metadata["version_scope"] == "frozen_v1_branch"
    assert not {name for name in set(sys.modules) - before if name.startswith("strategies")}


@pytest.mark.parametrize("change", ["code", "spec", "freeze", "extra_source"])
def test_modified_frozen_artifacts_fail_before_any_import(tmp_path, monkeypatch, change):
    delivery, restored, source, config = profile_fixture(tmp_path, monkeypatch)
    if change == "code":
        (source / "src/strategies/__init__.py").write_text("changed")
    elif change == "spec":
        spec = json.loads((config / "spec.json").read_text())
        spec["kwargs"]["variant"] = "efficient"
        (config / "spec.json").write_text(json.dumps(spec))
    elif change == "freeze":
        (config / "freeze.json").write_text("{}")
    else:
        (source / "src/strategies/extra.py").write_text("# added")
    with pytest.raises(ValueError):
        worker.verify_profile("baseline", delivery, restored)


def test_arbitrary_entrypoint_is_rejected_even_with_matching_local_hashes(tmp_path, monkeypatch):
    delivery, restored, _, _ = profile_fixture(tmp_path, monkeypatch, entrypoint="other:execute")
    with pytest.raises(ValueError, match="entrypoint"):
        worker.verify_profile("baseline", delivery, restored)


def test_adapter_preserves_frozen_parameters_instead_of_runner_placeholder():
    calls = []
    sentinel = object()
    def callback(client, **kwargs):
        calls.append((client, kwargs))
        return sentinel
    solve = worker._solver_adapter(callback, {"variant": "rollout", "rollout_config": {"particles": 4}})
    assert solve("guarded-client", problem=3, variant="baseline", max_actions=20000) is sentinel
    assert calls == [("guarded-client", {"problem": 3, "max_actions": 20000,
                                       "variant": "rollout", "rollout_config": {"particles": 4}})]
    with pytest.raises(ValueError, match="Q3"):
        solve("guarded-client", problem=4, variant="baseline", max_actions=20000)
    assert len(calls) == 1


def test_worker_rejects_import_contamination_before_changing_paths(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "strategies", object())
    before = list(sys.path)
    with pytest.raises(ValueError, match="fresh worker"):
        worker.load_solver(tmp_path, {"entrypoint": "strategies:run_search"}, {})
    assert sys.path == before

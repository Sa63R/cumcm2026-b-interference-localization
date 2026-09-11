"""R8 qualification and practice ownership gates; all network paths are disabled."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import zipfile

import pytest

from experiments import run_q4_clear_before_probe_practice as entry
from practice_control import BridgeError
from simulator_client import SimulatorClient


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Network forbidden in offline launcher tests")
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(SimulatorClient, "_exchange", fail)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    evaluator = root / "experiments/evaluate_q4_clear_before_probe.py"
    evaluator.parent.mkdir()
    evaluator.write_bytes(b"inert evaluator fixture")
    source_bytes = b"frozen source fixture"
    source = {"src/frozen.py": hashlib.sha256(source_bytes).hexdigest()}
    selection = {"selected": entry.SELECTED, "source_sha256": dict(source),
        "specs": {entry.SELECTED: copy.deepcopy(entry.SPEC)},
        "reserved_seeds": copy.deepcopy(entry.STAGES), "evaluator_sha256": entry.digest(evaluator)}
    selection_path, qualification_path = root / "selection.json", root / "qualification.json"
    dump(selection_path, selection)
    evidence = {}
    for stage, seeds in entry.STAGES.items():
        directory = root / "results/q4_clear_before_probe" / stage
        manifest = {"stage": stage, "seeds": seeds, "source_sha256": source,
                    "specs": selection["specs"], "selection_sha256": entry.digest(selection_path)}
        dump(directory / "manifest.json", manifest)
        dump(directory / "freeze.json", {"manifest_sha256": entry.manifest_digest(manifest),
                                          "git_commit": "b" * 40})
        dump(directory / "summary.json", {"fixture": True})
        for name in ("independent_audit.json", "clear_before_probe_audit.json"):
            dump(directory / name, {"all_passed": True, "records": len(seeds), "passed_records": len(seeds)})
        with zipfile.ZipFile(directory / "source.zip", "w") as archive:
            archive.writestr("src/frozen.py", source_bytes)
        for name in entry.EVIDENCE_NAMES:
            path = directory / name
            evidence[path.relative_to(root).as_posix()] = entry.digest(path)
    qualification = {"phase": "confirmation", "selected": entry.SELECTED, "passed": True,
        "decision_rule": entry.DECISION_RULE, "source_sha256": dict(source),
        "evaluator_sha256": entry.digest(evaluator), "selection_sha256": entry.digest(selection_path),
        "evidence_sha256": evidence}
    dump(qualification_path, qualification)
    monkeypatch.setattr(entry, "ROOT", root)
    monkeypatch.setattr(entry, "hashes", lambda: dict(source))
    sim_dir = root / "simulator"
    sim_dir.mkdir()
    (sim_dir / "jammers-simulator-full.exe").write_bytes(b"inert")
    host = SimpleNamespace(root=root, selection=selection, qualification=qualification,
        selection_path=selection_path, qualification_path=qualification_path, source=source,
        sim_dir=sim_dir, output=root / "output", calls=[], solves=[], events=[],
        state={"active": False, "mode": "", "case_code": "", "phase": ""},
        before_solver=None, after_solver=None, completed=True)

    class Bridge:
        def __init__(self, port): host.events.append("bridge")
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def current_test(self): return host.state

    @contextmanager
    def lock(path):
        assert path == sim_dir / ".practice-control/controller.lock"
        host.events.append("lock")
        yield

    def solve(client, **kwargs):
        host.solves.append(kwargs)
        return {"mock_policy": True}

    def run_once(bridge, **kwargs):
        host.calls.append(kwargs)
        kwargs["output"].mkdir()
        if host.before_solver: host.before_solver()
        kwargs["solver"](object(), problem=kwargs["problem"], variant=kwargs["variant"],
                         max_actions=kwargs["max_actions"])
        if host.after_solver: host.after_solver()
        host.events.append("registered_mock")
        return {"completed": host.completed}

    def bounds(result, run_dir):
        assert host.events[-1] == "registered_mock"
        host.events.append("bounds")
        return {"official_all_clear_verified": host.completed,
                "time_to_conditional_lower_bound_ratio": 3.0 if host.completed else None}

    monkeypatch.setattr(entry, "PracticeBridge", Bridge)
    monkeypatch.setattr(entry, "controller_lock", lock)
    monkeypatch.setattr(entry, "run_once", run_once)
    monkeypatch.setattr(entry, "run_q4_clear_before_probe", solve)
    host.original_bounds = entry.post_registration_bounds
    monkeypatch.setattr(entry, "post_registration_bounds", bounds)

    def main(*extra):
        monkeypatch.setattr("sys.argv", ["practice.py", "--selection", str(selection_path),
            "--qualification", str(qualification_path), "--robot-id", "TEST-TEAM",
            "--simulator-dir", str(sim_dir), "--output", str(host.output), *extra])
        return entry.main()
    host.main = main
    return host


def test_real_preflight_is_offline(rig, capsys):
    assert rig.main("--preflight-only") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected"] == "compact_clear_before_probe"
    assert result["config"] == "center_once" and result["source_commit"] == "b" * 40
    assert result["spec"] == entry.SPEC and len(result["source_archive_sha256"]) == 2
    assert not rig.events and not rig.output.exists()


@pytest.mark.parametrize("key,value", [("passed", False), ("passed", 1), ("phase", "development"),
    ("selected", "compact_combo"), ("decision_rule", "other"), ("selection_sha256", "0" * 64),
    ("source_sha256", {}), ("evaluator_sha256", "0" * 64), ("evidence_sha256", {})])
def test_bad_qualification_rejected_before_bridge(rig, key, value):
    rig.qualification[key] = value
    dump(rig.qualification_path, rig.qualification)
    with pytest.raises(ValueError): rig.main()
    assert not rig.events


@pytest.mark.parametrize("change", ["source", "selected", "spec", "seeds", "evaluator", "evidence",
                                    "zip_content", "zip_member", "zip_missing"])
def test_frozen_inputs_and_archives_rejected(rig, change):
    if change == "source": rig.source["src/frozen.py"] = "0" * 64
    elif change == "selected": rig.selection["selected"] = "compact_combo"
    elif change == "spec": rig.selection["specs"][entry.SELECTED]["kwargs"]["max_expansions"] = 1
    elif change == "seeds": rig.selection["reserved_seeds"]["confirmation"] = [1]
    elif change == "evaluator": (rig.root / "experiments/evaluate_q4_clear_before_probe.py").write_bytes(b"changed")
    elif change == "evidence": (rig.root / next(iter(rig.qualification["evidence_sha256"]))).write_bytes(b"changed")
    else:
        path = rig.root / "results/q4_clear_before_probe/confirmation/source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            if change != "zip_missing": archive.writestr("src/frozen.py", b"changed")
            if change == "zip_member": archive.writestr("extra", b"extra")
    dump(rig.selection_path, rig.selection)
    with pytest.raises(ValueError): rig.main()
    assert not rig.events


@pytest.mark.parametrize("name,key,value", [("clear_before_probe_audit.json", "passed_records", 63),
    ("independent_audit.json", "all_passed", False), ("manifest.json", "seeds", [1]),
    ("freeze.json", "manifest_sha256", "0" * 64), ("freeze.json", "git_commit", "c" * 40)])
def test_hash_alone_does_not_replace_stage_and_audit_checks(rig, name, key, value):
    relative = f"results/q4_clear_before_probe/confirmation/{name}"
    path = rig.root / relative
    data = json.loads(path.read_bytes())
    data[key] = value
    dump(path, data)
    rig.qualification["evidence_sha256"][relative] = entry.digest(path)
    dump(rig.qualification_path, rig.qualification)
    with pytest.raises(ValueError): rig.main()
    assert not rig.events


@pytest.mark.parametrize("state", [None, {}, {"active": True, "mode": "practice"},
    {"active": False, "mode": "formal"}, {"active": False, "mode": "unknown"},
    {"active": False, "case_code": "OLD"}, {"active": False, "phase": "ended"}, {"active": 0}])
def test_only_explicit_idle_is_accepted(rig, state):
    rig.state = state
    with pytest.raises(ValueError, match="occupied"): rig.main()
    assert not rig.calls and not rig.solves


def test_shared_lock_blocks_before_bridge(rig, monkeypatch):
    @contextmanager
    def busy(path):
        raise BridgeError("held")
        yield
    monkeypatch.setattr(entry, "controller_lock", busy)
    with pytest.raises(BridgeError): rig.main()
    assert not rig.events


def test_only_qualified_r8_is_injected_and_bounds_follow_registration(rig):
    assert rig.main("--repeat", "2") == 0
    assert rig.events[:2] == ["lock", "bridge"]
    assert len(rig.calls) == len(rig.solves) == 2
    for call, solve in zip(rig.calls, rig.solves):
        assert call["method_label"] == entry.SELECTED and call["method_metadata"]["spec"] == entry.SPEC
        assert solve == {"problem": 4, "max_actions": 20000, **entry.SPEC["kwargs"]}
    assert json.loads((rig.output / "results.json").read_bytes())[0]["time_to_conditional_lower_bound_ratio"] == 3
    assert rig.original_bounds is entry.shared_practice.post_registration_bounds
    assert entry.shared_practice.SELECTED == "compact_joint"


@pytest.mark.parametrize("when", ["before_solver", "after_solver"])
def test_source_drift_stops_batch(rig, when):
    def drift(): rig.source["src/frozen.py"] = "0" * 64
    setattr(rig, when, drift)
    with pytest.raises(ValueError): rig.main("--repeat", "2")
    assert len(rig.calls) == 1 and "bounds" not in rig.events
    assert len(rig.solves) == (when == "after_solver")


@pytest.mark.parametrize("kwargs", [dict(problem=3, variant="triangular", max_actions=20000),
    dict(problem=4, variant="formal", max_actions=20000), dict(problem=4, variant="triangular", max_actions=10000)])
def test_injected_solver_refuses_other_problem_or_budget(rig, kwargs):
    assert rig.main() == 0
    with pytest.raises(ValueError): rig.calls[0]["solver"](object(), **kwargs)
    assert len(rig.solves) == 1


def test_incomplete_run_stops_repeats_and_output_is_preserved(rig):
    rig.completed = False
    assert rig.main("--repeat", "2") == 1
    assert len(rig.calls) == 1
    original = (rig.output / "results.json").read_bytes()
    with pytest.raises(FileExistsError): rig.main()
    assert (rig.output / "results.json").read_bytes() == original and len(rig.calls) == 1


@pytest.mark.parametrize("args", [("--repeat", "0"), ("--repeat", "21"), ("--robot-id", " "),
                                  ("--robot-id", "x\n123")])
def test_invalid_controls_fail_before_bridge(rig, args):
    with pytest.raises(ValueError): rig.main(*args)
    assert not rig.events

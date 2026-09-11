"""Offline gates for the frozen Q4 practice entry; no live bridge or HTTP."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from experiments import run_q4_selected_practice as entry
from practice_control import BridgeError
from simulator_client import SimulatorClient


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This test must not contact the simulator or any network")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(SimulatorClient, "_exchange", forbidden)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    evidence = root / "audit.json"
    dump(evidence, {"passed": True})
    selection_path = root / "selection.json"
    qualification_path = root / "qualification.json"
    spec = {"entrypoint": "strategies.q4_state_search:run_q4_state_search",
            "kwargs": {"mode": "state_pruned", "max_expansions": 200}}
    source = {"src/frozen.py": "a" * 64}
    selection = {"selected": "state_pruned", "spec": spec,
                 "source_sha256": source, "git_commit": "b" * 40}
    dump(selection_path, selection)
    qualification = {"selected": "state_pruned", "passed": True,
                     "selection_sha256": digest(selection_path),
                     "evidence_sha256": {"audit.json": digest(evidence)}}
    dump(qualification_path, qualification)
    monkeypatch.setattr(entry, "ROOT", root)
    monkeypatch.setattr(entry, "SPECS", {"state_pruned": json.loads(json.dumps(spec)),
                                         "triangular": {}})
    monkeypatch.setattr(entry, "source_hashes", lambda: dict(source))
    sim_dir = root / "simulator"
    sim_dir.mkdir()
    (sim_dir / "jammers-simulator-full.exe").write_bytes(b"not an executable")
    host = SimpleNamespace(root=root, evidence=evidence, selection_path=selection_path,
        qualification_path=qualification_path, selection=selection, qualification=qualification,
        source=source, sim_dir=sim_dir, output=root / "output", calls=[], solves=[],
        events=[], state={"active": False, "mode": "", "case_code": "", "phase": ""},
        completed=True, before_solver=None, after_solver=None)

    class FakeBridge:
        def __init__(self, port):
            host.events.append(("bridge", port))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            host.events.append(("bridge_exit",))

        def current_test(self):
            host.events.append(("state",))
            return host.state

    @contextmanager
    def lock(path):
        host.events.append(("lock", path))
        yield

    def solver(client, **kwargs):
        host.solves.append(kwargs)
        return {"solved": True}

    def run_once(bridge, **kwargs):
        host.calls.append(kwargs)
        if host.before_solver:
            host.before_solver()
        kwargs["solver"](object(), problem=kwargs["problem"],
                         variant=kwargs["variant"], max_actions=kwargs["max_actions"])
        if host.after_solver:
            host.after_solver()
        return {"completed": host.completed, "problem": kwargs["problem"]}

    monkeypatch.setattr(entry, "PracticeBridge", FakeBridge)
    monkeypatch.setattr(entry, "controller_lock", lock)
    monkeypatch.setattr(entry, "run_q4_state_search", solver)
    monkeypatch.setattr(entry, "run_once", run_once)

    def main(*extra):
        monkeypatch.setattr("sys.argv", ["run_q4_selected_practice.py",
            "--selection", str(selection_path), "--qualification", str(qualification_path),
            "--robot-id", "TEST-TEAM", "--simulator-dir", str(sim_dir),
            "--output", str(host.output), *extra])
        return entry.main()

    host.main = main
    return host


def test_valid_frozen_preflight_only_is_offline_without_output(rig, capsys):
    assert rig.main("--preflight-only") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preflight_passed"] is True
    assert payload["source_commit"] == rig.selection["git_commit"]
    assert payload["selection_sha256"] == digest(rig.selection_path)
    assert payload["qualification_sha256"] == digest(rig.qualification_path)
    assert payload["practice_entry_sha256"] == digest(Path(entry.__file__))
    assert not rig.events and not rig.calls and not rig.output.exists()


@pytest.mark.parametrize("change", ["source", "spec", "triangular", "unknown"])
def test_changed_source_or_selected_configuration_is_rejected_offline(rig, change):
    if change == "source":
        rig.source["src/frozen.py"] = "c" * 64
    elif change == "spec":
        rig.selection["spec"]["kwargs"]["max_expansions"] = 201
        dump(rig.selection_path, rig.selection)
    else:
        rig.selection["selected"] = change
        dump(rig.selection_path, rig.selection)
    with pytest.raises(ValueError):
        rig.main()
    assert not rig.events and not rig.output.exists()


@pytest.mark.parametrize("change", ["failed", "wrong_candidate", "selection_hash",
    "empty", "missing_map", "wrong_map", "changed_file", "missing_file", "directory",
    "absolute", "traversal", "invalid_hash"])
def test_missing_or_mismatched_qualification_evidence_is_rejected(rig, change):
    q = rig.qualification
    if change == "failed":
        q["passed"] = False
    elif change == "wrong_candidate":
        q["selected"] = "another_method"
    elif change == "selection_hash":
        q["selection_sha256"] = "0" * 64
    elif change == "empty":
        q["evidence_sha256"] = {}
    elif change == "missing_map":
        q.pop("evidence_sha256")
    elif change == "wrong_map":
        q["evidence_sha256"] = []
    elif change == "changed_file":
        dump(rig.evidence, {"passed": False})
    elif change == "missing_file":
        q["evidence_sha256"] = {"absent.json": "a" * 64}
    elif change == "directory":
        q["evidence_sha256"] = {"simulator": "a" * 64}
    elif change == "absolute":
        q["evidence_sha256"] = {str(rig.evidence): digest(rig.evidence)}
    elif change == "traversal":
        outside = rig.root.parent / "outside.json"
        dump(outside, {"passed": True})
        q["evidence_sha256"] = {"../outside.json": digest(outside)}
    else:
        q["evidence_sha256"] = {"audit.json": "not-a-sha"}
    dump(rig.qualification_path, q)
    with pytest.raises(ValueError):
        rig.main()
    assert not rig.events and not rig.output.exists()


@pytest.mark.parametrize("state", [None, {}, {"active": True, "mode": "practice"},
    {"active": False, "mode": "formal"}, {"active": False, "mode": "unknown"},
    {"active": False, "mode": "practice", "case_code": "ABCD-EFGH-IJKL-MNOP"},
    {"active": False, "mode": "practice", "phase": "ended"}, {"active": 0}])
def test_busy_formal_unknown_or_finished_session_is_left_untouched(rig, state):
    rig.state = state
    with pytest.raises(ValueError, match="occupied"):
        rig.main()
    assert not rig.calls and not rig.solves and not rig.output.exists()


@pytest.mark.parametrize("args", [("--robot-id", "   "), ("--robot-id", "x" * 65),
    ("--robot-id", "abc\n123"), ("--robot-id", "abc\u200b123"),
    ("--repeat", "0"), ("--repeat", "21")])
def test_invalid_team_or_repeat_is_rejected_before_bridge(rig, args):
    with pytest.raises(ValueError):
        rig.main(*args)
    assert not rig.events and not rig.calls and not rig.output.exists()


def test_missing_simulator_executable_is_rejected_before_bridge(rig):
    rig.sim_dir = rig.root
    with pytest.raises(ValueError, match="executable"):
        rig.main("--simulator-dir", str(rig.root))
    assert not rig.events and not rig.calls


def test_held_controller_lock_never_opens_bridge(rig, monkeypatch):
    @contextmanager
    def busy(path):
        raise BridgeError("Another practice controller holds the lock")
        yield
    monkeypatch.setattr(entry, "controller_lock", busy)
    with pytest.raises(BridgeError, match="holds"):
        rig.main()
    assert not rig.events and not rig.calls and not rig.output.exists()


def test_exact_selected_solver_kwargs_and_metadata_for_two_practices(rig):
    assert rig.main("--repeat", "2") == 0
    assert len(rig.calls) == len(rig.solves) == 2
    assert rig.events[0][0] == "lock"
    for call, solve in zip(rig.calls, rig.solves):
        assert call["problem"] == 4 and call["variant"] == "triangular"
        assert call["method_label"] == "state_pruned"
        assert call["method_metadata"]["spec"] == rig.selection["spec"]
        assert call["method_metadata"]["declared_mode"] == "practice"
        assert solve == {"problem": 4, "max_actions": 20000, "mode": "state_pruned", "max_expansions": 200}
    assert rig.calls[0]["output"].name == "run-001"
    assert rig.calls[1]["output"].name == "run-002"
    assert len(json.loads((rig.output / "results.json").read_text())) == 2
    assert (rig.output / "result-001.json").is_file()
    assert (rig.output / "result-002.json").is_file()


def test_unsuccessful_practice_stops_batch(rig):
    rig.completed = False
    assert rig.main("--repeat", "2") == 1
    assert len(rig.calls) == 1
    assert not (rig.output / "result-002.json").exists()


@pytest.mark.parametrize("problem,variant", [(3, "triangular"), (4, "efficient"), (4, "formal")])
def test_selected_solver_rejects_unexpected_problem_or_profile(rig, problem, variant):
    assert rig.main() == 0
    with pytest.raises(ValueError, match="Unexpected"):
        rig.calls[0]["solver"](object(), problem=problem, variant=variant, max_actions=20000)
    assert len(rig.solves) == 1


@pytest.mark.parametrize("drift", ["source", "evidence", "selection", "qualification"])
def test_batch_stops_before_next_practice_when_frozen_evidence_drifts(rig, drift):
    def change():
        if drift == "source":
            rig.source["src/frozen.py"] = "d" * 64
        elif drift == "evidence":
            dump(rig.evidence, {"passed": False})
        else:
            path = rig.selection_path if drift == "selection" else rig.qualification_path
            path.write_bytes(path.read_bytes() + b" ")
    rig.after_solver = change
    with pytest.raises(ValueError):
        rig.main("--repeat", "2")
    assert len(rig.calls) == len(rig.solves) == 1
    assert not (rig.output / "result-002.json").exists()


def test_source_drift_after_start_is_rejected_before_policy_client_use(rig):
    rig.before_solver = lambda: rig.source.update({"src/frozen.py": "e" * 64})
    with pytest.raises(ValueError, match="Frozen Q4 code"):
        rig.main()
    assert len(rig.calls) == 1 and not rig.solves


def test_existing_output_is_not_overwritten_or_started(rig):
    rig.output.mkdir()
    marker = rig.output / "keep.txt"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        rig.main()
    assert not rig.calls and marker.read_text() == "existing evidence"

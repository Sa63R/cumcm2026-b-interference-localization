"""Practice-only compact-cover gates, with every network path disabled."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from experiments import run_q4_cover_practice as entry
from practice_control import BridgeError
from simulator_client import SimulatorClient


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No simulator or network contact is allowed in this test")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(SimulatorClient, "_exchange", forbidden)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    evidence = root / "audit.json"
    dump(evidence, {"passed": True})
    selection_path, qualification_path = root / "selection.json", root / "qualification.json"
    source = {"src/frozen.py": "a" * 64}
    selection = {"selected": entry.SELECTED, "profile": "compact_22", "schedule": "joint",
                 "specs": {entry.SELECTED: copy.deepcopy(entry.SPEC)},
                 "source_sha256": dict(source), "git_commit": "b" * 40}
    dump(selection_path, selection)
    qualification = {"selected": entry.SELECTED, "profile": "compact_22", "schedule": "joint",
                     "passed": True, "selection_sha256": entry.digest(selection_path),
                     "evidence_sha256": {"audit.json": entry.digest(evidence)}}
    dump(qualification_path, qualification)
    monkeypatch.setattr(entry, "ROOT", root)
    monkeypatch.setattr(entry, "frozen_hashes", lambda: dict(source))
    sim_dir = root / "simulator"
    sim_dir.mkdir()
    (sim_dir / "jammers-simulator-full.exe").write_bytes(b"inert test file")
    host = SimpleNamespace(root=root, evidence=evidence, selection_path=selection_path,
        qualification_path=qualification_path, selection=selection, qualification=qualification,
        source=source, sim_dir=sim_dir, output=root / "output", calls=[], solves=[], events=[],
        state={"active": False, "mode": "", "case_code": "", "phase": ""},
        completed=True, cleared=10, before_solver=None, after_solver=None, change_record=None,
        skip_registration=False, registration_error=False, bounds_calls=[])

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
        run = kwargs["output"]
        run.mkdir()
        if host.before_solver:
            host.before_solver()
        kwargs["solver"](object(), problem=kwargs["problem"],
                         variant=kwargs["variant"], max_actions=kwargs["max_actions"])
        summary = {"completed": host.completed, "problem": 4,
            "data_origin": "simulator_http_session", "declared_mode": "practice",
            "case_code": f"CASE-{len(host.calls)}", "pending_request": None,
            "state": {"session": "exited", "cleared_count": host.cleared, "virtual_time_s": 3000.0},
            "search": {"time_breakdown": {}, "action_history": [
                {"action": "clear", "result": "success", "channel": i+1,
                 "position": [float(i*20), 0.0]} for i in range(host.cleared)]}}
        dump(run / "summary.json", summary)
        registered = run.parent / "registered"
        registered.mkdir(exist_ok=True)
        record_path = registered / f"practice-{len(host.calls)}.json"
        record = {"data_origin": "registered_official_practice", "problem": 4,
            "case_code": summary["case_code"], "source_total": 10,
            "source_total_source": "official_simulator_result_file",
            "official_result_session_time_matched": True,
            "summary_sha256": entry.digest(run / "summary.json"),
            "cleared_count": host.cleared, "virtual_time_s": 3000.0,
            "search_completed": host.completed}
        if host.change_record:
            host.change_record(record, summary, run)
        dump(record_path, record)
        if host.registration_error:
            raise ValueError("registration rejected")
        if not host.skip_registration:
            dump(run / "registration.json", {"record": str(record_path),
                "case_code": summary["case_code"], "source_total": 10})
        host.events.append(("registered",))
        if host.after_solver:
            host.after_solver()
        return {"completed": host.completed, "problem": 4, "case_code": summary["case_code"],
                "summary": str(run / "summary.json"), "source_total": 10,
                "cleared_count": host.cleared, "virtual_time_s": 3000.0}

    original_analyze = entry.session_lower_bounds.analyze

    def analyze(path):
        assert host.events[-1][0] == "registered"
        assert (path.parent / "registration.json").is_file()
        host.events.append(("bounds",))
        host.bounds_calls.append(path)
        return {"summary_sha256": entry.digest(path), "problem": 4,
                "actual_virtual_time_s": 3000.0,
                "conditional_guaranteed_all_clear_lower_s": 1000.0}

    monkeypatch.setattr(entry, "PracticeBridge", FakeBridge)
    monkeypatch.setattr(entry, "controller_lock", lock)
    monkeypatch.setattr(entry, "run_q4_cover_search", solver)
    monkeypatch.setattr(entry, "run_once", run_once)
    monkeypatch.setattr(entry.session_lower_bounds, "analyze", analyze)
    host.original_analyze = original_analyze

    def main(*extra):
        monkeypatch.setattr("sys.argv", ["run_q4_cover_practice.py",
            "--selection", str(selection_path), "--qualification", str(qualification_path),
            "--robot-id", "TEST-TEAM", "--simulator-dir", str(sim_dir),
            "--output", str(host.output), *extra])
        return entry.main()
    host.main = main
    return host


def test_preflight_only_is_offline(rig, capsys):
    assert rig.main("--preflight-only") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preflight_passed"] and payload["profile"] == "compact_22"
    assert payload["source_commit"] == rig.selection["git_commit"]
    assert payload["selection_sha256"] == entry.digest(rig.selection_path)
    assert payload["practice_entry_sha256"] == entry.digest(Path(entry.__file__))
    assert not rig.events and not rig.calls and not rig.output.exists()


@pytest.mark.parametrize("key,value", [("profile", "compact_25"), ("schedule", "deferred"),
                                      ("selected", "state_pruned"), ("specs", {})])
def test_wrong_frozen_configuration_rejected(rig, key, value):
    rig.selection[key] = value
    dump(rig.selection_path, rig.selection)
    with pytest.raises(ValueError, match="Frozen Q4 cover"):
        rig.main()
    assert not rig.events


@pytest.mark.parametrize("change", ["failed", "candidate", "selection_hash", "missing", "empty",
                                   "changed_file", "absent_file", "traversal", "absolute", "bad_sha"])
def test_invalid_qualification_rejected_before_bridge(rig, change):
    q = rig.qualification
    if change == "failed": q["passed"] = False
    elif change == "candidate": q["selected"] = "other"
    elif change == "selection_hash": q["selection_sha256"] = "0" * 64
    elif change == "missing": q.pop("evidence_sha256")
    elif change == "empty": q["evidence_sha256"] = {}
    elif change == "changed_file": dump(rig.evidence, {"passed": False})
    elif change == "absent_file": q["evidence_sha256"] = {"absent.json": "0" * 64}
    elif change == "traversal": q["evidence_sha256"] = {"../audit.json": "0" * 64}
    elif change == "absolute": q["evidence_sha256"] = {str(rig.evidence): entry.digest(rig.evidence)}
    else: q["evidence_sha256"] = {"audit.json": "invalid"}
    dump(rig.qualification_path, q)
    with pytest.raises(ValueError): rig.main()
    assert not rig.events and not rig.output.exists()


@pytest.mark.parametrize("state", [None, {}, {"active": True, "mode": "practice"},
    {"active": False, "mode": "formal"}, {"active": False, "mode": "unknown"},
    {"active": False, "mode": "practice", "case_code": "OLD"},
    {"active": False, "phase": "ended"}, {"active": 0}])
def test_nonidle_session_never_started_or_touched(rig, state):
    rig.state = state
    with pytest.raises(ValueError, match="occupied"): rig.main()
    assert not rig.calls and not rig.solves and not rig.bounds_calls


def test_shared_lock_precedes_bridge_and_never_bypassed(rig, monkeypatch):
    @contextmanager
    def busy(path):
        assert path == rig.sim_dir / ".practice-control/controller.lock"
        raise BridgeError("held")
        yield
    monkeypatch.setattr(entry, "controller_lock", busy)
    with pytest.raises(BridgeError, match="held"): rig.main()
    assert not rig.events and not rig.calls


def test_two_registered_runs_fixed_solver_and_postexit_bounds(rig, capsys):
    assert rig.main("--repeat", "2") == 0
    assert len(rig.calls) == len(rig.solves) == len(rig.bounds_calls) == 2
    assert rig.events[0][0] == "lock"
    for call, solve in zip(rig.calls, rig.solves):
        assert call["problem"] == 4 and call["variant"] == "triangular"
        assert call["method_label"] == "compact_joint"
        assert call["method_metadata"]["spec"] == entry.SPEC
        assert solve == {"problem": 4, "max_actions": 20000, **entry.SPEC["kwargs"]}
        bound = json.loads((call["output"] / "lower_bounds.json").read_bytes())
        assert bound["official_all_clear_verified"] is True
        assert bound["time_to_conditional_lower_bound_ratio"] == 3.0
        assert bound["conditional_lower_bound_gap_s"] == 2000.0
    results = json.loads((rig.output / "results.json").read_bytes())
    assert len(results) == 2 and all(r["time_to_conditional_lower_bound_ratio"] == 3 for r in results)
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_real_offline_bound_analyzer_used_on_ended_observation_fixture(rig, monkeypatch):
    monkeypatch.setattr(entry.session_lower_bounds, "analyze", rig.original_analyze)
    assert rig.main() == 0
    bounds = json.loads((rig.output / "run-001/lower_bounds.json").read_bytes())
    assert bounds["analysis_kind"] == "offline_physical_oracle_bounds_from_observations"
    assert bounds["observed_cleared_sources"] == 10
    assert bounds["empty_channel_necessary_action_cost_s"] == 50
    assert bounds["time_to_conditional_lower_bound_ratio"] == pytest.approx(
        3000 / bounds["conditional_guaranteed_all_clear_lower_s"])


def test_incomplete_registered_run_saves_conditional_bound_without_allclear_ratio(rig):
    rig.completed, rig.cleared = False, 9
    assert rig.main("--repeat", "2") == 1
    assert len(rig.calls) == len(rig.bounds_calls) == 1
    bounds = json.loads((rig.output / "run-001/lower_bounds.json").read_bytes())
    assert bounds["official_all_clear_verified"] is False
    assert bounds["time_to_conditional_lower_bound_ratio"] is None


@pytest.mark.parametrize("change", ["session", "mode", "case", "source", "summary_hash", "time"])
def test_active_mismatched_or_unverified_evidence_never_reaches_analyzer(rig, change):
    def alter(record, summary, run):
        if change in ("session", "mode"):
            if change == "session": summary["state"]["session"] = "active"
            else: summary["declared_mode"] = "formal"
            dump(run / "summary.json", summary)
            record["summary_sha256"] = entry.digest(run / "summary.json")
        elif change == "case": record["case_code"] = "ANOTHER"
        elif change == "source": record["source_total_source"] = "unverified"
        elif change == "time": record["virtual_time_s"] = 3100.0
        else: record["summary_sha256"] = "0" * 64
    rig.change_record = alter
    with pytest.raises(ValueError, match="registration evidence"): rig.main()
    assert len(rig.calls) == 1 and not rig.bounds_calls


@pytest.mark.parametrize("failure", ["missing", "rejected"])
def test_unsuccessful_registration_never_reaches_analyzer(rig, failure):
    rig.skip_registration = failure == "missing"
    rig.registration_error = failure == "rejected"
    with pytest.raises((ValueError, FileNotFoundError)): rig.main("--repeat", "2")
    assert len(rig.calls) == 1 and not rig.bounds_calls
    assert (rig.output / "run-001/summary.json").is_file()
    assert (rig.output / "results.json").is_file()


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("drift", ["source", "evidence", "selection", "qualification"])
def test_source_and_evidence_rechecked_before_policy_and_after_registration(rig, when, drift):
    def alter():
        if drift == "source": rig.source["src/frozen.py"] = "c" * 64
        elif drift == "evidence": dump(rig.evidence, {"passed": False})
        else:
            path = rig.selection_path if drift == "selection" else rig.qualification_path
            path.write_bytes(path.read_bytes() + b" ")
    setattr(rig, "before_solver" if when == "before" else "after_solver", alter)
    with pytest.raises(ValueError): rig.main("--repeat", "2")
    assert len(rig.calls) == 1 and not rig.bounds_calls
    assert len(rig.solves) == (when == "after")


@pytest.mark.parametrize("problem,variant", [(3, "triangular"), (4, "formal"), (4, "efficient")])
def test_solver_rejects_other_problem_and_transport_label(rig, problem, variant):
    assert rig.main() == 0
    with pytest.raises(ValueError, match="Unexpected"):
        rig.calls[0]["solver"](object(), problem=problem, variant=variant, max_actions=20000)
    assert len(rig.solves) == 1


@pytest.mark.parametrize("args", [("--repeat", "0"), ("--repeat", "21"),
                                  ("--robot-id", "  "), ("--robot-id", "x\n123")])
def test_invalid_controls_rejected_before_bridge(rig, args):
    with pytest.raises(ValueError): rig.main(*args)
    assert not rig.events


def test_existing_output_not_overwritten(rig):
    rig.output.mkdir()
    marker = rig.output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError): rig.main()
    assert not rig.calls and marker.read_text() == "keep"

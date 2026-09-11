"""Run the qualified compact Q4 cover in practice; bound analysis is post-exit only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_cover_study import frozen_hashes
from experiments import session_lower_bounds
from practice_control.bridge import PracticeBridge
from practice_control.runner import controller_lock, run_once, validate_run, write_json
from strategies.q4_cover_search import run_q4_cover_search

SELECTED = "compact_joint"
SPEC = {"entrypoint": "strategies.q4_cover_search:run_q4_cover_search",
        "kwargs": {"profile": "compact_22", "schedule": "joint", "max_expansions": 200}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(selection_path, qualification_path):
    selection = json.loads(selection_path.read_bytes())
    qualification = json.loads(qualification_path.read_bytes())
    if (selection.get("selected") != SELECTED or selection.get("profile") != "compact_22"
            or selection.get("schedule") != "joint"
            or selection.get("specs", {}).get(SELECTED) != SPEC
            or selection.get("source_sha256") != frozen_hashes()):
        raise ValueError("Frozen Q4 cover code or configuration changed")
    if (qualification.get("passed") is not True
            or qualification.get("selected", SELECTED) != SELECTED
            or qualification.get("profile", "compact_22") != "compact_22"
            or qualification.get("schedule", "joint") != "joint"
            or qualification.get("selection_sha256") != digest(selection_path)):
        raise ValueError("Missing or mismatched Q4 cover qualification")
    evidence = qualification.get("evidence_sha256")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("Qualification must contain frozen evidence files")
    for relative, expected in evidence.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Invalid qualification evidence path or SHA256")
        try:
            path = (ROOT / relative).resolve(strict=True)
            matched = path.is_relative_to(ROOT.resolve()) and path.is_file() and digest(path) == expected
        except OSError as exc:
            raise ValueError("Qualification evidence unavailable") from exc
        if not matched:
            raise ValueError("Qualification evidence changed")
    return selection, qualification


def post_registration_bounds(result, run_dir):
    """Only consume the returned, exited summary and its completed registration.

    The registration supplies the public terminal source count. This function
    never opens simulator storage or a result file from an active session.
    """
    summary_path = (run_dir / "summary.json").resolve(strict=True)
    if Path(result["summary"]).resolve(strict=True) != summary_path:
        raise ValueError("Returned summary does not belong to this practice run")
    summary = json.loads(summary_path.read_bytes())
    registration_path = run_dir / "registration.json"
    registration = json.loads(registration_path.read_bytes())
    record_path = Path(registration["record"]).resolve(strict=True)
    if not record_path.is_relative_to((run_dir.parent / "registered").resolve(strict=True)):
        raise ValueError("Registration record does not belong to this practice batch")
    record = json.loads(record_path.read_bytes())
    state = summary.get("state", {})
    if (summary.get("data_origin") != "simulator_http_session"
            or summary.get("declared_mode") != "practice" or summary.get("problem") != 4
            or state.get("session") != "exited" or summary.get("pending_request") is not None
            or record.get("data_origin") != "registered_official_practice"
            or record.get("problem") != 4
            or record.get("source_total_source") != "official_simulator_result_file"
            or record.get("official_result_session_time_matched") is not True
            or record.get("summary_sha256") != digest(summary_path)
            or any(x.get("case_code") != result.get("case_code") for x in (summary, registration, record))
            or record.get("source_total") != result.get("source_total")
            or registration.get("source_total") != result.get("source_total")
            or record.get("cleared_count") != result.get("cleared_count")
            or state.get("cleared_count") != result.get("cleared_count")
            or record.get("virtual_time_s") != result.get("virtual_time_s")
            or state.get("virtual_time_s") != result.get("virtual_time_s")):
        raise ValueError("Exited Q4 practice and successful registration evidence do not match")
    total = record["source_total"]
    if type(total) is not int or not 10 <= total <= 16:
        raise ValueError("Registered source total is outside the allowed range")
    bounds = session_lower_bounds.analyze(summary_path)
    lower = bounds["conditional_guaranteed_all_clear_lower_s"]
    if (bounds.get("summary_sha256") != digest(summary_path) or bounds.get("problem") != 4
            or bounds.get("actual_virtual_time_s") != result["virtual_time_s"]
            or not isinstance(lower, (int, float)) or not math.isfinite(lower) or lower <= 0):
        raise ValueError("Post-exit lower-bound analysis is inconsistent")
    full = (result.get("completed") is True and summary.get("completed") is True
            and record.get("search_completed") is True and record["cleared_count"] == total)
    bounds.update(registration_sha256=digest(registration_path),
                  registered_record_sha256=digest(record_path), official_source_total=total,
                  official_all_clear_verified=full,
                  time_to_conditional_lower_bound_ratio=result["virtual_time_s"] / lower if full else None,
                  conditional_lower_bound_gap_s=result["virtual_time_s"] - lower if full else None)
    destination = run_dir / "lower_bounds.json"
    write_json(destination, bounds)
    return {"lower_bounds": str(destination),
            "lower_bounds_sha256": digest(destination),
            "conditional_guaranteed_all_clear_lower_s": lower,
            "official_all_clear_verified": full,
            "time_to_conditional_lower_bound_ratio": bounds["time_to_conditional_lower_bound_ratio"],
            "conditional_lower_bound_gap_s": bounds["conditional_lower_bound_gap_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"research/q4_cover_search/selection.json")
    parser.add_argument("--qualification", type=Path, default=ROOT/"research/q4_cover_search/qualification.json")
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selection, qualification = preflight(args.selection, args.qualification)
    metadata = {"selected": SELECTED, "profile": "compact_22", "schedule": "joint",
                "problem": 4, "declared_mode": "practice", "source_commit": selection["git_commit"],
                "spec": selection["specs"][SELECTED], "selection_sha256": digest(args.selection),
                "qualification_sha256": digest(args.qualification), "practice_entry_sha256": digest(Path(__file__)),
                "lower_bound_analyzer_sha256": digest(Path(session_lower_bounds.__file__))}
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, **metadata}), flush=True)
        return 0
    if not args.robot_id or args.simulator_dir is None or not 1 <= args.repeat <= 20:
        raise ValueError("Need robot-id, simulator-dir, and repeat in [1,20]")
    validate_run(4, "triangular", args.repeat, 20000, args.robot_id)
    sim_dir = args.simulator_dir.resolve(strict=True)
    if not (sim_dir / "jammers-simulator-full.exe").is_file():
        raise ValueError("simulator-dir must contain the simulator executable")
    output = (args.output or ROOT/"results/practice_batches"/
              datetime.now(timezone.utc).strftime("q4-cover-%Y%m%dT%H%M%S%fZ")).resolve()

    def verify_snapshot():
        current_selection, current_qualification = preflight(args.selection, args.qualification)
        if (current_selection != selection or current_qualification != qualification
                or digest(args.selection) != metadata["selection_sha256"]
                or digest(args.qualification) != metadata["qualification_sha256"]
                or digest(Path(__file__)) != metadata["practice_entry_sha256"]
                or digest(Path(session_lower_bounds.__file__)) != metadata["lower_bound_analyzer_sha256"]):
            raise ValueError("Frozen Q4 cover batch evidence changed")

    def solver(client, *, problem, variant, max_actions):
        if problem != 4 or variant != "triangular":
            raise ValueError("Unexpected practice-controller problem/profile")
        verify_snapshot()
        # The legacy transport label enables Q4 only; the actual frozen method
        # and kwargs are passed explicitly and recorded independently.
        result = run_q4_cover_search(client, problem=4, max_actions=max_actions, **SPEC["kwargs"])
        verify_snapshot()
        return result

    with controller_lock(sim_dir/".practice-control/controller.lock"):
        with PracticeBridge(args.debug_port) as bridge:
            state = bridge.current_test()
            if (not isinstance(state, dict) or state.get("active") is not False
                    or state.get("mode") not in (None, "", "practice")
                    or state.get("case_code") or state.get("phase")):
                raise ValueError("Simulator is occupied or not explicitly idle; existing session left untouched")
            output.mkdir(parents=True, exist_ok=False)
            write_json(output/"batch.json", {**metadata, "requested_runs": args.repeat})
            results = []
            try:
                for i in range(args.repeat):
                    verify_snapshot()
                    run_dir = output/f"run-{i+1:03d}"
                    result = run_once(bridge, problem=4, robot_id=args.robot_id,
                        variant="triangular", max_actions=20000, output=run_dir,
                        simulator_dir=sim_dir, solver=solver, method_label=SELECTED, method_metadata=metadata)
                    results.append(result)
                    verify_snapshot()
                    result.update(post_registration_bounds(result, run_dir))
                    verify_snapshot()
                    write_json(output/f"result-{i+1:03d}.json", result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    if not result["completed"] or not result["official_all_clear_verified"]:
                        break
            finally:
                write_json(output/"results.json", results)
            return (0 if len(results) == args.repeat and all(
                r.get("completed") and r.get("official_all_clear_verified") for r in results) else 1)


if __name__ == "__main__":
    raise SystemExit(main())

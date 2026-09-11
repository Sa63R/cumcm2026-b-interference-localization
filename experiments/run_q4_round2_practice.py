"""Run the qualified round2 compact_combo through the Q4 practice-only controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import hashes
from experiments import run_q4_cover_practice as shared_practice
from experiments.run_q4_cover_practice import post_registration_bounds
from experiments import session_lower_bounds
from practice_control.bridge import PracticeBridge
from practice_control.runner import controller_lock, run_once, validate_run, write_json
from strategies.q4_range_scheduling import run_q4_range_scheduling

SELECTED = "compact_combo"
SPEC = {"entrypoint": "strategies.q4_range_scheduling:run_q4_range_scheduling",
        "kwargs": {"config": "onroute", "max_expansions": 200}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(selection_path, qualification_path):
    selection = json.loads(selection_path.read_bytes())
    qualification = json.loads(qualification_path.read_bytes())
    if (selection.get("specs", {}).get(SELECTED) != SPEC
            or selection.get("source_sha256") != hashes()):
        raise ValueError("Frozen Q4 round2 code or configuration changed")
    # These fields are emitted by evaluate_q4_round2; unlike the previous
    # qualification schema there is no single 'passed' or 'selected' field.
    if (qualification.get("primary_passed") is not True
            or qualification.get("secondary_passed") is not True
            or qualification.get("promoted") != SELECTED
            or qualification.get("spec") != SPEC
            or not isinstance(selection.get("testing_order"), str)
            or qualification.get("decision_rule") != selection["testing_order"]
            or qualification.get("selection_sha256") != digest(selection_path)):
        raise ValueError("Missing or mismatched Q4 round2 qualification")
    evidence = qualification.get("evidence_sha256")
    required = {f"results/q4_round2/{stage}/{name}" for stage in ("confirmation", "stress")
                for name in ("summary.json", "independent_audit.json", "manifest.json", "freeze.json", "source.zip")}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError("Qualification must contain the ten frozen independent evidence files")
    for relative, expected in evidence.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Invalid qualification evidence path or SHA256")
        try:
            path = (ROOT / relative).resolve(strict=True)
            matched = path.is_relative_to(ROOT.resolve()) and path.is_file() and digest(path) == expected
        except OSError as exc:
            raise ValueError("Qualification evidence unavailable") from exc
        if not matched:
            raise ValueError("Qualification evidence changed")
    return selection, qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"research/q4_round2/selection.json")
    parser.add_argument("--qualification", type=Path, default=ROOT/"research/q4_round2/qualification.json")
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selection, qualification = preflight(args.selection, args.qualification)
    metadata = {"selected": SELECTED, "profile": "compact_22", "config": "onroute",
                "problem": 4, "declared_mode": "practice", "source_commit": selection["git_commit"],
                "spec": selection["specs"][SELECTED], "selection_sha256": digest(args.selection),
                "qualification_sha256": digest(args.qualification), "practice_entry_sha256": digest(Path(__file__)),
                "post_registration_helper_sha256": digest(Path(shared_practice.__file__)),
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
              datetime.now(timezone.utc).strftime("q4-round2-%Y%m%dT%H%M%S%fZ")).resolve()

    def verify_snapshot():
        current_selection, current_qualification = preflight(args.selection, args.qualification)
        if (current_selection != selection or current_qualification != qualification
                or digest(args.selection) != metadata["selection_sha256"]
                or digest(args.qualification) != metadata["qualification_sha256"]
                or digest(Path(__file__)) != metadata["practice_entry_sha256"]
                or digest(Path(shared_practice.__file__)) != metadata["post_registration_helper_sha256"]
                or digest(Path(session_lower_bounds.__file__)) != metadata["lower_bound_analyzer_sha256"]):
            raise ValueError("Frozen Q4 round2 batch evidence changed")

    def solver(client, *, problem, variant, max_actions):
        if problem != 4 or variant != "triangular":
            raise ValueError("Unexpected practice-controller problem/profile")
        verify_snapshot()
        # The controller's legacy transport label only enables Q4. The actual
        # fixed method and kwargs are independently recorded and passed here.
        result = run_q4_range_scheduling(client, problem=4, max_actions=max_actions, **SPEC["kwargs"])
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
                    # Reuse the original helper directly. It only accepts an
                    # exited practice summary with a matching registered result.
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

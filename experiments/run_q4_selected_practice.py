"""Run the locally qualified Q4 candidate through the practice-only controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_state_study import SPECS, source_hashes
from practice_control.bridge import PracticeBridge
from practice_control.runner import controller_lock, run_once, validate_run, write_json
from strategies.q4_state_search import run_q4_state_search


def preflight(selection_path, qualification_path):
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    if not isinstance(selected, str) or selected not in SPECS or selected == "triangular":
        raise ValueError("Expected an explicitly selected Q4 candidate")
    if selection["source_sha256"] != source_hashes() or selection["spec"] != SPECS[selected]:
        raise ValueError("Frozen Q4 code or configuration changed")
    if (qualification.get("passed") is not True or qualification.get("selected") != selected
            or qualification.get("selection_sha256") != hashlib.sha256(selection_bytes).hexdigest()):
        raise ValueError("Missing or mismatched qualification evidence")
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
            matched = (path.is_relative_to(ROOT.resolve()) and path.is_file()
                       and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
        except OSError as exc:
            raise ValueError("Qualification evidence unavailable") from exc
        if not matched:
            raise ValueError("Qualification evidence changed")
    return selection, qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"research/q4_state_search/selection.json")
    parser.add_argument("--qualification", type=Path, default=ROOT/"research/q4_state_search/qualification.json")
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selection, qualification = preflight(args.selection, args.qualification)
    metadata = {"profile": selection["selected"], "problem": 4, "declared_mode": "practice",
                "source_commit": selection["git_commit"], "spec": selection["spec"],
                "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                "qualification_sha256": hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
                "practice_entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, **metadata}), flush=True)
        return 0
    if not args.robot_id or args.simulator_dir is None or not 1 <= args.repeat <= 20:
        raise ValueError("Need robot-id, simulator-dir, and repeat in [1,20]")
    validate_run(4, "triangular", args.repeat, 20000, args.robot_id)
    sim_dir = args.simulator_dir.resolve(strict=True)
    if not (sim_dir / "jammers-simulator-full.exe").is_file():
        raise ValueError("simulator-dir must contain the simulator executable")
    output = args.output or ROOT/"results/practice_batches"/datetime.now(timezone.utc).strftime("q4-state-%Y%m%dT%H%M%S%fZ")

    def verify_snapshot():
        current_selection, current_qualification = preflight(args.selection, args.qualification)
        if (current_selection != selection or current_qualification != qualification
                or hashlib.sha256(args.selection.read_bytes()).hexdigest() != metadata["selection_sha256"]
                or hashlib.sha256(args.qualification.read_bytes()).hexdigest() != metadata["qualification_sha256"]):
            raise ValueError("Frozen Q4 batch evidence changed")

    def solver(client, *, problem, variant, max_actions):
        if problem != 4 or variant != "triangular":
            raise ValueError("Unexpected practice-controller problem/profile")
        verify_snapshot()
        # 'triangular' is the controller's legacy allowed Q4 entry label only.
        # The actual method name and frozen kwargs are always recorded above.
        return run_q4_state_search(client, problem=4, max_actions=max_actions,
                                   **selection["spec"]["kwargs"])

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
            for i in range(args.repeat):
                verify_snapshot()
                result = run_once(bridge, problem=4, robot_id=args.robot_id,
                    variant="triangular", max_actions=20000, output=output/f"run-{i+1:03d}",
                    simulator_dir=sim_dir, solver=solver, method_label=selection["selected"], method_metadata=metadata)
                results.append(result)
                write_json(output/f"result-{i+1:03d}.json", result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if not result["completed"]:
                    break
            write_json(output/"results.json", results)
            return 0 if len(results)==args.repeat and all(r["completed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

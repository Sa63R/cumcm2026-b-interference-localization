"""Run only the independently qualified R8 method through the shared Q4 practice controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import hashes, digest as manifest_digest
from experiments import run_q4_cover_practice as shared_practice
from experiments.run_q4_cover_practice import post_registration_bounds
from experiments import session_lower_bounds
from practice_control.bridge import PracticeBridge
from practice_control.runner import controller_lock, run_once, validate_run, write_json
from strategies.q4_clear_before_probe import run_q4_clear_before_probe

SELECTED = "compact_clear_before_probe"
SPEC = {"entrypoint": "strategies.q4_clear_before_probe:run_q4_clear_before_probe",
        "kwargs": {"max_expansions": 200}}
DECISION_RULE = ("Protocol.md: all-clear and all audits, positive random mean (confirmation CI95 lower>0 "
                 "and mean reduction>=0.5%), nonnegative stress mean, p95<=1.05 both; single fixed candidate")
STAGES = {"confirmation": list(range(617101, 617165)), "stress": list(range(617201, 617243))}
EVIDENCE_NAMES = ("summary.json", "independent_audit.json", "manifest.json", "freeze.json",
                  "clear_before_probe_audit.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_file(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid evidence path")
    path = (ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Evidence must be a workspace file")
    return path


def preflight(selection_path, qualification_path):
    selection = json.loads(selection_path.read_bytes())
    qualification = json.loads(qualification_path.read_bytes())
    source = hashes()
    evaluator_sha = digest(local_file("experiments/evaluate_q4_clear_before_probe.py"))
    if (selection.get("selected") != SELECTED or selection.get("specs", {}).get(SELECTED) != SPEC
            or selection.get("reserved_seeds") != STAGES or selection.get("source_sha256") != source
            or selection.get("evaluator_sha256") != evaluator_sha):
        raise ValueError("Frozen R8 code or configuration changed")
    if (qualification.get("phase") != "confirmation" or qualification.get("passed") is not True
            or qualification.get("selected") != SELECTED or qualification.get("decision_rule") != DECISION_RULE
            or qualification.get("source_sha256") != source
            or qualification.get("evaluator_sha256") != evaluator_sha
            or qualification.get("selection_sha256") != digest(selection_path)):
        raise ValueError("Missing or mismatched R8 qualification")
    evidence = qualification.get("evidence_sha256")
    required = {f"results/q4_clear_before_probe/{stage}/{name}"
                for stage in STAGES for name in EVIDENCE_NAMES}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError("Need all ten frozen R8 evidence files, including the dedicated audits")
    for relative, expected in evidence.items():
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)
                or digest(local_file(relative)) != expected):
            raise ValueError("Qualification evidence changed")
    archive_hashes, commits = {}, set()
    for stage, seeds in STAGES.items():
        directory = f"results/q4_clear_before_probe/{stage}"
        manifest = json.loads(local_file(f"{directory}/manifest.json").read_bytes())
        freeze = json.loads(local_file(f"{directory}/freeze.json").read_bytes())
        if (manifest.get("stage") != stage or manifest.get("seeds") != seeds
                or manifest.get("specs") != selection["specs"] or manifest.get("source_sha256") != source
                or manifest.get("selection_sha256") != digest(selection_path)
                or freeze.get("manifest_sha256") != manifest_digest(manifest)):
            raise ValueError("Independent R8 experiment differs from frozen selection")
        commit = freeze.get("git_commit")
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Invalid frozen source commit")
        commits.add(commit)
        for name, count in (("independent_audit.json", len(seeds) * len(selection["specs"])),
                            ("clear_before_probe_audit.json", len(seeds))):
            audit = json.loads(local_file(f"{directory}/{name}").read_bytes())
            if (audit.get("all_passed") is not True or audit.get("records") != count
                    or audit.get("passed_records") != count):
                raise ValueError("Incomplete or failed independent R8 audit")
        archive_path = local_file(f"{directory}/source.zip")
        # The frozen qualification hashes its ten JSON files. Separately check
        # every archived source member and bind both archive bytes to this batch.
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(source) or set(archive.namelist()) != set(source):
                raise ValueError("Frozen source archive member set differs")
            if any(hashlib.sha256(archive.read(p)).hexdigest() != h for p, h in source.items()):
                raise ValueError("Frozen source archive content changed")
        archive_hashes[f"{directory}/source.zip"] = digest(archive_path)
    if len(commits) != 1:
        raise ValueError("Independent R8 stages have different frozen source commits")
    return selection, qualification, {"source_commit": commits.pop(), "source_archive_sha256": archive_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"research/q4_clear_before_probe/selection.json")
    parser.add_argument("--qualification", type=Path, default=ROOT/"research/q4_clear_before_probe/qualification.json")
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selection, qualification, archive_metadata = preflight(args.selection, args.qualification)
    metadata = {"selected": SELECTED, "profile": "compact_22", "config": "center_once",
                "problem": 4, "declared_mode": "practice", **archive_metadata,
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
              datetime.now(timezone.utc).strftime("q4-r8-%Y%m%dT%H%M%S%fZ")).resolve()

    def verify_snapshot():
        current_selection, current_qualification, current_archives = preflight(args.selection, args.qualification)
        if (current_selection != selection or current_qualification != qualification or current_archives != archive_metadata
                or digest(args.selection) != metadata["selection_sha256"]
                or digest(args.qualification) != metadata["qualification_sha256"]
                or digest(Path(__file__)) != metadata["practice_entry_sha256"]
                or digest(Path(shared_practice.__file__)) != metadata["post_registration_helper_sha256"]
                or digest(Path(session_lower_bounds.__file__)) != metadata["lower_bound_analyzer_sha256"]):
            raise ValueError("Frozen R8 batch evidence changed")

    def solver(client, *, problem, variant, max_actions):
        if problem != 4 or variant != "triangular" or max_actions != 20000:
            raise ValueError("Unexpected practice-controller problem/profile/budget")
        verify_snapshot()
        # 'triangular' is the unchanged controller's Q4 transport label, not
        # the executed algorithm: only the qualified R8 function is injected.
        result = run_q4_clear_before_probe(client, problem=4, max_actions=max_actions, **SPEC["kwargs"])
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

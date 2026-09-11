"""Run only the independently qualified R12 method through the shared Q4 practice controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
from strategies.q4_joint_continuation import run_q4_joint_continuation

SELECTED = "compact_joint_continuation"
SPEC = {"entrypoint": "strategies.q4_joint_continuation:run_q4_joint_continuation",
        "kwargs": {"config": "after_active_miss_optical", "max_expansions": 200}}
STAGES = {"confirmation": list(range(621101, 621229)), "stress": list(range(621301, 621385))}


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
    from experiments import evaluate_q4_joint_continuation as frozen
    selection = json.loads(selection_path.read_bytes())
    qualification = json.loads(qualification_path.read_bytes())
    source = hashes()
    expected_identity = dict(source_sha256=source, rl_identity=frozen.rl_identity(),
        evaluator_sha256=digest(Path(frozen.__file__)),
        evaluator_dependencies_sha256=frozen.evaluator_dependencies(),
        protocol_sha256=digest(ROOT/"research/q4_joint_continuation/PROTOCOL.md"))
    if (selection.get("selected") != SELECTED or selection.get("specs", {}).get(SELECTED) != SPEC
            or selection.get("reserved_seeds") != STAGES
            or qualification.get("phase") != "confirmation" or qualification.get("passed") is not True
            or qualification.get("selected") != SELECTED
            or qualification.get("selection_sha256") != digest(selection_path)):
        raise ValueError("Need the frozen independently qualified joint continuation configuration")
    if any(value.get(key) != expected for value in (selection, qualification)
           for key, expected in expected_identity.items()):
        raise ValueError("Frozen source, evaluator, protocol or RL identity changed")
    evidence, archive_hashes, commits, comparisons = {}, {}, set(), {}
    for stage, seeds in STAGES.items():
        report, combined, manifest = frozen.read_set(stage, evidence)
        if (manifest["stage"] != stage or manifest["seeds"] != seeds
                or manifest["specs"] != selection["specs"]
                or manifest["selection_sha256"] != digest(selection_path)
                or combined.get("rl_audits_all_passed") is not True):
            raise ValueError("Independent paired experiment or audits differ")
        rows = report["rows"]
        expected_pairs = {(seed, label) for seed in seeds for label in selection["specs"]}
        if (len(rows) != len(expected_pairs) or
                {(r["seed"], r["strategy"]) for r in rows} != expected_pairs):
            raise ValueError("Independent report does not contain the complete frozen pair matrix")
        reference = {r["seed"]: r for r in rows if r["strategy"] == frozen.REFERENCE}
        if any(r["case_sha256"] != reference[r["seed"]]["case_sha256"] or
               r["common_lower_bound_s"] != reference[r["seed"]]["common_lower_bound_s"] or
               type(r["successful"]) is not bool or not all(
                   isinstance(r[k], (float, int)) and not isinstance(r[k], bool) and math.isfinite(r[k]) and r[k] > 0
                   for k in ("virtual_time_s", "penalized_time_s", "common_lower_bound_s")) for r in rows):
            raise ValueError("Independent report has mismatched cases/bounds or invalid completed costs")
        directory = ROOT/"results/q4_joint_continuation"/stage
        freeze = json.loads((directory/"freeze.json").read_bytes())
        if freeze["manifest_sha256"] != manifest_digest(manifest):
            raise ValueError("Frozen manifest differs")
        commit = freeze.get("git_commit")
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Invalid frozen source commit")
        commits.add(commit)
        for name, count in (("independent_audit.json", len(seeds)*len(selection["specs"])),
                            ("clear_before_probe_audit.json", len(seeds)),
                            ("joint_visibility_prefix_audit.json", len(seeds)),
                            ("joint_continuation_prefix_audit.json", len(seeds))):
            audit = json.loads((directory/name).read_bytes())
            if audit.get("records") != count or audit.get("passed_records") != count:
                raise ValueError("Independent audit incomplete")
        archive_path = directory/"source.zip"
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(source) or set(archive.namelist()) != set(source):
                raise ValueError("Frozen source archive member set differs")
            if any(hashlib.sha256(archive.read(p)).hexdigest() != h for p,h in source.items()):
                raise ValueError("Frozen source archive content differs")
        archive_hashes[archive_path.relative_to(ROOT).as_posix()] = digest(archive_path)
        comparisons[stage] = frozen.comparison(report, SELECTED, frozen.REFERENCE)
    expected_evidence = {f"results/q4_joint_continuation/{stage}/{name}" for stage in STAGES
        for name in ("manifest.json", "freeze.json", "summary.json", "source.zip", "independent_audit.json",
                     "clear_before_probe_audit.json", "joint_visibility_prefix_audit.json",
                     "joint_continuation_prefix_audit.json")}
    expected_evidence |= {f"results/q4_joint_continuation/{stage}-rl/{name}" for stage in STAGES
                          for name in ("freeze.json", "comparison.json")}
    if set(evidence) != expected_evidence or any(digest(local_file(p)) != value for p, value in evidence.items()):
        raise ValueError("Need all twenty local frozen evidence files")
    if (evidence != qualification["evidence_sha256"] or len(commits) != 1
            or comparisons != qualification["comparisons_vs_incumbent"][SELECTED]):
        raise ValueError("Qualification evidence or paired comparison changed")
    random, stress = comparisons["confirmation"], comparisons["stress"]
    if not (random["all_clear"] and stress["all_clear"] and random["saving_ci95_s"][0] > 0
            and random["mean_reduction_fraction"] >= .005 and stress["mean_saved_s"] >= 0
            and random["p95_ratio"] <= 1.05 and stress["p95_ratio"] <= 1.05):
        raise ValueError("Independent qualification gate failed")
    return selection, qualification, {"source_commit": commits.pop(), "source_archive_sha256": archive_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"research/q4_joint_continuation/selection.json")
    parser.add_argument("--qualification", type=Path, default=ROOT/"research/q4_joint_continuation/qualification.json")
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selection, qualification, archive_metadata = preflight(args.selection, args.qualification)
    metadata = {"selected": SELECTED, "profile": "compact_22", "config": "after_active_miss_optical",
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
              datetime.now(timezone.utc).strftime("q4-r12-%Y%m%dT%H%M%S%fZ")).resolve()

    def verify_snapshot():
        current_selection, current_qualification, current_archives = preflight(args.selection, args.qualification)
        if (current_selection != selection or current_qualification != qualification or current_archives != archive_metadata
                or digest(args.selection) != metadata["selection_sha256"]
                or digest(args.qualification) != metadata["qualification_sha256"]
                or digest(Path(__file__)) != metadata["practice_entry_sha256"]
                or digest(Path(shared_practice.__file__)) != metadata["post_registration_helper_sha256"]
                or digest(Path(session_lower_bounds.__file__)) != metadata["lower_bound_analyzer_sha256"]):
            raise ValueError("Frozen R12 batch evidence changed")

    def solver(client, *, problem, variant, max_actions):
        if problem != 4 or variant != "triangular" or max_actions != 20000:
            raise ValueError("Unexpected practice-controller problem/profile/budget")
        verify_snapshot()
        # 'triangular' is the unchanged controller's Q4 transport label, not
        # the executed algorithm: only the qualified R12 function is injected.
        result = run_q4_joint_continuation(client, problem=4, max_actions=max_actions, **SPEC["kwargs"])
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

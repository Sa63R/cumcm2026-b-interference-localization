"""Apply predeclared R12 continuation gates against qualified R9 probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import hashes, write_json
from experiments.evaluate_q4_round2 import comparison

RESEARCH = ROOT / "research/q4_joint_continuation"
RL_ROOT = ROOT.parent / "q4-r9-joint-visibility"
RESULTS = ROOT / "results/q4_joint_continuation"
REFERENCE = "compact_joint_probe"
CANDIDATES = ("compact_joint_optical", "compact_joint_continuation")
DEVELOPMENT = {"development": list(range(621001, 621025)),
               "development-stress": list(range(621031, 621045))}
INDEPENDENT = {"confirmation": list(range(621101, 621229)),
               "stress": list(range(621301, 621385))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator_dependencies():
    return {"experiments/evaluate_q4_round2.py": sha(ROOT / "experiments/evaluate_q4_round2.py")}


def development_identity():
    """Identity only; no case generator, execution or file writes."""
    return dict(source_sha256=hashes(), rl_identity=rl_identity(), evaluator_sha256=sha(__file__),
        evaluator_dependencies_sha256=evaluator_dependencies(), protocol_sha256=sha(RESEARCH / "PROTOCOL.md"),
        specs_sha256=sha(RESEARCH / "development-specs.json"),
        specs=json.loads((RESEARCH / "development-specs.json").read_bytes()))


def candidate_passes(random, stress, phase):
    if phase not in {"development", "confirmation"}:
        raise ValueError("Unknown comparison phase")
    mean_gate = random["mean_saved_s"] > 0 if phase == "development" else (
        random["saving_ci95_s"][0] > 0 and random["mean_reduction_fraction"] >= .005)
    return bool(random["all_clear"] and stress["all_clear"] and mean_gate
        and stress["mean_saved_s"] >= 0 and random["p95_ratio"] <= 1.05 and stress["p95_ratio"] <= 1.05)


def choose_candidate(accepted, means):
    if not set(accepted).issubset(CANDIDATES) or len(accepted) != len(set(accepted)):
        raise ValueError("Invalid accepted candidate list")
    if not accepted:
        return None
    selected = min(accepted, key=lambda label: means[label])
    if len(accepted) == 2 and abs(means[CANDIDATES[0]] - means[CANDIDATES[1]]) < 5.:
        selected = CANDIDATES[0]
    return selected


def rl_identity():
    paths = [RL_ROOT / "experiments/q4_frozen_rl_worker.py",
             RL_ROOT / "experiments/run_q4_frozen_rl_compare.py"]
    paths += [RL_ROOT / "research/q4_joint_visibility/rl_reference" / name for name in ("reference.json", "source.zip", "checkpoint.pt")]
    return {path.relative_to(RL_ROOT).as_posix(): sha(path) for path in paths}


def read_set(name, evidence):
    directory = RESULTS / name
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if manifest["source_sha256"] != hashes():
        raise ValueError("Experiment source changed")
    audit_names = ["independent_audit.json", "clear_before_probe_audit.json", "joint_visibility_prefix_audit.json"]
    if "compact_joint_continuation" in manifest["specs"]:
        audit_names.append("joint_continuation_prefix_audit.json")
    for filename in audit_names:
        if json.loads((directory / filename).read_bytes()).get("all_passed") is not True:
            raise ValueError("Require all physical and independent prefix audits")
    report = json.loads((directory / "summary.json").read_bytes())
    rl_directory = RESULTS / (name + "-rl")
    rl_freeze = json.loads((rl_directory / "freeze.json").read_bytes())
    combined = json.loads((rl_directory / "comparison.json").read_bytes())
    identities = rl_identity()
    if (rl_freeze["reference_sha256"] != identities["research/q4_joint_visibility/rl_reference/reference.json"] or
            rl_freeze["worker_sha256"] != identities["experiments/q4_frozen_rl_worker.py"] or
            rl_freeze["runner_sha256"] != identities["experiments/run_q4_frozen_rl_compare.py"] or
            rl_freeze["state_manifest_sha256"] != sha(directory / "manifest.json") or
            rl_freeze["state_summary_sha256"] != sha(directory / "summary.json") or
            rl_freeze["reference_arm"] is not False or
            combined["freeze_sha256"] != sha(rl_directory / "freeze.json")):
        raise ValueError("RL comparison does not match the frozen case matrix or code")
    common_rows = [row for row in combined["rows"] if row["strategy"] != "compact_macro_ppo512"]
    if common_rows != report["rows"]:
        raise ValueError("RL comparison changed state-search rows")
    reference_rows = {row["seed"]: row for row in report["rows"] if row["strategy"] == REFERENCE}
    rl_rows = [row for row in combined["rows"] if row["strategy"] == "compact_macro_ppo512"]
    if len(rl_rows) != len(reference_rows) or {r["seed"] for r in rl_rows} != reference_rows.keys():
        raise ValueError("Incomplete RL pairing")
    for row in rl_rows:
        paired = reference_rows[row["seed"]]
        if row["case_sha256"] != paired["case_sha256"] or row["common_lower_bound_s"] != paired["common_lower_bound_s"]:
            raise ValueError("RL uses a different scene or lower bound")
    for filename in ["manifest.json", "freeze.json", "summary.json", "source.zip"] + audit_names:
        path = directory / filename
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for filename in ("freeze.json", "comparison.json"):
        path = rl_directory / filename
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return report, combined, manifest


def main(phase):
    destination = RESEARCH / ("development-decision.json" if phase == "development" else "qualification.json")
    if destination.exists():
        raise ValueError("Preserve prior decision")
    declared_seeds = DEVELOPMENT if phase == "development" else INDEPENDENT
    inputs, reports, combined, manifests, comparisons = {}, {}, {}, {}, {}
    for name, seeds in declared_seeds.items():
        reports[name], combined[name], manifests[name] = read_set(name, inputs)
        if manifests[name]["seeds"] != seeds:
            raise ValueError("Dataset differs from predeclared seeds")
    specs = manifests[next(iter(declared_seeds))]["specs"]
    if any(m["specs"] != specs for m in manifests.values()):
        raise ValueError("Stages have different specifications")
    if phase == "development":
        expected = json.loads((RESEARCH / "development-specs.json").read_bytes())
        if specs != expected:
            raise ValueError("Development specification differs")
        frozen = json.loads((RESEARCH / "development-freeze.json").read_bytes())
        if (frozen["source_sha256"] != hashes() or frozen["rl_identity"] != rl_identity() or
                frozen["evaluator_sha256"] != sha(__file__) or
                frozen["evaluator_dependencies_sha256"] != evaluator_dependencies() or
                frozen["protocol_sha256"] != sha(RESEARCH / "PROTOCOL.md") or
                frozen["specs_sha256"] != sha(RESEARCH / "development-specs.json") or frozen["specs"] != specs):
            raise ValueError("Development source or RL identity differs from freeze")
    else:
        selection = json.loads((RESEARCH / "selection.json").read_bytes())
        if (specs != selection["specs"] or selection["source_sha256"] != hashes() or
                selection["rl_identity"] != rl_identity() or
                selection["evaluator_sha256"] != sha(__file__) or
                selection["evaluator_dependencies_sha256"] != evaluator_dependencies() or
                selection["protocol_sha256"] != sha(RESEARCH / "PROTOCOL.md")):
            raise ValueError("Independent source/specification changed after selection")
        if any(m["selection_sha256"] != sha(RESEARCH / "selection.json") for m in manifests.values()):
            raise ValueError("Independent run used a different selection")
    candidates = [label for label in CANDIDATES if label in specs]
    if len(candidates) != (2 if phase == "development" else 1):
        raise ValueError("Wrong candidate count")
    accepted = []
    for label in candidates:
        comparisons[label] = {name: comparison(report, label, REFERENCE) for name, report in reports.items()}
        random, stress = comparisons[label].values()
        if candidate_passes(random, stress, phase):
            accepted.append(label)
    selected = choose_candidate(accepted, {label: row["mean_time_s"]
        for label, row in reports[next(iter(declared_seeds))]["summaries"].items()})
    decision = dict(phase=phase, selected=selected, passed=selected is not None,
        source_sha256=hashes(), rl_identity=rl_identity(), evaluator_sha256=sha(__file__),
        evaluator_dependencies_sha256=evaluator_dependencies(),
        protocol_sha256=sha(RESEARCH / "PROTOCOL.md"), comparisons_vs_incumbent=comparisons,
        summaries={name: report["summaries"] for name, report in combined.items()},
        evidence_sha256=inputs,
        scope="Local paired generated scenes; RL reference fixed separately; all failures retained")
    if phase == "development" and selected:
        chosen = {label: specs[label] for label in ("compact_baseline", "compact_clear_before_probe", REFERENCE, selected)}
        selection = dict(role="Frozen one candidate before independent cases", selected=selected, specs=chosen,
            source_sha256=hashes(), rl_identity=rl_identity(), evaluator_sha256=sha(__file__),
            evaluator_dependencies_sha256=evaluator_dependencies(),
            protocol_sha256=decision["protocol_sha256"], reserved_seeds=INDEPENDENT,
            development_evidence_sha256=inputs)
        for path in (RESEARCH / "selection.json", RESEARCH / "selection-specs.json"):
            if path.exists():
                raise ValueError("Preserve prior selection")
        write_json(RESEARCH / "selection.json", selection)
        write_json(RESEARCH / "selection-specs.json", chosen)
    if phase == "confirmation":
        decision["selection_sha256"] = sha(RESEARCH / "selection.json")
    write_json(destination, decision)
    print(json.dumps({"phase": phase, "selected": selected, "comparisons": comparisons}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    main(parser.parse_args().phase)

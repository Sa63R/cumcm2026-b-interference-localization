"""Apply the predeclared R9 paired gates, with an independently frozen RL arm."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import hashes, write_json
from experiments.evaluate_q4_round2 import comparison

RESEARCH = ROOT / "research/q4_joint_visibility"
RESULTS = ROOT / "results/q4_joint_visibility"
REFERENCE = "compact_clear_before_probe"
CANDIDATES = ("compact_joint_probe", "compact_joint_optical")
DEVELOPMENT = {"development": list(range(618001, 618025)),
               "development-stress": list(range(618031, 618045))}
INDEPENDENT = {"confirmation": list(range(618101, 618229)),
               "stress": list(range(618301, 618385))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rl_identity():
    paths = [ROOT / "experiments/q4_frozen_rl_worker.py",
             ROOT / "experiments/run_q4_frozen_rl_compare.py"]
    paths += [RESEARCH / "rl_reference" / name for name in ("reference.json", "source.zip", "checkpoint.pt")]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def read_set(name, evidence):
    directory = RESULTS / name
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if manifest["source_sha256"] != hashes():
        raise ValueError("Experiment source changed")
    for filename in ("independent_audit.json", "clear_before_probe_audit.json", "joint_visibility_prefix_audit.json"):
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
    for filename in ("manifest.json", "freeze.json", "summary.json", "source.zip", "independent_audit.json",
                     "clear_before_probe_audit.json", "joint_visibility_prefix_audit.json"):
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
                frozen["protocol_sha256"] != sha(RESEARCH / "PROTOCOL.md")):
            raise ValueError("Development source or RL identity differs from freeze")
    else:
        selection = json.loads((RESEARCH / "selection.json").read_bytes())
        if (specs != selection["specs"] or selection["source_sha256"] != hashes() or
                selection["rl_identity"] != rl_identity() or
                selection["evaluator_sha256"] != sha(__file__) or
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
        mean_gate = random["mean_saved_s"] > 0 if phase == "development" else (
            random["saving_ci95_s"][0] > 0 and random["mean_reduction_fraction"] >= .005)
        if (random["all_clear"] and stress["all_clear"] and mean_gate and stress["mean_saved_s"] >= 0
                and random["p95_ratio"] <= 1.05 and stress["p95_ratio"] <= 1.05):
            accepted.append(label)
    selected = None
    if accepted:
        random = reports[next(iter(declared_seeds))]["summaries"]
        selected = min(accepted, key=lambda label: random[label]["mean_time_s"])
        if len(accepted) == 2 and abs(random[CANDIDATES[0]]["mean_time_s"]-random[CANDIDATES[1]]["mean_time_s"]) < 5.:
            selected = CANDIDATES[0]
    decision = dict(phase=phase, selected=selected, passed=selected is not None,
        source_sha256=hashes(), rl_identity=rl_identity(), evaluator_sha256=sha(__file__),
        protocol_sha256=sha(RESEARCH / "PROTOCOL.md"), comparisons_vs_incumbent=comparisons,
        summaries={name: report["summaries"] for name, report in combined.items()},
        evidence_sha256=inputs,
        scope="Local paired generated scenes; RL reference fixed separately; all failures retained")
    if phase == "development" and selected:
        chosen = {label: specs[label] for label in ("compact_baseline", "compact_combo", REFERENCE, selected)}
        selection = dict(role="Frozen one candidate before independent cases", selected=selected, specs=chosen,
            source_sha256=hashes(), rl_identity=rl_identity(), evaluator_sha256=sha(__file__),
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

"""Frozen R16 two-candidate decision against R12, requiring complete paired RL evidence."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import hashes, digest, report_rows
from experiments.evaluate_q4_round2 import comparison

RESEARCH = ROOT / "research/q4_feedback_symmetry"
RESULTS = ROOT / "results/q4_feedback_symmetry"
RL_ROOT = ROOT.parent / "q4-r9-joint-visibility"
REFERENCE = "compact_joint_continuation"
CANDIDATES = ("compact_feedback_mean", "compact_feedback_centers")
RL_LABEL = "compact_macro_ppo512"
SPECS = {
    "compact_baseline": {"entrypoint": "strategies.q4_cover_search:run_q4_cover_search", "kwargs": {"profile": "compact_22", "schedule": "joint", "max_expansions": 200}},
    "compact_joint_probe": {"entrypoint": "strategies.q4_joint_visibility:run_q4_joint_visibility", "kwargs": {"config": "probe", "max_expansions": 200}},
    REFERENCE: {"entrypoint": "strategies.q4_joint_continuation:run_q4_joint_continuation", "kwargs": {"config": "after_active_miss_optical", "max_expansions": 200}},
    CANDIDATES[0]: {"entrypoint": "strategies.q4_feedback_symmetry:run_q4_feedback_symmetry", "kwargs": {"config": "bearing_mean", "max_expansions": 200}},
    CANDIDATES[1]: {"entrypoint": "strategies.q4_feedback_symmetry:run_q4_feedback_symmetry", "kwargs": {"config": "early_centers", "max_expansions": 200}},
}
DEVELOPMENT = {"development": list(range(625001, 625025)), "development-stress": list(range(625031, 625045))}
INDEPENDENT = {"confirmation": list(range(625101, 625229)), "stress": list(range(625301, 625385))}
AUDITS = {"independent_audit.json": None, "joint_visibility_prefix_audit.json": "compact_joint_probe",
          "joint_continuation_prefix_audit.json": REFERENCE, "feedback_symmetry_mean_audit.json": CANDIDATES[0],
          "feedback_symmetry_centers_audit.json": CANDIDATES[1]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def selected_specs(selected):
    if selected not in CANDIDATES:
        raise ValueError("Unknown selected candidate")
    return {label: SPECS[label] for label in ("compact_baseline", "compact_joint_probe", REFERENCE, selected)}


def stage_specs(name):
    if name in DEVELOPMENT:
        return SPECS
    if name not in INDEPENDENT:
        raise ValueError("Unknown dataset")
    selection = read(RESEARCH / "selection.json")
    expected = selected_specs(selection["selected"])
    if selection.get("selected_specs") != expected:
        raise ValueError("Independent selection specifications differ")
    return expected


def choose_candidate(accepted, means):
    if not set(accepted).issubset(CANDIDATES) or len(accepted) != len(set(accepted)):
        raise ValueError("Invalid accepted candidate list")
    if not accepted:
        return None
    selected = min(accepted, key=lambda label: means[label])
    if len(accepted) == 2 and abs(means[CANDIDATES[0]] - means[CANDIDATES[1]]) < 5.:
        selected = CANDIDATES[0]
    return selected


def evaluator_dependencies():
    return {"experiments/evaluate_q4_round2.py": sha(ROOT / "experiments/evaluate_q4_round2.py")}


def rl_identity():
    names = ["experiments/q4_frozen_rl_worker.py", "experiments/run_q4_frozen_rl_compare.py"]
    names += ["research/q4_joint_visibility/rl_reference/" + n for n in ("reference.json", "source.zip", "checkpoint.pt")]
    result = {n: sha(RL_ROOT / n) for n in names}
    ref = read(RL_ROOT / names[2])
    if (ref["source_archive_sha256"] != result[names[3]] or ref["checkpoint_sha256"] != result[names[4]]
            or ref["training_enabled"] is not False):
        raise ValueError("RL source/model differs from the fixed reference")
    return result


def development_identity():
    """Read identities only; never generates cases, freezes or runs a strategy."""
    if read(RESEARCH / "development-specs.json") != SPECS:
        raise ValueError("Five fixed development specifications differ")
    return dict(source_sha256=hashes(), rl_identity=rl_identity(), evaluator_sha256=sha(__file__),
                evaluator_dependencies_sha256=evaluator_dependencies(), protocol_sha256=sha(RESEARCH / "PROTOCOL.md"),
                specs_sha256=sha(RESEARCH / "development-specs.json"), specs=SPECS)


def candidate_passes(random, stress, phase):
    if phase not in {"development", "confirmation"}:
        raise ValueError("Unknown phase")
    mean_gate = random["mean_saved_s"] > 0 if phase == "development" else (
        random["saving_ci95_s"][0] > 0 and random["mean_reduction_fraction"] >= .005)
    return bool(random["all_clear"] and stress["all_clear"] and mean_gate and stress["mean_saved_s"] >= 0
                and random["p95_ratio"] <= 1.05 and stress["p95_ratio"] <= 1.05)


def validate_matrix(rows, seeds, labels, physical_stage):
    expected = {(seed, label) for seed in seeds for label in labels}
    keys = [(r["seed"], r["strategy"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete, duplicated or unexpected paired matrix")
    base = {r["seed"]: r for r in rows if r["strategy"] == "compact_baseline"}
    for row in rows:
        lower = row["common_lower_bound_s"]
        if (row["stage"] != physical_stage or not math.isfinite(lower) or lower <= 0
                or not math.isfinite(row["virtual_time_s"]) or row["virtual_time_s"] < 0
                or row["penalized_time_s"] != (row["virtual_time_s"] if row["successful"] else 360000)
                or row["time_over_lower_bound"] != row["virtual_time_s"] / lower
                or row["penalized_time_over_lower_bound"] != row["penalized_time_s"] / lower
                or row["case_sha256"] != base[row["seed"]]["case_sha256"]
                or lower != base[row["seed"]]["common_lower_bound_s"]):
            raise ValueError("Stage, paired case/bound or failure penalty differs")


def validate_audit(audit, label, seeds, rows, directory):
    expected = {(label, seed) for seed in seeds} if label else {(r["strategy"], r["seed"]) for r in rows}
    if (audit.get("all_passed") is not True or audit.get("records") != len(expected)
            or audit.get("passed_records") != len(expected) or audit.get("errors")
            or len(audit.get("audits", [])) != len(expected)
            or any(a.get("passed") is not True for a in audit["audits"])):
        raise ValueError("Incomplete or failed audit")
    if label is None:
        pairs = [(a["strategy"], a["case_id"]) for a in audit["audits"]]
        if len(set(pairs)) != len(expected) or set(pairs) != {(r["strategy"], r["case_id"]) for r in rows}:
            raise ValueError("Physical audit matrix differs")
        row_by_case = {(r["strategy"], r["case_id"]): r for r in rows}
        for item in audit["audits"]:
            current = row_by_case[item["strategy"], item["case_id"]]
            if any(item.get(key) != current[key] for key in
                   ("successful", "virtual_time_s", "common_lower_bound_s", "time_over_lower_bound")):
                raise ValueError("Physical audit contains stale outcome, time or bound")
        return
    keys = [(a["strategy"], a["seed"]) for a in audit["audits"]]
    if len(set(keys)) != len(expected) or set(keys) != expected:
        raise ValueError("Prefix audit skipped or duplicated a record")
    required = {f"records/{label}-{seed}.json.gz" for seed in seeds} | {"manifest.json", "freeze.json", "independent_audit.json"}
    inputs = audit.get("input_sha256", {})
    if not required.issubset(inputs):
        raise ValueError("Prefix audit lacks complete input hashes")
    for relative, expected_hash in inputs.items():
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory.resolve()) or sha(target) != expected_hash:
            raise ValueError("Prefix audit input changed or escapes its directory")


def validate_rl_pairing(combined, report, seeds, physical_stage, specs=None):
    specs = SPECS if specs is None else specs
    if combined.get("rl_audits_all_passed") is not True:
        raise ValueError("RL audits failed or incomplete")
    rows = combined["rows"]
    validate_matrix(rows, seeds, (*specs, RL_LABEL), physical_stage)
    if [r for r in rows if r["strategy"] != RL_LABEL] != report["rows"]:
        raise ValueError("RL report changed state rows")
    core = {k: combined[k] for k in ("summaries", "paired_vs_compact_baseline", "rows")}
    if report_rows(sorted(rows, key=lambda r: (r["seed"], r["strategy"]))) != core:
        raise ValueError("RL statistics differ from complete paired rows")


def read_set(name, evidence, frozen=None):
    declared = {**DEVELOPMENT, **INDEPENDENT}
    if name not in declared:
        raise ValueError("Unknown dataset")
    frozen = frozen or read(RESEARCH / "development-freeze.json")
    seeds, directory = declared[name], RESULTS / name
    specs = stage_specs(name)
    audits = {filename: label for filename, label in AUDITS.items() if label is None or label in specs}
    physical_stage = "stress" if name.endswith("stress") else "confirmation" if name == "confirmation" else "pilot"
    manifest, stage_freeze = read(directory / "manifest.json"), read(directory / "freeze.json")
    if (manifest["source_sha256"] != frozen["source_sha256"] or manifest["specs"] != specs
            or manifest["seeds"] != seeds or manifest["stage"] != physical_stage
            or manifest["failure_penalty_s"] != 360000 or stage_freeze["manifest_sha256"] != digest(manifest)):
        raise ValueError("Dataset differs from the frozen stage/specification")
    if name in INDEPENDENT and manifest["selection_sha256"] != sha(RESEARCH / "selection.json"):
        raise ValueError("Independent run used a different selection")
    with zipfile.ZipFile(directory / "source.zip") as archive:
        for relative, expected in frozen["source_sha256"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError("Archived source differs")
    report = read(directory / "summary.json")
    validate_matrix(report["rows"], seeds, specs, physical_stage)
    if report_rows(sorted(report["rows"], key=lambda r: (r["seed"], r["strategy"]))) != report:
        raise ValueError("State statistics differ from complete paired rows")
    expected_files = {f"{label}-{seed}.json.gz" for label in specs for seed in seeds}
    if {p.name for p in (directory / "records").glob("*.json.gz")} != expected_files:
        raise ValueError("Incomplete raw state records")
    row_map = {(r["strategy"], r["seed"]): r for r in report["rows"]}
    for label in specs:
        for seed in seeds:
            path = directory / "records" / f"{label}-{seed}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            if record["row"] != row_map[label, seed] or record["spec"] != specs[label]:
                raise ValueError("Raw state identity or row differs")
            evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for filename, label in audits.items():
        validate_audit(read(directory / filename), label, seeds, report["rows"], directory)

    rl_directory = RESULTS / (name + "-rl")
    rf, combined = read(rl_directory / "freeze.json"), read(rl_directory / "comparison.json")
    identity = frozen["rl_identity"]
    reference = read(RL_ROOT / "research/q4_joint_visibility/rl_reference/reference.json")
    if (rf["reference_sha256"] != identity["research/q4_joint_visibility/rl_reference/reference.json"]
            or rf["worker_sha256"] != identity["experiments/q4_frozen_rl_worker.py"]
            or rf["runner_sha256"] != identity["experiments/run_q4_frozen_rl_compare.py"]
            or rf["state_manifest_sha256"] != sha(directory / "manifest.json")
            or rf["state_summary_sha256"] != sha(directory / "summary.json")
            or rf["reference_arm"] is not False or rf["spec"] != reference["spec"]
            or combined["freeze_sha256"] != sha(rl_directory / "freeze.json")):
        raise ValueError("RL source/spec/case-matrix identity differs")
    base = [r for r in report["rows"] if r["strategy"] == "compact_baseline"]
    case = lambda r: (r["case_sha256"], r["common_lower_bound_s"], r["stage"])
    if Counter(map(case, rf["cases"])) != Counter(map(case, base)):
        raise ValueError("RL freeze lacks the full paired case matrix")
    validate_rl_pairing(combined, report, seeds, physical_stage, specs)
    rl_rows = {r["seed"]: r for r in combined["rows"] if r["strategy"] == RL_LABEL}
    raw_rl = list(rl_directory.glob("worker-*/*.json.gz"))
    seen = set()
    for path in raw_rl:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        seed = record["row"]["seed"]
        if (seed in seen or record["row"] != rl_rows.get(seed) or record["spec"] != rf["spec"]
                or record["audit"].get("passed") is not True
                or record["policy_reference_sha256"] != rf["reference_sha256"]):
            raise ValueError("Raw RL identity, audit or pairing differs")
        seen.add(seed)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    if seen != set(seeds) or len(raw_rl) != len(seeds):
        raise ValueError("Incomplete raw RL records")
    for filename in ("manifest.json", "freeze.json", "summary.json", "source.zip", *audits):
        path = directory / filename
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for filename in ("freeze.json", "comparison.json"):
        path = rl_directory / filename
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    return report, combined, manifest


def main(phase):
    if phase not in {"development", "confirmation"}:
        raise ValueError("Unknown phase")
    destination = RESEARCH / ("development-decision.json" if phase == "development" else "qualification.json")
    outputs = [destination] + ([RESEARCH / "selection.json", RESEARCH / "selection-specs.json"] if phase == "development" else [])
    if any(p.exists() for p in outputs):
        raise ValueError("Preserve prior decision and selection")
    identity, frozen = development_identity(), read(RESEARCH / "development-freeze.json")
    if any(frozen.get(k) != value for k, value in identity.items()):
        raise ValueError("Frozen source, RL, evaluator dependency, protocol or specification changed")
    if phase == "confirmation":
        selection = read(RESEARCH / "selection.json")
        if (selection.get("passed") is not True or selection["selected"] not in CANDIDATES
                or selection["selected"] != read(RESEARCH / "development-decision.json")["selected"]
                or selection.get("selected_specs") != selected_specs(selection["selected"])
                or read(RESEARCH / "selection-specs.json") != selection["selected_specs"]
                or selection["reserved_seeds"] != INDEPENDENT
                or any(selection.get(k) != value for k, value in identity.items())
                or selection["development_decision_sha256"] != sha(RESEARCH / "development-decision.json")
                or selection["development_freeze_sha256"] != sha(RESEARCH / "development-freeze.json")):
            raise ValueError("Independent selection or its evidence changed")
    declared = DEVELOPMENT if phase == "development" else INDEPENDENT
    evidence, reports, combined = {}, {}, {}
    for name in declared:
        reports[name], combined[name], _ = read_set(name, evidence, frozen)
    candidates = CANDIDATES if phase == "development" else (selection["selected"],)
    comparisons = {label: {name: comparison(r, label, REFERENCE) for name, r in reports.items()} for label in candidates}
    accepted = [label for label in candidates if candidate_passes(*comparisons[label].values(), phase)]
    selected = choose_candidate(accepted, {label: reports[next(iter(declared))]["summaries"][label]["mean_time_s"] for label in candidates})
    passed = selected is not None
    decision = dict(phase=phase, passed=passed, selected=selected,
        primary_reference=REFERENCE, **identity, comparisons_vs_incumbent=comparisons,
        summaries={name: r["summaries"] for name, r in combined.items()}, evidence_sha256=evidence,
        development_freeze_sha256=sha(RESEARCH / "development-freeze.json"), report_complete=True,
        scope="Complete local paired evidence; fixed RL; failures retained; no official score claim")
    if phase == "confirmation":
        decision["selection_sha256"] = sha(RESEARCH / "selection.json")
    write_new(destination, decision)
    if phase == "development" and passed:
        write_new(RESEARCH / "selection.json", dict(passed=True, selected=selected, selected_specs=selected_specs(selected), **identity,
            role="One fixed candidate; root must separately authorize independent runs", reserved_seeds=INDEPENDENT,
            development_decision_sha256=sha(destination), development_freeze_sha256=sha(RESEARCH / "development-freeze.json"),
            development_evidence_sha256=evidence))
        write_new(RESEARCH / "selection-specs.json", selected_specs(selected))
    print(json.dumps({"phase": phase, "selected": decision["selected"], "comparisons": comparisons}))
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    main(parser.parse_args().phase)

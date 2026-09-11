"""Frozen candidate registry and archive checks; never imports/runs a simulator.

This tool reads already produced JSON/gzip records only. It does not establish
that an operator has never inspected a held-out case, sign records, replay the
physics, prove geometric certificates, or replace the final statistical report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

DIRECTIONS = ("state", "rl", "geo")
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
SCHEMA = 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_from(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timestamp needs an explicit timezone")
    return result


def write_once(path, value):
    """Exclusive creation, including a canonical content digest (not a signature)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(value, payload_sha256=digest(value))
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def read_sealed(path, kind):
    result = read_json(path)
    payload = {k: v for k, v in result.items() if k != "payload_sha256"}
    if result.get("payload_sha256") != digest(payload):
        raise ValueError("Record content digest mismatch")
    if result.get("kind") != kind or result.get("schema_version") != SCHEMA:
        raise ValueError("Wrong registry/selection record kind or schema")
    if result.get("tool_sha256") != file_hash(__file__):
        raise ValueError("Selection tool changed since registration")
    return result


def live_identity(source, spec, protocol):
    """Same byte hash inventory as research_v1_eval.identity, with explicit cwd."""
    source = Path(source)
    paths = sorted((source / "src").rglob("*.py"))
    if not paths:
        raise ValueError("Source directory has no Python sources")
    paths += [source / "experiments/research_v1_eval.py",
              source / "experiments/run_q3_comparison.py"]
    checkpoints = {key: file_hash(path_from(value, source))
                   for key, value in spec.get("kwargs", {}).items()
                   if key in ("checkpoint", "weights") and value}
    return dict(source_sha256={p.relative_to(source).as_posix(): file_hash(p) for p in paths},
                spec_sha256=digest(spec), checkpoint_sha256=checkpoints,
                protocol_sha256=digest(protocol))


def entry_record(entry, base, protocol, registered_at):
    paths = {key: path_from(entry[key], base)
             for key in ("source_dir", "spec", "freeze", "environment")}
    spec, frozen, environment = (read_json(paths[key]) for key in ("spec", "freeze", "environment"))
    ident = live_identity(paths["source_dir"], spec, protocol)
    if frozen["identity"] != ident or frozen["strategy"] != spec["name"]:
        raise ValueError(f"Live source/spec/checkpoint does not match freeze: {entry['id']}")
    if read_json(paths["source_dir"] / "research/v1_protocol.json") != protocol:
        raise ValueError("Source directory uses a different protocol")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    cwd=paths["source_dir"], text=True).strip()
    if frozen["git_commit"] != commit:
        raise ValueError("Frozen Git commit differs from actual source HEAD")
    if timestamp(frozen["frozen_at"]) > timestamp(registered_at):
        raise ValueError("Candidate freeze must precede registry")
    if environment.get("system") != "Linux" or not environment.get("python_version"):
        raise ValueError("Register the actual Linux/Python environment")
    if not isinstance(environment.get("packages"), dict):
        raise ValueError("Environment packages must be a name-to-version object")
    return dict(id=entry["id"], **{k: str(v) for k, v in paths.items()},
                strategy=spec["name"], spec_value=spec, identity=ident, git_commit=commit,
                frozen_at=frozen["frozen_at"], environment_value=environment,
                input_sha256={key: file_hash(paths[key]) for key in ("spec", "freeze", "environment")})


def shared_environment_and_physics(entries):
    first = next(iter(entries.values()))
    reference = first["environment_value"]
    def physics(entry):
        hashes = entry["identity"]["source_sha256"]
        return {k: v for k, v in hashes.items() if k.startswith(("src/simulation/", "src/simulator_client/"))
                or k in ("experiments/research_v1_eval.py", "experiments/run_q3_comparison.py")}
    expected = physics(first)
    if not any(k.startswith("src/simulation/") for k in expected):
        raise ValueError("Missing frozen simulation implementation")
    for entry in entries.values():
        if entry["environment_value"] != reference:
            raise ValueError("Candidate environments differ (including package versions)")
        if physics(entry) != expected:
            raise ValueError("Scene generation, physical client or common harness bytes differ")
    return dict(environment_sha256=digest(reference), shared_physics_sha256=expected)


def register(plan_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    plan_path = Path(plan_path).resolve()
    plan, base, now = read_json(plan_path), plan_path.parent, utc_now()
    protocol_path, rule_path = (path_from(plan[key], base) for key in ("protocol", "rule"))
    protocol = read_json(protocol_path)
    if set(plan["directions"]) != set(DIRECTIONS):
        raise ValueError("Register exactly state, rl and geo directions")
    specifications = [plan["baseline"]]
    directions = {}
    for name in DIRECTIONS:
        branch = plan["directions"][name]
        candidates = branch["candidates"]
        if not 1 <= len(candidates) <= 3:
            raise ValueError("Each direction needs one to three candidates")
        ids = [entry["id"] for entry in candidates]
        if branch["fallback"] not in ids:
            raise ValueError("Fallback must be a registered candidate")
        directions[name] = dict(tie_order=ids, fallback=branch["fallback"])
        specifications.extend(candidates)
    ids = [entry["id"] for entry in specifications]
    if any(not isinstance(name, str) or not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every candidate and baseline needs a unique nonempty id")
    entries = {entry["id"]: entry_record(entry, base, protocol, now) for entry in specifications}
    if len({entry["strategy"] for entry in entries.values()}) != len(entries):
        raise ValueError("Specification names must also be unique for archival audits")
    baseline = plan["baseline"]["id"]
    if entries[baseline]["strategy"] != protocol["primary_reference"]:
        raise ValueError("Baseline must be the protocol's declared primary reference")
    if "rollout_config" in protocol:
        reference_spec = dict(name=protocol["primary_reference"], entrypoint="strategies:run_search",
                              kwargs=dict(variant="rollout", rollout_config=protocol["rollout_config"]))
        if entries[baseline]["spec_value"] != reference_spec:
            raise ValueError("Baseline specification differs from predeclared rollout reference")
    return write_once(output, dict(schema_version=SCHEMA, kind="candidate_registry", registered_at=now,
        tool_sha256=file_hash(__file__), plan_path=str(plan_path), plan_sha256=file_hash(plan_path),
        protocol_path=str(protocol_path), protocol_value=protocol, protocol_sha256=digest(protocol),
        rule_path=str(rule_path), rule_sha256=file_hash(rule_path), baseline=baseline,
        directions=directions, entries=entries, **shared_environment_and_physics(entries),
        access_scope="Registration reads no scenario records. Operator must register before first selection-case access."))


def verify_registry(path):
    registry = read_sealed(path, "candidate_registry")
    if file_hash(registry["rule_path"]) != registry["rule_sha256"]:
        raise ValueError("Selection rule changed")
    protocol = registry["protocol_value"]
    if read_json(registry["protocol_path"]) != protocol or digest(protocol) != registry["protocol_sha256"]:
        raise ValueError("Protocol changed")
    for candidate in registry["entries"].values():
        if entry_record(candidate, Path(path).parent, protocol, registry["registered_at"]) != candidate:
            raise ValueError("Registered source/spec/checkpoint/environment/freeze changed")
    shared_environment_and_physics(registry["entries"])
    return registry


def percentile(values, fraction=.95):
    """The exact linear interpolation definition used by the shared report."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def load_evaluation(directory, entry, split, protocol):
    """Bind complete rows to individual gzip archives and post-run truth hashes."""
    directory = Path(directory).resolve()
    manifest, rows = (read_json(directory / name) for name in ("manifest.json", "rows.json"))
    partition = protocol["partitions"][split]
    seeds = list(range(partition["seed_start"], partition["seed_stop_exclusive"]))
    if manifest != dict(identity=entry["identity"], strategy=entry["strategy"], split=split, seeds=seeds):
        raise ValueError("Evaluation manifest differs from frozen identity/full partition")
    if sorted(row["seed"] for row in rows) != seeds or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Missing, duplicate or unexpected row seed/case id")
    filenames = {f"case-{seed}.json.gz" for seed in seeds}
    if {p.name for p in directory.glob("case-*.json.gz")} != filenames:
        raise ValueError("Missing or unexpected per-case archive")
    archive_hashes = {}
    for row in rows:
        if row["strategy"] != entry["strategy"]:
            raise ValueError("Row strategy does not match registered candidate")
        metrics = [row[key] for key in (*COMPONENTS, "virtual_time_s", "penalized_time_s", "program_runtime_s")]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in metrics):
            raise ValueError("Nonfinite or negative evaluation metric")
        for key in ("source_total", "cleared_total", "failed_clear_count", "measurement_count", "action_count"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
                raise ValueError("Invalid recorded count")
        successful = bool(row["all_cleared"] and row["completion_certified"] and row["accepted_exit"]
                          and row["cleared_total"] == row["source_total"] and not row["errors"])
        if row["successful"] is not successful:
            raise ValueError("Success flag contradicts completion facts")
        penalty = row["virtual_time_s"] if successful else protocol["limits"]["virtual_seconds_per_case"]
        if row["penalized_time_s"] != penalty:
            raise ValueError("Incorrect failure penalty or successful cost")
        archive = directory / f"case-{row['seed']}.json.gz"
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        if record["row"] != row or record["spec"] != entry["spec_value"]:
            raise ValueError("Case gzip differs from rows/specification")
        evaluation = record["evaluation"]
        truth = evaluation["ground_truth"]
        if record.get("evaluation_phase") != "after_policy_termination" or evaluation.get("kind") != "local_research_only":
            raise ValueError("Archive is not explicitly post-policy local research evaluation")
        if (truth["seed"] != row["seed"] or truth["case_id"] != row["case_id"]
                or truth["problem"] != 3 or digest(truth) != row["case_sha256"]):
            raise ValueError("Archived ground truth does not match row scenario hash")
        for key in ("all_cleared", "source_total", "cleared_total", "failed_clear_count", "measurement_count", "action_count", "virtual_time_s"):
            if evaluation[key] != row[key]:
                raise ValueError("Evaluator and row disagree")
        if len(record["history"]) != row["action_count"]:
            raise ValueError("History action count differs from row")
        if any(evaluation["time_breakdown_s"][key] != row[key] for key in COMPONENTS):
            raise ValueError("Evaluator and row fee components differ")
        if abs(sum(row[key] for key in COMPONENTS) - row["virtual_time_s"]) > 2e-6:
            raise ValueError("Recorded fee components do not sum to virtual time")
        archive_hashes[archive.name] = file_hash(archive)
    values = [row["penalized_time_s"] for row in rows]
    return dict(directory=str(directory), rows=rows,
                evidence=dict(manifest_sha256=file_hash(directory / "manifest.json"),
                              rows_sha256=file_hash(directory / "rows.json"), archives_sha256=archive_hashes),
                summary=dict(runs=len(rows), successful_runs=sum(r["successful"] for r in rows),
                             failed_clear_count=sum(r["failed_clear_count"] for r in rows),
                             mean_penalized_s=statistics.mean(values), p95_penalized_s=percentile(values)))


def load_partition(index, registry, split, wanted):
    if set(index) != set(wanted):
        raise ValueError("Evaluation index must contain exactly all required registered identities")
    data = {key: load_evaluation(index[key], registry["entries"][key], split, registry["protocol_value"])
            for key in wanted}
    baseline = data[registry["baseline"]]
    expected = {row["seed"]: (row["case_id"], row["case_sha256"]) for row in baseline["rows"]}
    for item in data.values():
        if {r["seed"]: (r["case_id"], r["case_sha256"]) for r in item["rows"]} != expected:
            raise ValueError("Methods do not share every scenario SHA/case identity")
    return data


def reliable(summary):
    return summary["successful_runs"] == summary["runs"] and summary["failed_clear_count"] == 0


def select(registry, data):
    baseline = data[registry["baseline"]]["summary"]
    usable = reliable(baseline) and baseline["p95_penalized_s"] > 0
    decisions = {}
    for direction, branch in registry["directions"].items():
        screens = {}
        for key in branch["tie_order"]:
            summary = data[key]["summary"]
            ratio = summary["p95_penalized_s"] / baseline["p95_penalized_s"] if baseline["p95_penalized_s"] > 0 else None
            screens[key] = dict(**summary, p95_time_ratio=ratio,
                                passed=bool(usable and reliable(summary) and ratio <= 1.05))
        passed = [key for key in branch["tie_order"] if screens[key]["passed"]]
        # min is stable: exact floating numerical ties follow frozen array order.
        chosen = min(passed, key=lambda k: screens[k]["mean_penalized_s"]) if passed else branch["fallback"]
        decisions[direction] = dict(selected=chosen, passed_extended_screening=bool(passed),
                                    fallback_used=not bool(passed), baseline_usable=usable, screens=screens)
    return decisions


def evidence(data):
    return {key: {k: v for k, v in value.items() if k != "rows"} for key, value in data.items()}


def read_index(path):
    path = Path(path).resolve()
    return {key: str(path_from(value, path.parent)) for key, value in read_json(path).items()}


def finalize(registry_path, index_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    registry = verify_registry(registry_path)
    data = load_partition(read_index(index_path), registry, "validation_extended", registry["entries"])
    return write_once(output, dict(schema_version=SCHEMA, kind="selection_decision", finalized_at=utc_now(),
        tool_sha256=file_hash(__file__), registry_path=str(Path(registry_path).resolve()),
        registry_file_sha256=file_hash(registry_path), registry_payload_sha256=registry["payload_sha256"],
        rule_sha256=registry["rule_sha256"], evaluations=evidence(data), decisions=select(registry, data)))


def audit_final(registry_path, selection_path, index_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    registry = verify_registry(registry_path)
    selected = read_sealed(selection_path, "selection_decision")
    if (selected["registry_file_sha256"] != file_hash(registry_path)
            or selected["registry_payload_sha256"] != registry["payload_sha256"]
            or selected["rule_sha256"] != registry["rule_sha256"]):
        raise ValueError("Selection decision belongs to a different registry/rule")
    if timestamp(selected["finalized_at"]) < timestamp(registry["registered_at"]):
        raise ValueError("Selection predates candidate registration")
    old = load_partition({k: v["directory"] for k, v in selected["evaluations"].items()},
                         registry, "validation_extended", registry["entries"])
    if evidence(old) != selected["evaluations"] or select(registry, old) != selected["decisions"]:
        raise ValueError("Selection evidence or deterministic decision changed")
    wanted = [registry["baseline"]] + [selected["decisions"][d]["selected"] for d in DIRECTIONS]
    index_path = Path(index_path).resolve()
    index = read_json(index_path)
    if set(index) != {"final_random", "final_stress"}:
        raise ValueError("Final audit requires exactly random and stress partitions")
    audits = {}
    for split in ("final_random", "final_stress"):
        paths = {k: str(path_from(v, index_path.parent)) for k, v in index[split].items()}
        data = load_partition(paths, registry, split, wanted)
        audits[split] = dict(evaluations=evidence(data),
                             baseline_usable=reliable(data[registry["baseline"]]["summary"]),
                             all_methods_reliable=all(reliable(v["summary"]) for v in data.values()))
    return write_once(output, dict(schema_version=SCHEMA, kind="final_identity_archive_audit", audited_at=utc_now(),
        tool_sha256=file_hash(__file__), registry_file_sha256=file_hash(registry_path),
        selection_file_sha256=file_hash(selection_path), selected={d: selected["decisions"][d] for d in DIRECTIONS},
        partitions=audits, identity_and_archive_checks_passed=True,
        all_final_methods_reliable=all(item["all_methods_reliable"] for item in audits.values()),
        scope="Identity/archive audit only; independent physical replay, geometric certificates, lower bounds and statistical acceptance remain separate."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("--input", type=Path, required=True)
    registration.add_argument("--output", type=Path, required=True)
    for name in ("finalize", "audit-final"):
        command = commands.add_parser(name)
        command.add_argument("--registry", type=Path, required=True)
        command.add_argument("--evaluations", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "audit-final":
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.input, args.output)
    elif args.command == "finalize":
        result = finalize(args.registry, args.evaluations, args.output)
    else:
        result = audit_final(args.registry, args.selection, args.evaluations, args.output)
    print(json.dumps(dict(output=str(args.output), kind=result["kind"],
                          payload_sha256=result["payload_sha256"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Observation-prefix geometry diagnostic on the two already opened Q4 sets.

This does not replay a policy or estimate counterfactual task time. Evaluation,
truth and row payloads are skipped lexically, never decoded or consulted.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from localization import CandidateRegion
from planning.joint_visibility_region import joint_visibility_outer
from experiments.audit_q4_joint_visibility import audit_joint_visibility_certificate

ALLOWED_SEEDS = {
    "combination-development": list(range(610001, 610025)),
    "combination-development-stress": list(range(610031, 610045)),
}
LABEL = "compact_combo"
EXPECTED_SPEC = {"entrypoint": "strategies.q4_range_scheduling:run_q4_range_scheduling",
                 "kwargs": {"config": "onroute", "max_expansions": 200}}
IDENTITY_FILES = ("src/geometry/__init__.py", "src/localization/__init__.py",
    "src/planning/joint_visibility_region.py", "experiments/audit_q4_joint_visibility.py",
    "experiments/diagnose_q4_joint_visibility.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def skip_value(text, at):
    """Skip a JSON value without interpreting its content or keys."""
    if text[at] not in '[{"':
        while at < len(text) and text[at] not in ",}] \t\r\n":
            at += 1
        return at
    stack, quoted, escaped = [], False, False
    if text[at] == '"':
        quoted = True
        at += 1
    else:
        stack.append(text[at])
        at += 1
    while at < len(text):
        char = text[at]
        at += 1
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
                if not stack:
                    return at
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (stack.pop(), char) not in (("[", "]"), ("{", "}")):
                raise ValueError("Invalid skipped JSON structure")
            if not stack:
                return at
    raise ValueError("Unterminated skipped JSON value")


def observation_record(path):
    # Decompression is unavoidable for a gzip object. Forbidden subtrees are
    # skipped structurally, without JSON-decoding their contents into objects.
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    decoder = json.JSONDecoder()
    at = len(text)-len(text.lstrip())
    if text[at] != "{":
        raise ValueError("Record must be a JSON object")
    at += 1
    output = {}
    while True:
        while text[at].isspace() or text[at] == ",":
            at += 1
        if text[at] == "}":
            break
        key, at = decoder.raw_decode(text, at)
        while text[at].isspace():
            at += 1
        if not isinstance(key, str) or text[at] != ":":
            raise ValueError("Invalid record field")
        at += 1
        while text[at].isspace():
            at += 1
        if key in {"summary", "history", "spec"}:
            if key in output:
                raise ValueError("Repeated observation field")
            output[key], at = decoder.raw_decode(text, at)
        else:
            at = skip_value(text, at)
    if set(output) != {"summary", "history", "spec"}:
        raise ValueError("Incomplete observation record")
    return output


def xy(value):
    return tuple(value[k] for k in ("x", "y")) if isinstance(value, dict) else tuple(value)


def resolver_inputs(record):
    summary = record["summary"]
    parameters = summary["strategy_parameters"]
    reports = summary["action_history"]
    wire = [a for a in record["history"] if a["action"] in {"/measure", "/clear"}
            and a["response"].get("accepted") is True]
    if len(reports) != len(wire):
        raise ValueError("Accepted wire/policy action counts differ")
    entries = defaultdict(list)
    for kind, key in (("chain_source", "chain_route_log"), ("early_service", "early_service_log")):
        for i, event in enumerate(parameters[key]):
            if kind == "chain_source" and event["selected_kind"] != "source":
                continue
            channel = event["selected_channel"] if kind == "chain_source" else event["channel"]
            prefix = event["after_actual_action_count"]
            if type(prefix) is not int or not 0 <= prefix <= len(wire):
                raise ValueError("Invalid resolver prefix")
            entries[prefix].append((channel, {"kind": kind, "event_index": i}))
    regions, positives, negatives, measurements = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    known, cleared, near = set(), set(), set()
    for n in range(len(wire)+1):
        channels = defaultdict(list)
        for channel, origin in entries.get(n, []):
            channels[channel].append(origin)
        for channel, origins in sorted(channels.items()):
            region = regions.get(channel)
            reason = None
            disk = region.enclosing_disk() if region is not None and region.vertices else None
            if channel in cleared:
                reason = "already_cleared"
            elif channel not in known or region is None or not region.vertices:
                reason = "no_canonical_positive_region"
            elif channel in near or disk.radius <= 19.9:
                reason = "already_ready"
            elif not negatives[channel]:
                reason = "no_negative_evidence"
            yield dict(channel=channel, after_actual_action_count=n, resolver_origins=origins,
                actual_prefix_sha256=digest(wire[:n]),
                same_channel_measurement_prefix=list(measurements[channel]),
                canonical_vertices=[list(p) for p in region.vertices] if region else [],
                positive_positions=[list(p) for p in positives[channel]],
                negative_positions=[list(p) for p in negatives[channel]],
                old_mec_radius_m=disk.radius if disk else None, skip_reason=reason)
        if n == len(wire):
            break
        action, report = wire[n], reports[n]
        response = action["response"]
        channel, position = action["channel"], xy(action["position"])
        kind = action["action"][1:]
        result = response["measure_result" if kind == "measure" else "clear_result"]
        if (report["action"] != kind or report["channel"] != channel
                or tuple(report["position"]) != position or report["result"] != result
                or abs(report["virtual_time_s"]-response["virtual_time_s"]) > 2e-6):
            raise ValueError("Policy/wire observation prefix differs")
        if kind == "clear":
            if result == "success":
                cleared.add(channel)
            continue
        if channel in cleared:
            continue
        item = dict(actual_action_index=n, wire_index=action["index"], position=list(position), result=result)
        if result == "direction":
            if report["bearing_deg"] != response["svd_deg"]:
                raise ValueError("Reported bearing differs from accepted wire")
            item["bearing_deg"] = response["svd_deg"]
            regions.setdefault(channel, CandidateRegion()).observe(position, response["svd_deg"])
            positives[channel].append(position)
            known.add(channel)
        elif result == "near":
            positives[channel].append(position)
            known.add(channel)
            near.add(channel)
        elif result == "no_signal":
            negatives[channel].append(position)
        else:
            raise ValueError("Unknown accepted measurement result")
        measurements[channel].append(item)


def metrics(events):
    good = [e for e in events if e.get("audit_passed") is True]
    def numbers(key):
        values = [e[key] for e in good]
        return dict(total=sum(values), mean=statistics.mean(values) if values else None,
                    median=statistics.median(values) if values else None, maximum=max(values, default=None))
    return dict(resolver_entries=len(events), skipped=dict(Counter(e["skip_reason"] for e in events if e.get("skip_reason"))),
        geometry_calls=sum("geometry_runtime_s" in e for e in events), audited=len(good),
        audit_failures=sum(e.get("audit_passed") is False for e in events),
        fallbacks=sum(e.get("geometry_status") == "fallback" for e in good),
        deleted_any_cells=sum(e["deleted_cells"] > 0 for e in good),
        excluded_any_old_vertex=sum(e["old_vertices_excluded"] > 0 for e in good),
        radius_reduced_over_1e_6_m=sum(e["radius_reduction_m"] > 1e-6 for e in good),
        became_ready=sum(e["became_ready"] for e in good),
        radius_reduction_m=numbers("radius_reduction_m"),
        geometry_runtime_s=numbers("geometry_runtime_s"), audit_runtime_s=numbers("audit_runtime_s"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-geometry", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing diagnostic directory")
    if any(n is not None and n <= 0 for n in (args.max_records, args.max_geometry)):
        raise ValueError("Diagnostic limits must be positive")
    identities = {p: sha(ROOT/p) for p in IDENTITY_FILES}
    inputs = []
    for directory in args.input:
        if directory.name not in ALLOWED_SEEDS:
            raise ValueError("Only the two already opened combination-development sets are permitted")
        manifest = json.loads((directory/"manifest.json").read_bytes())
        if manifest["seeds"] != ALLOWED_SEEDS[directory.name] or manifest["specs"][LABEL] != EXPECTED_SPEC:
            raise ValueError("Unexpected old dataset seed/spec identity")
        if json.loads((directory/"independent_audit.json").read_bytes()).get("all_passed") is not True:
            raise ValueError("Old observation evidence needs its completed original audit")
        for p in ("src/geometry/__init__.py", "src/localization/__init__.py"):
            if manifest["source_sha256"][p] != identities[p]:
                raise ValueError("Region reconstruction differs from original geometry source")
        inputs.append(dict(directory=str(directory.resolve()), name=directory.name, seeds=manifest["seeds"],
            evidence_sha256={p: sha(directory/p) for p in ("manifest.json", "freeze.json", "source.zip", "independent_audit.json")}))
    freeze = dict(kind="observation_prefix_geometry_only", source_sha256=identities, inputs=inputs,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        python=sys.version, platform=platform.platform(), max_records=args.max_records, max_geometry=args.max_geometry,
        decoded_record_fields=["summary", "history", "spec"],
        forbidden_fields="All other record payloads skipped lexically, including evaluation/row; no source truth, fresh cases or counterfactual runtime")
    write(args.output/"freeze.json", freeze)
    events, records, failures = [], [], []
    called = 0
    for dataset in inputs:
        for seed in dataset["seeds"]:
            if args.max_records is not None and len(records) >= args.max_records:
                break
            path = Path(dataset["directory"])/"records"/f"{LABEL}-{seed}.json.gz"
            record_info = dict(dataset=dataset["name"], seed=seed, input_path=str(path), input_sha256=sha(path))
            records.append(record_info)
            try:
                record = observation_record(path)
                if record["spec"] != EXPECTED_SPEC:
                    raise ValueError("Record strategy differs from old frozen spec")
                for entry in resolver_inputs(record):
                    row = dict(dataset=dataset["name"], seed=seed, channel=entry["channel"],
                        after_actual_action_count=entry["after_actual_action_count"],
                        resolver_origins=entry["resolver_origins"], skip_reason=entry["skip_reason"])
                    events.append(row)
                    if entry["skip_reason"]:
                        continue
                    if args.max_geometry is not None and called >= args.max_geometry:
                        row["skip_reason"] = "pilot_geometry_limit"
                        continue
                    called += 1
                    name = f'{dataset["name"]}-{seed}-c{entry["channel"]}-n{entry["after_actual_action_count"]}.json.gz'
                    artifact = dict(record_identity=record_info, observation_input=entry,
                                    observation_input_sha256=digest(entry))
                    t = time.perf_counter()
                    outer, evidence = joint_visibility_outer(entry["canonical_vertices"],
                        entry["positive_positions"], entry["negative_positions"])
                    row["geometry_runtime_s"] = time.perf_counter()-t
                    artifact["geometry"] = evidence
                    t = time.perf_counter()
                    try:
                        audit = audit_joint_visibility_certificate(evidence)
                        row["audit_passed"] = audit.get("passed") is True
                        artifact["audit"] = audit
                    except Exception as error:
                        row["audit_passed"] = False
                        artifact["audit"] = dict(passed=False, error_type=type(error).__name__, error=str(error))
                    row["audit_runtime_s"] = time.perf_counter()-t
                    row.update(geometry_status=evidence["status"], fallback_reason=evidence.get("fallback_reason"),
                        deleted_cells=evidence.get("deleted_intersecting_cells", 0),
                        old_vertices_excluded=len(evidence["old_vertices_excluded"]),
                        old_radius_m=evidence["old_disk"]["radius_m"], new_radius_m=evidence["new_disk"]["radius_m"],
                        radius_reduction_m=evidence["old_disk"]["radius_m"]-evidence["new_disk"]["radius_m"],
                        became_ready=evidence["became_ready"])
                    artifact["metrics"] = row.copy()
                    output = args.output/"prefixes"/name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(output, "xt", encoding="utf-8") as stream:
                        json.dump(artifact, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    row.update(artifact=str(output.relative_to(args.output)), artifact_sha256=sha(output))
            except Exception as error:
                failures.append(dict(**record_info, error_type=type(error).__name__, error=str(error)))
    if identities != {p: sha(ROOT/p) for p in IDENTITY_FILES}:
        failures.append(dict(error_type="SourceChanged", error="Diagnostic source changed during execution"))
    result = dict(scope="Geometry at old resolver entries only; not counterfactual time or a performance validation set",
        complete_requested_records=len(records) == sum(len(d["seeds"]) for d in inputs),
        all_audits_passed=not failures and not any(e.get("audit_passed") is False for e in events),
        records=records, failures=failures, summaries={name: metrics([e for e in events if e["dataset"] == name]) for name in ALLOWED_SEEDS},
        overall=metrics(events), events=events, freeze_sha256=sha(args.output/"freeze.json"))
    write(args.output/"summary.json", result)
    print(json.dumps({k: result[k] for k in ("complete_requested_records", "all_audits_passed", "overall", "failures")}))
    return 0 if result["all_audits_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

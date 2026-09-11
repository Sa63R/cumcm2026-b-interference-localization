"""Fetch verified, selected S3 snapshots; never treats a stored PID as live.

Credentials stay in the external JSON file. Only the fixed cost-choice-v2 task
prefix is read, regardless of a legacy prefix in that credential file.
Default downloads are summaries/comparisons/environment. Models, cases, and
training evidence are opt-in; training logs cannot be selected. Select optional
artifacts individually by their exact manifest-relative path.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "lianghao/bwc/shumo/q3-cost2-20260912-r1"
HASH = re.compile(r"[0-9a-f]{64}")
LIMITS = {"pointer": 65536, "manifest": 16 << 20, "metadata": 16 << 20,
          "case": 64 << 20, "evidence": 64 << 20, "model": 2 << 30}


def validate_prefix(prefix):
    if prefix != PREFIX:
        raise ValueError("Only the dedicated autonomy prefix is permitted")
    return prefix


def relative_path(name):
    if not isinstance(name, str) or not 1 <= len(name) <= 1024:
        raise ValueError("Invalid snapshot relative path")
    parts = name.split("/")
    for part in parts:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", part)
                or part.endswith(".") or part in {".", ".."}
                or re.fullmatch(r"(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", part.split(".")[0])):
            raise ValueError("Unsafe snapshot relative path")
    return parts


def parsed_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON member")
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def read_object(client, bucket, key, limit, destination=None, expected=None):
    if (not isinstance(key, str) or not (key == "live/LATEST.json"
            or re.fullmatch(r"live/manifests/[A-Za-z0-9_.-]+\.json", key)
            or re.fullmatch(r"live/objects/[0-9a-f]{64}", key))):
        raise ValueError("Object key is outside the permitted snapshot namespaces")
    response = client.get_object(Bucket=bucket, Key=PREFIX + "/" + key)
    body = response["Body"]
    chunks, count, digest = [], 0, hashlib.sha256()
    target = None
    try:
        if destination is not None:
            target = destination.open("xb")
        declared = response.get("ContentLength")
        if declared is not None and (type(declared) is not int or not 0 <= declared <= limit):
            raise ValueError("Object size exceeds the requested download limit")
        for chunk in iter(lambda: body.read(1 << 20), b""):
            count += len(chunk)
            if count > limit:
                raise ValueError("Object stream exceeds the requested download limit")
            digest.update(chunk)
            if target is None:
                chunks.append(chunk)
            else:
                target.write(chunk)
        if declared is not None and count != declared:
            raise ValueError("Object content length differs")
        actual = {"sha256": digest.hexdigest(), "bytes": count}
        if expected is not None and actual != expected:
            raise ValueError("Snapshot object checksum or byte count differs")
        return b"".join(chunks) if target is None else actual
    finally:
        if target is not None:
            target.close()
        body.close()


def metadata_path(name):
    parts = relative_path(name)
    return ((len(parts) == 2 and parts[1] in {"summary.json", "comparisons.json", "environment.json"})
            or (len(parts) == 4 and parts[1] == "evaluation"
                and parts[3] in {"summary.json", "comparisons.json"}))


def optional_path(name, kind):
    parts = relative_path(name)
    if kind == "model":
        return ((len(parts) == 3 and re.fullmatch(r"trial-[123]", parts[1]) and parts[2].endswith(".pt"))
                or (len(parts) == 4 and parts[1] == "evaluation" and parts[3].endswith(".pt")))
    if kind == "case":
        return (len(parts) == 4 and parts[1] == "evaluation"
                and bool(re.fullmatch(r"case-\d+\.json\.gz", parts[3])))
    if kind == "evidence":
        return (len(parts) == 4 and parts[1:3] == ["trial-1", "evidence"]
                and bool(re.fullmatch(r"group-[0-9]+\.json\.gz", parts[3])))
    return False


def _numeric_fields(value, names):
    result = {}
    for name in names:
        item = value.get(name)
        if item is None or type(item) in (bool, int) or (type(item) is float and math.isfinite(item)):
            if name in value:
                result[name] = item
    return result


def anonymous_summary(documents):
    """Only public numeric metrics and controlled status labels reach stdout."""
    task_names = sorted({relative_path(name)[0] for name in documents})
    tasks = []
    summary_fields = ("runs", "expected_runs", "complete", "successful_runs", "failed_clear_count",
                      "raw_mean_total_time_s", "penalized_mean_total_time_s", "raw_p95_total_time_s",
                      "raw_max_total_time_s", "physical_mean_lower_s", "ratio_of_sums", "audit_passed_records")
    compare_fields = ("pairs", "safe_full_clear_pairs", "mean_seconds_saved", "mean_reduction_fraction",
                      "p95_time_ratio", "max_time_ratio", "wins", "losses", "maximum_paired_regression_s",
                      "reference_time_over_physical_lower", "candidate_time_over_physical_lower",
                      "strict_upgrade_on_supplied_cases")
    for index, name in enumerate(task_names, 1):
        state = documents.get(name + "/summary.json", {})
        status = state.get("status")
        task = {"task": f"task-{index:03d}", "reported_status": status if status in {
            "running", "complete", "interrupted", "error", "pending"} else "unknown",
            "training": _numeric_fields(state, ("training_complete", "update", "episodes", "attempted_episodes",
                                                 "optimizer_steps", "elapsed_training_s", "next_seed"))}
        blocks = state.get("completed_blocks", [])
        task["completed_blocks"] = [n for n in blocks if type(n) is int and 0 <= n <= 1000] if isinstance(blocks, list) else []
        task["block_phases"] = {}
        if isinstance(state.get("blocks"), dict):
            for key, value in state["blocks"].items():
                if re.fullmatch(r"\d{1,4}", key) and isinstance(value, dict) and value.get("phase") in {
                        "training", "model_ready", "evaluating", "complete"}:
                    task["block_phases"][key] = value["phase"]
        environment = documents.get(name + "/environment.json", {})
        task["environment"] = _numeric_fields(environment, ("cpu_slots", "requested_slots", "visible_quota_slots", "nice"))
        evaluations = {}
        labels = sorted({relative_path(path)[2] for path in documents
                         if len(relative_path(path)) == 4 and relative_path(path)[:2] == [name, "evaluation"]})
        label_map = {label: label if re.fullmatch(r"(?:block-\d+|initial|state_search|development|confirmation|final|stress)", label)
                     else f"evaluation-{number:03d}" for number, label in enumerate(labels, 1)}
        for path, value in sorted(documents.items()):
            parts = relative_path(path)
            if len(parts) != 4 or parts[0] != name or parts[1] != "evaluation":
                continue
            label = label_map[parts[2]]
            entry = evaluations.setdefault(label, {})
            if parts[-1] == "summary.json":
                entry.update(_numeric_fields(value, summary_fields))
            elif parts[-1] == "comparisons.json":
                entry["comparisons"] = {}
                for baseline in ("state_search", "initial"):
                    comparison = value.get(baseline)
                    if not isinstance(comparison, dict):
                        continue
                    public = _numeric_fields(comparison, compare_fields)
                    interval = comparison.get("paired_mean_savings_ci95_s")
                    if (isinstance(interval, list) and len(interval) == 2
                            and all(type(x) in (int, float) and math.isfinite(x) for x in interval)):
                        public["paired_mean_savings_ci95_s"] = interval
                    entry["comparisons"][baseline] = public
        task["evaluations"] = evaluations
        tasks.append(task)
    return tasks


def fetch_status(client, bucket, *, prefix=PREFIX, output=None, models=(), cases=(), evidence=()):
    validate_prefix(prefix)
    models, cases, evidence = tuple(models), tuple(cases), tuple(evidence)
    if output is not None and Path(output).exists():
        raise ValueError("Use a new output directory")
    pointer_bytes = read_object(client, bucket, "live/LATEST.json", LIMITS["pointer"])
    pointer = parsed_json(pointer_bytes)
    key, expected_hash, stamp = (pointer.get(name) for name in ("manifest_key", "sha256", "captured_utc"))
    if not isinstance(key, str) or not re.fullmatch(r"live/manifests/[A-Za-z0-9_.-]+\.json", key):
        raise ValueError("Pointer manifest key is outside the permitted manifest directory")
    if not isinstance(expected_hash, str) or not HASH.fullmatch(expected_hash):
        raise ValueError("Pointer has no valid manifest SHA256")
    if not isinstance(stamp, str) or not re.fullmatch(r"\d{8}T\d{6}(?:\.\d{1,6})?Z", stamp):
        raise ValueError("Pointer has no valid snapshot timestamp")
    datetime.strptime(stamp, "%Y%m%dT%H%M%S.%fZ" if "." in stamp else "%Y%m%dT%H%M%SZ")
    manifest_bytes = read_object(client, bucket, key, LIMITS["manifest"])
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
        raise ValueError("Manifest SHA256 differs from the pointer")
    manifest = parsed_json(manifest_bytes)
    session = manifest.get("session")
    if (type(manifest.get("schema")) is not int or manifest["schema"] != 1 or manifest.get("captured_utc") != stamp
            or not isinstance(session, str) or not re.fullmatch(r"[a-f0-9]{12}", session)
            or key != f"live/manifests/{stamp}-{session}.json" or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Manifest identity differs from the pointer")
    entries = manifest["files"]
    for name, meta in entries.items():
        relative_path(name)
        if (not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str)
                or not HASH.fullmatch(meta["sha256"]) or type(meta.get("bytes")) is not int
                or not 0 <= meta["bytes"] <= 2**63 - 1
                or meta.get("object_key") != "live/objects/" + meta["sha256"]):
            raise ValueError("Manifest has an invalid content object identity")
    selected = {name: "metadata" for name in entries if metadata_path(name)}
    for kind, names in (("model", models), ("case", cases), ("evidence", evidence)):
        for name in names:
            if not optional_path(name, kind) or name not in entries:
                raise ValueError("Requested optional artifact is absent or has the wrong type")
            selected[name] = kind
    if output is None:
        base = ROOT / "handoff"
        base.mkdir(parents=True, exist_ok=True)
        if not base.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Default output directory escaped the project")
        output = base / f"autonomy-status-{stamp}-{expected_hash[:12]}"
        suffix = 1
        while output.exists():
            output = base / f"autonomy-status-{stamp}-{expected_hash[:12]}-{suffix:03d}"
            suffix += 1
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "pointer.json").write_bytes(pointer_bytes)
    (output / "manifest.json").write_bytes(manifest_bytes)
    documents, verified = {}, {}
    for name, kind in sorted(selected.items()):
        meta = entries[name]
        if meta["bytes"] > LIMITS[kind]:
            raise ValueError("Selected artifact exceeds its type-specific limit")
        destination = output.joinpath(*relative_path(name))
        if not destination.resolve().is_relative_to(output):
            raise ValueError("Download path escaped the output directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        try:
            verified[name] = read_object(client, bucket, meta["object_key"], LIMITS[kind], temporary,
                                         {key: meta[key] for key in ("sha256", "bytes")})
            if kind == "metadata":
                documents[name] = parsed_json(temporary.read_bytes())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    result = dict(ok=True, captured_utc=stamp,
                  verified_pointer={"manifest_key": key, "manifest_sha256": expected_hash,
                                    "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest()},
                  verified_objects=len(verified), downloaded_bytes=sum(v["bytes"] for v in verified.values()),
                  optional_models=len(set(models)), optional_cases=len(set(cases)),
                  optional_evidence=len(set(evidence)),
                  reported_final_sync=manifest.get("final_sync") is True,
                  process_liveness="not_observed",
                  scope="Verified per-file stored snapshots; running/PID fields are not live-process evidence. File capture times can differ.",
                  tasks=anonymous_summary(documents))
    (output / "fetch_index.json").write_text(json.dumps({"files": verified}, indent=2) + "\n", encoding="utf-8")
    (output / "verified_status.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def client_from_config(path):
    conf = parsed_json(Path(path).read_bytes())
    endpoint = urlsplit(conf.get("endpoint", ""))
    if endpoint.scheme != "https" or not endpoint.netloc or endpoint.username or endpoint.password:
        raise ValueError("External storage configuration requires an HTTPS endpoint")
    for name in ("bucket", "access_key_id", "secret_access_key"):
        if not isinstance(conf.get(name), str) or not conf[name]:
            raise ValueError("External storage configuration is incomplete")
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", endpoint_url=conf["endpoint"],
        aws_access_key_id=conf["access_key_id"], aws_secret_access_key=conf["secret_access_key"],
        region_name=conf.get("region", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      request_checksum_calculation="when_required", response_checksum_validation="when_required",
                      connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}))
    return client, conf["bucket"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="External q3-object-storage.json; never copied or printed")
    parser.add_argument("--output", type=Path, help="New local directory; default handoff/autonomy-status-<manifest timestamp>-<hash>")
    parser.add_argument("--prefix", default=PREFIX, choices=(PREFIX,))
    parser.add_argument("--model", action="append", default=[], help="Exact manifest-relative .pt path; repeat to select")
    parser.add_argument("--case", action="append", default=[], help="Exact manifest-relative case-N.json.gz path; repeat to select")
    parser.add_argument("--evidence", action="append", default=[],
                        help="Exact manifest-relative TASK/trial-1/evidence/group-N.json.gz path; no glob; repeat to select")
    args = parser.parse_args(argv)
    try:
        client, bucket = client_from_config(args.config)
        result = fetch_status(client, bucket, prefix=args.prefix, output=args.output,
                              models=args.model, cases=args.case, evidence=args.evidence)
    except Exception as exc:
        # Do not print provider errors, endpoint, bucket, local credential path,
        # credentials, raw metadata, or a traceback with private filesystem names.
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

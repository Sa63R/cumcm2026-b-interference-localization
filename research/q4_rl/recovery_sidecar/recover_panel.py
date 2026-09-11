"""Opt-in recovery beside the frozen evaluator; never modify its source tree.

prepare is read-only on the old output, validates every record, and freezes the
exact retained/missing matrix in a NEW output. execute requires that freeze's
byte SHA and an outer CPU supervisor. No network or database access exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
import zlib

VERSION = "q4-frozen-panel-recovery-v2-artifact-cap"
STOP_RESERVE_BYTES = 4096
OUTER_LOG_RESERVE_BYTES = 8 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda _: invalid("Nonfinite JSON"))


def invalid(message):
    raise ValueError(message)


def require(condition, message):
    if not condition:
        invalid(message)


def inside(root, value):
    path = (root / value).resolve()
    require(path != root and path.is_relative_to(root), "Path must stay inside task root")
    return path


def put(path, value):
    # Exclusive creation: the sidecar never overwrites an old evidence file.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class ArtifactLimit(RuntimeError):
    pass


class ArtifactBudget:
    """Logical bytes of unique NEW file inodes, including partials and reports.

    Old retained/admin hard links are excluded explicitly. Filesystem allocation,
    directories and supervisor logs outside this output require the disk guard.
    A small reserve belongs to the cap and lets an administrative stop be saved.
    """
    def __init__(self, root, maximum, excluded=(), *, outer_reserve=OUTER_LOG_RESERVE_BYTES):
        require(type(maximum) is int and maximum > STOP_RESERVE_BYTES + outer_reserve, "Artifact cap must exceed metadata/log reserves")
        self.root, self.maximum = root.resolve(), maximum
        self.outer_reserve = outer_reserve
        self.excluded = {Path(p).resolve() for p in excluded}
        self.used = self.scan()

    def scan(self):
        seen, total = set(), 0
        for path in self.root.rglob("*"):
            if not path.is_file() or path.resolve() in self.excluded:
                continue
            require(not path.is_symlink(), "Unexpected artifact symlink")
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                total += info.st_size
                seen.add(identity)
        return total

    def check(self, additional, *, reserve=True):
        limit = self.maximum - self.outer_reserve - (STOP_RESERVE_BYTES if reserve else 0)
        if self.used + additional > limit:
            raise ArtifactLimit("max_new_artifact_bytes")
        if reserve and self.outer_reserve and self.outer_bytes() > self.outer_reserve:
            raise ArtifactLimit("outer_log_reserve_exceeded")

    def outer_bytes(self):
        total = 0
        # Prune evaluation before traversal; never walk thousands of raw files
        # at every write. Only this new parent run's outside-output artifacts.
        for directory, dirs, names in os.walk(self.root.parent):
            dirs[:] = [d for d in dirs if (Path(directory) / d).resolve() != self.root]
            for name in names:
                path = Path(directory) / name
                if path.is_file():
                    total += path.stat().st_size
        return total

    @contextmanager
    def writer(self, path, *, append=False, reserve=True):
        path = path.resolve()
        require(path.is_relative_to(self.root) and path not in self.excluded, "Cannot write excluded/foreign artifact")
        self.check(0, reserve=reserve)
        budget = self
        with path.open("ab" if append else "xb") as raw:
            before = path.stat().st_size
            class GuardedWriter:
                def write(self, data):
                    budget.check(len(data), reserve=reserve)
                    old = raw.tell()
                    try:
                        return raw.write(data)
                    finally:
                        budget.used += raw.tell() - old
                        budget.check(0, reserve=reserve)
                def flush(self):
                    return raw.flush()
                def fileno(self):
                    return raw.fileno()
                def tell(self):
                    return raw.tell()
            stream = GuardedWriter()
            initial_used = self.used
            try:
                yield stream
            finally:
                raw.flush()
                require(path.stat().st_size - before == self.used - initial_used, "Artifact write accounting mismatch")
                self.check(0, reserve=reserve)

    def put(self, path, value, *, reserve=True):
        payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        with self.writer(path, reserve=reserve) as stream:
            stream.write(payload)

    def copy(self, source, target):
        with Path(source).open("rb") as old, self.writer(target) as new:
            for block in iter(lambda: old.read(1024 * 1024), b""):
                new.write(block)

    def audit(self):
        require(self.scan() == self.used, "Unaccounted new artifact bytes")
        self.check(0)

    def stop(self, reason="max_new_artifact_bytes"):
        # This metadata uses the 4096-byte reserve, still inside the exact cap.
        self.put(self.root / "administrative_stop.json", {
            "stop_reason": reason, "administrative_interruption": True,
            "max_new_artifact_bytes": self.maximum, "bytes_before_stop_record": self.used,
            "outer_log_reserve_bytes": self.outer_reserve, "observed_parent_run_bytes": self.outer_bytes(),
            "partial_and_inflight_preserved": True, **publication_state(self.root),
            "new_output_required_after_review": True}, reserve=False)
        self.check(0, reserve=False)


def publication_state(output):
    published = [name for name in ("summary.json", "summary_vs_r8.json", "summary_vs_prior_rl.json")
        if (output / name).exists()]
    return {"performance_report_published": bool(published), "published_report_files": published,
        "panel_complete_marker_exists": (output / "panel_complete.json").exists(),
        "resource_exit_75_qualifies_for_promotion": False}


def bind_file(path, expected):
    require(sha(path) == expected, "Frozen file SHA mismatch: " + path.name)


def load_evaluator(root):
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "src"), str(root)]
    module = importlib.import_module("experiments.q4_rl_evaluate")
    require(module.ROOT.resolve() == root, "Evaluator imported from another tree")
    module._set_cpu_environment()
    return module


def verify_zip(path, sources):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(sources),
            "Archived source set differs from freeze")
        for name, expected in sources.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected,
                "Archived source SHA differs: " + name)


def validate_contract(root, original, control, panel_name, evaluator, expected_manifest, expected_plan):
    bind_file(original / "manifest.json", expected_manifest)
    bind_file(control / "plan.json", expected_plan)
    manifest, frozen, plan = read(original / "manifest.json"), read(original / "freeze.json"), read(control / "plan.json")
    require(frozen == {"manifest_sha256": canonical(manifest)}, "Manifest/freeze mismatch")
    panels = [p for p in plan["panels"] if p["run"] == panel_name]
    require(len(panels) == 1, "Panel not uniquely preregistered")
    panel = panels[0]
    specs = evaluator.validate_specs(read(control / "specs.json"))
    bind_file(control / "specs.json", plan["control_sha256"]["specs.json"])
    require(read(original / "specs.original.json") == specs == manifest["specs"], "Specs content mismatch")
    require(sha(original / "specs.original.json") == manifest["raw_specs_sha256"] == sha(control / "specs.json"),
        "Original specs bytes mismatch")
    requests = evaluator.case_requests("development", start=panel["start"], count=panel["count"],
        families=panel["families"], source_modes=["mixed", "all_directional"])
    require(manifest["requests"] == requests and len(requests) == panel["scenarios"], "Requested matrix changed")
    require(list(specs) == plan["arms"], "Arm identity/order changed")
    require(manifest["source_sha256"] == evaluator.source_hashes(), "Current source differs from freeze")
    require(manifest["artifact_sha256"] == evaluator.artifact_hashes(specs), "Current model/config differs")
    require(manifest["artifact_sha256"] == {m["path"]: m["sha256"] for m in plan["models"]}, "Plan model SHA mismatch")
    require(manifest["protocol_sha256"] == sha(evaluator.PROTOCOL_PATH), "Protocol SHA mismatch")
    for key, value in dict(stage="development", reference="r9_probe", bootstrap_samples=5000,
            bootstrap_seed=4260911, workers=10, numerical_threads_per_worker=1,
            selection_sha256=None, formal_simulator=False, practice_simulator=False).items():
        require(manifest.get(key) == value, "Frozen evaluation convention differs: " + key)
    require(plan["prior_rl_reference"] == "macro_v2_ppo512" and plan["bootstrap_samples"] == 5000,
        "Prior reference/bootstrap changed")
    verify_zip(original / "source.zip", manifest["source_sha256"])
    return manifest, plan, panel


def expected_matrix(manifest, evaluator):
    matrix = {}
    for request_index, request in enumerate(manifest["requests"]):
        # Pure scene construction verifies identity only; no simulator or action.
        case = evaluator.build_case(**request)
        case_hash = canonical(case.evaluation_config())
        for label in manifest["specs"]:
            name = case.case_id + "--" + label + ".json.gz"
            require(name not in matrix and Path(name).name == name, "Duplicate/unsafe matrix identity")
            matrix[name] = dict(request_index=request_index, label=label,
                case_id=case.case_id, case_sha256=case_hash)
    return matrix


def close(a, b):
    return (isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a)
            and isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6))


def validate_record(record, item, manifest):
    request = manifest["requests"][item["request_index"]]
    label = item["label"]
    require(record["request"] == request and record["spec"] == manifest["specs"][label], "Record request/spec mismatch")
    require(record["evaluation_phase"] == "after_policy_termination", "Record not a final evaluator record")
    row, evaluation, bound = record["row"], record["evaluation"], record["common_lower_bound"]
    identity = dict(case_id=item["case_id"], case_sha256=item["case_sha256"], strategy=label,
        problem=4, seed=request["seed"], stage=request["split"], family=request["family"], source_mode=request["source_mode"])
    require(all(row.get(k) == v for k, v in identity.items()), "Record row identity mismatch")
    require(evaluation["case_id"] == item["case_id"] and canonical(evaluation["ground_truth"]) == item["case_sha256"],
        "Record environment identity mismatch")
    for key in ("virtual_time_s", "source_total", "cleared_total", "measurement_count", "failed_clear_count", "action_count"):
        require(close(row[key], evaluation[key]), "Row/evaluation mismatch: " + key)
    for key, value in evaluation["time_breakdown_s"].items():
        require(close(row[key], value), "Time component mismatch")
    require(row["all_cleared"] == evaluation["all_cleared"], "All-clear flag mismatch")
    require(row["audit_passed"] == record["audit"]["passed"], "Stored audit flag mismatch")
    require(isinstance(record["history"], list) and isinstance(row["successful"], bool), "Invalid record/history schema")
    if row["successful"]:
        require(all(row[k] is True for k in ("all_cleared", "completion_certified", "accepted_exit", "audit_passed"))
            and not row["errors"], "Invalid claimed successful result")
    require(close(row["common_lower_bound_s"], bound["common_lower_bound_s"])
        and close(row["common_lower_bound_rounded_s"], bound["common_lower_bound_rounded_s"])
        and row["common_lower_bound_s"] > 0, "Lower-bound record mismatch")
    penalized = row["virtual_time_s"] if row["successful"] else 360000.
    require(close(row["penalized_time_s"], penalized), "Changed failure penalty")
    require(close(row["penalized_time_over_lower_bound"], penalized / row["common_lower_bound_s"]), "Changed penalized ratio")
    require((close(row["time_over_lower_bound"], row["virtual_time_s"] / row["common_lower_bound_s"])
        if row["successful"] else row["time_over_lower_bound"] is None), "Changed success ratio")
    return row


def read_record(path):
    # Read to gzip EOF: validates compressed CRC/footer, not just a JSON prefix.
    payload = gzip.decompress(path.read_bytes())
    return json.loads(payload, parse_constant=lambda _: invalid("Nonfinite record JSON"))


def inventory(directory, matrix, manifest):
    retained, damaged, rows = {}, {}, {}
    require(directory.is_dir(), "Original records directory missing")
    for path in sorted(directory.iterdir()):
        require(path.is_file() and not path.is_symlink() and path.name in matrix, "Unexpected record file: " + path.name)
        fingerprint = sha(path)
        try:
            record = read_record(path)
        except (gzip.BadGzipFile, EOFError, zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Never silently erase a partial file or treat it as a policy failure.
            damaged[path.name] = dict(sha256=fingerprint, bytes=path.stat().st_size,
                classification="administratively_unusable_record", error_type=type(exc).__name__)
            continue
        # Decodable but wrong identity/semantics is NOT an administrative retry.
        rows[path.name] = validate_record(record, matrix[path.name], manifest)
        retained[path.name] = fingerprint
    missing = [name for name in matrix if name not in retained]
    return retained, damaged, rows, missing


def link(source, target):
    require(not target.exists(), "Refusing to overwrite recovery evidence")
    os.link(source, target)  # Fail explicitly across volumes; no surprise bulk copy.


def prepare(root, original, output, control, panel_name, status_path, evaluator,
            manifest_sha, plan_sha, status_sha, max_new_artifact_bytes):
    require(type(max_new_artifact_bytes) is int and max_new_artifact_bytes > STOP_RESERVE_BYTES + OUTER_LOG_RESERVE_BYTES, "Invalid artifact cap")
    require(not output.exists(), "Recovery output must be new")
    manifest, plan, panel = validate_contract(root, original, control, panel_name, evaluator, manifest_sha, plan_sha)
    bind_file(status_path, status_sha)
    status = read(status_path)
    require(status.get("run") == panel_name and status.get("final_sync_ok") is True
        and status.get("termination_requested") is True and status.get("child_returncode") is not None
        and status.get("stop_reason") == "disk_free_below_minimum", "Missing finalized disk-stop/sync evidence")
    matrix = expected_matrix(manifest, evaluator)
    retained, damaged, rows, missing = inventory(original / "records", matrix, manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    (output / "administrative_records").mkdir()
    budget = ArtifactBudget(output, max_new_artifact_bytes,
        [output / "records" / n for n in retained] + [output / "administrative_records" / n for n in damaged])
    for name in ("manifest.json", "freeze.json", "source.zip", "specs.original.json"):
        budget.copy(original / name, output / name)
    budget.copy(status_path, output / "prior-supervisor-final.json")
    for name in retained:
        link(original / "records" / name, output / "records" / name)
    for name in damaged:
        link(original / "records" / name, output / "administrative_records" / name)
    recovery = dict(version=VERSION, panel=panel_name, max_new_artifact_bytes=max_new_artifact_bytes,
        outer_log_reserve_bytes=OUTER_LOG_RESERVE_BYTES,
        parent_run_bytes_at_prepare=budget.outer_bytes(),
        artifact_cap_scope="Unique new logical file bytes including copied metadata, new raw, inflight, receipts and reports; old retained/admin hardlinks excluded; filesystem/outer-supervisor overhead requires separate free-space guard",
        original_output=original.relative_to(root).as_posix(),
        control=control.relative_to(root).as_posix(), manifest_sha256=manifest_sha, plan_sha256=plan_sha,
        prior_supervisor_sha256=status_sha, sidecar_sha256=sha(Path(__file__)), expected_matrix=matrix,
        retained_record_sha256=retained, administrative_records=damaged,
        missing_matrix=[dict(record_name=name, **matrix[name]) for name in missing],
        expected_count=len(matrix), retained_count=len(retained), retained_failed_count=sum(not r["successful"] for r in rows.values()),
        classification_scope="Undecodable files preserved byte-for-byte; no inference about policy outcome; no selection on performance",
        validation_scope="gzip CRC/JSON, request/spec/row/environment and denominator identity; stored audits retained, not rerun",
        source_files_sha256={name: sha(output / name) for name in ("manifest.json", "freeze.json", "source.zip", "specs.original.json")})
    budget.put(output / "recovery.json", recovery)
    budget.put(output / "recovery.freeze.json", {"recovery_sha256": sha(output / "recovery.json")})
    budget.audit()
    return recovery


def write_record(output, name, record, budget):
    target = output / "records" / name
    temporary = output / "inflight" / name
    # Incomplete gzip stays in inflight if interrupted; the original is untouched.
    with budget.writer(temporary) as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
        raw.flush()
        os.fsync(raw.fileno())
    link(temporary, target)
    temporary.unlink()  # Only our new fully committed temporary hard link.


def execute(root, output, recovery_sha, evaluator, max_new_artifact_bytes):
    bind_file(output / "recovery.json", recovery_sha)
    saved = read(output / "recovery.json")
    require(type(max_new_artifact_bytes) is int and max_new_artifact_bytes == saved["max_new_artifact_bytes"],
        "Artifact cap differs from frozen prepare value")
    require(saved["outer_log_reserve_bytes"] == OUTER_LOG_RESERVE_BYTES, "Outer log reserve changed")
    excluded = [output / "records" / n for n in saved["retained_record_sha256"]]
    excluded += [output / "administrative_records" / n for n in saved["administrative_records"]]
    budget = ArtifactBudget(output, max_new_artifact_bytes, excluded)
    try:
        budget.check(0)
        return _execute(root, output, recovery_sha, evaluator, budget)
    except ArtifactLimit as exc:
        budget.stop(str(exc))
        print(json.dumps({"administrative_stop": str(exc), "new_artifact_bytes": budget.used,
            "cap": budget.maximum, **publication_state(output)}), flush=True)
        return 75


def _execute(root, output, recovery_sha, evaluator, budget):
    bind_file(output / "recovery.json", recovery_sha)
    require(read(output / "recovery.freeze.json") == {"recovery_sha256": recovery_sha}, "Recovery freeze mismatch")
    saved = read(output / "recovery.json")
    require(saved["version"] == VERSION and saved["sidecar_sha256"] == sha(Path(__file__)), "Sidecar identity changed")
    for name, expected in saved["source_files_sha256"].items():
        bind_file(output / name, expected)
    bind_file(output / "prior-supervisor-final.json", saved["prior_supervisor_sha256"])
    manifest, plan, panel = validate_contract(root, output, inside(root, saved["control"]), saved["panel"],
        evaluator, saved["manifest_sha256"], saved["plan_sha256"])
    matrix = expected_matrix(manifest, evaluator)
    require(matrix == saved["expected_matrix"], "Recovery matrix changed")
    retained, damaged, rows, missing = inventory(output / "records", matrix, manifest)
    require(not damaged and retained == saved["retained_record_sha256"], "Retained record bytes/set changed")
    require([dict(record_name=n, **matrix[n]) for n in missing] == saved["missing_matrix"], "Missing matrix changed")
    admin = output / "administrative_records"
    require({p.name for p in admin.iterdir()} == set(saved["administrative_records"]), "Administrative evidence set changed")
    for name, info in saved["administrative_records"].items():
        bind_file(admin / name, info["sha256"])
    from scripts.q4_training_pair import require_outer_supervisor
    require_outer_supervisor(root)
    # One immutable continuation per prepared output. After another interruption,
    # preserve this wave and prepare an explicitly reviewed further continuation.
    budget.put(output / "execution.started.json", {"recovery_sha256": recovery_sha,
        "missing_count": len(missing), "workers": 10, "time_epoch": time.time(),
        "max_new_artifact_bytes": budget.maximum, "outer_log_reserve_bytes": budget.outer_reserve,
        "parent_run_bytes_at_execute": budget.outer_bytes()})
    (output / "inflight").mkdir(exist_ok=False)
    began_wall, began_cpu = time.perf_counter(), time.process_time()
    with ProcessPoolExecutor(max_workers=10, initializer=evaluator._set_cpu_environment) as pool:
        # Bounded queue: stop leaves at most two batches in flight, not 4075 tasks.
        iterator = iter(missing)
        jobs = {}
        def submit():
            name = next(iterator, None)
            if name is None:
                return False
            item = matrix[name]
            future = pool.submit(evaluator.run_one, manifest["requests"][item["request_index"]], item["label"],
                manifest["specs"][item["label"]], expected_sources=manifest["source_sha256"], expected_artifacts=manifest["artifact_sha256"])
            jobs[future] = name
            return True
        for _ in range(20):
            submit()
        while jobs:
            future = next(as_completed(jobs))
            name = jobs.pop(future)
            record = future.result()  # Infrastructure exception is not a fake failed policy row.
            row = validate_record(record, matrix[name], manifest)
            write_record(output, name, record, budget)
            rows[name] = row
            with budget.writer(output / "completed.jsonl", append=True) as stream:
                stream.write((json.dumps({"record_name": name, "sha256": sha(output / "records" / name)}) + "\n").encode())
            print(json.dumps({"completed": len(rows), "expected": len(matrix), "new_record": name}), flush=True)
            submit()
    require(evaluator.source_hashes() == manifest["source_sha256"] and evaluator.artifact_hashes(manifest["specs"]) == manifest["artifact_sha256"],
        "Source/model changed during continuation; no qualified summary")
    complete, damaged, rows, missing = inventory(output / "records", matrix, manifest)
    require(not missing and not damaged and len(rows) == len(matrix), "Incomplete matrix; no report")
    for name, expected in saved["retained_record_sha256"].items():
        require(complete[name] == expected, "Retained record changed during recovery")
    all_rows = [rows[name] for name in matrix]
    # No final-named performance report is published unless ALL reports/metadata
    # fit. Cap interruption preserves any staged partials only under inflight.
    report_stage = output / "inflight-reports"
    report_stage.mkdir(exist_ok=False)
    summaries = {}
    for name, reference in (("summary.json", "r9_probe"), ("summary_vs_r8.json", "r8"),
            ("summary_vs_prior_rl.json", plan["prior_rl_reference"])):
        budget.put(report_stage / name, evaluator.report_rows(all_rows, reference=reference, samples=5000, bootstrap_seed=4260911))
        summaries[name] = sha(report_stage / name)
    prior = read(output / "prior-supervisor-final.json")
    budget.put(report_stage / "compute.json", {"prior_supervisor_sha256": saved["prior_supervisor_sha256"],
        "prior_segment_elapsed_s": prior.get("elapsed_s"), "recovery_segment_parent_wall_s": time.perf_counter() - began_wall,
        "recovery_segment_parent_cpu_s": time.process_time() - began_cpu,
        "retained_completed_row_process_cpu_s": sum(rows[n]["process_cpu_s"] for n in retained),
        "new_completed_row_process_cpu_s": sum(rows[n]["process_cpu_s"] for n in complete if n not in retained),
        "scope": "Compute spans both runs; row CPU excludes audit/bounds and interrupted work; cgroup CPU includes other tasks and is NOT task CPU. Archive the new final supervisor/resources after final sync; total task CPU remains unavailable without attributable process accounting."})
    budget.put(report_stage / "evidence.json", {"record_sha256": complete, "manifest_sha256": saved["manifest_sha256"],
        "summary_sha256": summaries["summary.json"], "all_summary_sha256": summaries,
        "recovery_sha256": recovery_sha, "administrative_record_sha256": {n: d["sha256"] for n, d in saved["administrative_records"].items()}})
    code = 0 if all(row["successful"] for row in all_rows) else 1
    budget.put(report_stage / "panel_complete.json", {"run": saved["panel"], "complete_rows": len(rows),
        "retained_rows": len(retained), "new_rows": len(rows) - len(retained), "failed_rows_retained": sum(not r["successful"] for r in all_rows),
        "reference": "r9_probe", "historical_reference": "r8", "prior_rl_reference": plan["prior_rl_reference"],
        "bootstrap_samples": 5000, "bootstrap_seed": 4260911, "evaluator_returncode": code,
        "max_new_artifact_bytes": budget.maximum,
        "outer_log_reserve_bytes": budget.outer_reserve,
        "observed_parent_run_bytes_before_publication": budget.outer_bytes(),
        "final_supervisor_sync_and_object_readback_still_required": True})
    budget.audit()
    # Publication makes only hard links, no new payload bytes. The marker is last.
    names = list(summaries) + ["compute.json", "evidence.json", "panel_complete.json"]
    require(all(not (output / name).exists() for name in names), "Report publication target exists")
    for name in names:
        link(report_stage / name, output / name)
    budget.audit()
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", required=True, help="New task-root-relative recovery output")
    parser.add_argument("--original")
    parser.add_argument("--control", default="launch/eval-memory-v4")
    parser.add_argument("--panel", default="eval-memory-v4-fixed")
    parser.add_argument("--stopped-status")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--stopped-status-sha256")
    parser.add_argument("--recovery-sha256")
    parser.add_argument("--max-new-artifact-bytes", type=int, required=True,
        help="Explicit immutable cap, e.g. 1000000000; excludes old record hardlinks")
    args = parser.parse_args()
    root = args.root.resolve()
    output = inside(root, args.output)
    evaluator = load_evaluator(root)
    if args.mode == "prepare":
        require(all((args.original, args.stopped_status, args.manifest_sha256, args.plan_sha256, args.stopped_status_sha256)), "Prepare requires all original SHA bindings")
        result = prepare(root, inside(root, args.original), output, inside(root, args.control), args.panel,
            inside(root, args.stopped_status), evaluator, args.manifest_sha256, args.plan_sha256, args.stopped_status_sha256,
            args.max_new_artifact_bytes)
        print(json.dumps({"prepared": True, "retained": result["retained_count"], "missing": len(result["missing_matrix"]),
            "damaged_preserved": len(result["administrative_records"]), "recovery_sha256": sha(output / "recovery.json")}))
        return 0
    require(args.recovery_sha256, "Execution requires the reviewed recovery SHA")
    return execute(root, output, args.recovery_sha256, evaluator, args.max_new_artifact_bytes)


if __name__ == "__main__":
    raise SystemExit(main())

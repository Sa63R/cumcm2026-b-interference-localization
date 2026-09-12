"""Read-only validation of paper data. Never imports project strategy/simulator code."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
checks: list[dict] = []
hash_cache: dict[str, str] = {}


def read(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data.decode("utf-8-sig"))


def sha(path):
    key = str(Path(path))
    if key not in hash_cache:
        hash_cache[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hash_cache[key]


def check(label, passed, **details):
    checks.append({"check": label, "passed": bool(passed), **details})


def equal(label, actual, expected):
    check(label, actual == expected)


def close(label, actual, expected, abs_tol=1e-7):
    ok = math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=abs_tol)
    check(label, ok, actual=actual, expected=expected, absolute_tolerance=abs_tol)


def walk(value, at="$"):
    if isinstance(value, dict):
        yield at, value
        for k, v in value.items():
            yield from walk(v, f"{at}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{at}[{i}]")


def path_hash(label, path, expected=None):
    exists = Path(path).exists()
    check(label + ":exists", exists, path=str(path))
    if expected is not None and exists:
        actual = sha(path)
        check(label + ":sha256", actual == expected.lower(), path=str(path),
              actual_sha256=actual, expected_sha256=expected)


def references(doc, label):
    """Check every absolute local path and every corresponding recorded file hash."""
    found = set()
    for at, obj in walk(doc):
        for k, value in obj.items():
            if not isinstance(value, str) or not re.match(r"^[A-Za-z]:[\\/]", value):
                continue
            digest = None
            if k == "path":
                digest = obj.get("sha256")
            elif k == "source":
                digest = obj.get("source_sha256", obj.get("sha256"))
            elif k.endswith("_path"):
                digest = obj.get(k[:-5] + "_sha256")
            elif k.endswith("_root"):
                digest = None
            identity = (value, digest)
            if identity not in found:
                path_hash(f"{label}:{at}.{k}", value, digest)
                found.add(identity)
    return len(found)


def validate():
    evidence = read(FINAL / "sources/evidence_experiments.json")
    ref_count = references(evidence, "evidence")
    equal("evidence:ids", [x["id"] for x in evidence["entries"]],
          [f"E{i:02d}" for i in range(1, 14)])

    four = read(FINAL / "data/four_methods.json")
    references(four, "four_methods")
    for partition, data in four.items():
        raw = read(data["source"])
        equal(f"four:{partition}:all_method_values", data["methods"], raw["methods"])
        equal(f"four:{partition}:all_comparisons", data["comparisons"], raw["comparisons"])
        equal(f"four:{partition}:protocol", data["protocol_sha256"], raw["protocol_sha256"])
        for method, row in data["methods"].items():
            prefix = f"four:{partition}:{method}"
            equal(prefix + ":sample_count", row["runs"], 256 if partition == "random" else 28)
            equal(prefix + ":all_successful", row["successful_runs"], row["runs"])
            equal(prefix + ":failed_clears", row["failed_clear_count"], 0)
            close(prefix + ":component_sum", sum(row["mean_components_s"].values()),
                  row["raw_mean_total_time_s"])

    with (FINAL / "data/four_methods.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    equal("four_csv:row_count", len(rows), 8)
    equal("four_csv:unique_keys", len({(r["partition"], r["method"]) for r in rows}), 8)
    for row in rows:
        partition, method = row["partition"], row["method"]
        ref = four[partition]["methods"][method]
        prefix = f"csv:{partition}:{method}"
        for csv_key, json_key in [("n", "runs"), ("success", "successful_runs"),
                                  ("mean_s", "raw_mean_total_time_s"),
                                  ("p95_s", "raw_p95_total_time_s")]:
            close(prefix + ":" + csv_key, float(row[csv_key]), ref[json_key])
        for key, value in ref["mean_components_s"].items():
            close(prefix + ":" + key, float(row[key]), value)
        comp = four[partition]["comparisons"].get(method)
        if comp:
            for key, value in [("saving_s", comp["mean_seconds_saved"]),
                               ("ci_low_s", comp["saving_ci95_s"][0]),
                               ("ci_high_s", comp["saving_ci95_s"][1])]:
                close(prefix + ":" + key, float(row[key]), value)
        else:
            equal(prefix + ":baseline_no_ci", [row[k] for k in ("saving_s", "ci_low_s", "ci_high_s")], ["", "", ""])

    labels = ["median", "state_worst", "rl_worst"]
    methods = ["baseline", "state", "rl", "geo"]
    for label in labels:
        data = read(FINAL / f"data/trajectory_{label}.json")
        references(data, f"trajectory:{label}")
        raw = read(data["source"])
        for key in ("seed", "selection", "case_sha256", "plotted"):
            equal(f"trajectory:{label}:source_{key}", data[key], raw[key])
        equal(f"trajectory:{label}:four_archives", len(data["archives"]), 4)
        for method, archive in zip(methods, data["archives"]):
            record = read(archive["source"])
            prefix = f"trajectory:{label}:{method}"
            equal(prefix + ":case_identity", record["row"]["case_sha256"], data["case_sha256"])
            equal(prefix + ":seed", record["row"]["seed"], data["seed"])
            equal(prefix + ":ground_truth", record["evaluation"]["ground_truth"], data["ground_truth"])
            equal(prefix + ":recorded_row", data["plotted"][method]["row"], record["row"])
            check(prefix + ":all_cleared", record["evaluation"]["all_cleared"])

    q2 = read(FINAL / "sources/q2_example.json")
    references(q2, "q2")
    raw = read(q2["archive_path"])
    manifest = read(q2["source_manifest_path"])
    frozen = manifest["inputs"]["state"]["manifest"]["identity"]["source_sha256"]
    for relative, entry in q2["geometry_source_hashes"].items():
        equal("q2:frozen_source:" + relative, entry["sha256"], frozen[relative])
    equal("q2:case_identity", raw["row"]["case_sha256"], q2["case_sha256"])
    equal("q2:seed", raw["row"]["seed"], q2["case_seed"])
    history = raw["summary"]["action_history"]
    observations = q2["observations"]
    equal("q2:two_observations", len(observations), 2)
    for i, item in enumerate(observations):
        idx = item["summary_action_index_zero_based"]
        equal(f"q2:observation_{i}:recorded_action", item["action"], history[idx])
        equal(f"q2:observation_{i}:one_based", item["summary_action_ordinal_one_based"], idx + 1)
        equal(f"q2:observation_{i}:channel", item["action"]["channel"], q2["channel"])
        radius = item["outer_mec_radius_m"]
        max_vertex = max(math.dist(v, item["outer_mec_center_m"]) for v in item["outer_region_vertices_m"])
        close(f"q2:observation_{i}:stored_circle_encloses_vertices", max_vertex, radius, 1e-6)
        equal(f"q2:observation_{i}:clearance_flag", item["satisfies_19_9_m"], radius <= 19.9)
    first, second = observations
    distance = math.dist(first["action"]["position"], second["action"]["position"])
    close("q2:point_distance", q2["distance_between_measurement_points_m"], distance)
    close("q2:movement_time", q2["movement_time_between_points_s"], distance / 5)
    close("q2:radius_reduction", q2["radius_reduction_fraction"],
          1 - second["outer_mec_radius_m"] / first["outer_mec_radius_m"])
    max_distance = max(math.dist(second["action"]["position"], v) for v in first["outer_region_vertices_m"])
    close("q2:max_vertex_distance", q2["max_second_point_distance_to_first_region_vertices_m"], max_distance)
    equal("q2:guaranteed_reception_flag", q2["second_point_has_guaranteed_reception"], max_distance <= 1000)
    equal("q2:angle_error", q2["geometry_parameters"]["bearing_error_deg"], 1.005)
    same_channel = [a for a in history[:second["summary_action_index_zero_based"] + 1]
                    if a["action"] == "measure" and a["channel"] == q2["channel"]]
    equal("q2:prefix_measure_count", len(same_channel), q2["prefix_check"]["same_channel_actual_measurements_through_second"])
    equal("q2:prefix_negative_count", sum(a["result"] == "no_signal" for a in same_channel),
          q2["prefix_check"]["same_channel_prior_negative_observations"])
    check("q2:probe_selection_exists_verbatim_in_archive",
          any(obj == q2["recorded_probe_selection"] for _, obj in walk(raw)))
    return ref_count


if __name__ == "__main__":
    reference_count = None
    try:
        reference_count = validate()
    except Exception as exc:
        check("validator:unexpected_exception", False, exception_type=type(exc).__name__, message=str(exc))
    failed = [c for c in checks if not c["passed"]]
    artifact_paths = [FINAL / "sources/evidence_experiments.json",
                      FINAL / "data/four_methods.json", FINAL / "data/four_methods.csv",
                      *(FINAL / f"data/trajectory_{label}.json"
                        for label in ("median", "state_worst", "rl_worst")),
                      FINAL / "sources/q2_example.json"]
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha(__file__),
        "scope": "Existing file/hash, numeric-copy and arithmetic validation only; no simulation, strategy, remote call or geometric reconstruction.",
        "geometry_limit": "Q2 checks archived actions, source identity and stored-vertex/circle arithmetic; it does not independently reconstruct the feasible polygon or prove circle minimality.",
        "status": "passed" if not failed else "failed",
        "check_count": len(checks),
        "failed_count": len(failed),
        "input_artifacts": [{"path": str(p), "sha256": sha(p)}
                            for p in artifact_paths if p.exists()],
        "evidence_unique_reference_count": reference_count,
        "unique_files_hashed": len(hash_cache),
        "checks": checks,
        "failures": failed,
    }
    output = FINAL / "sources/validation_data.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "check_count", "failed_count", "unique_files_hashed")}, ensure_ascii=False))
    for failure in failed:
        print(json.dumps(failure, ensure_ascii=False))
    raise SystemExit(bool(failed))

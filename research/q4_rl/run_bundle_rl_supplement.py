"""Two frozen TRAIN fixtures: retain three rules and add both prior PPO controls.

CPU-only local synthetic feedback, no training or official simulator. The two
PPO endpoints are both retained; this is not an endpoint selection experiment.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import zipfile

from experiments.q4_rl_evaluate import ROOT, source_hashes, file_digest, run_one, report_rows
from q4_rl.network import configure_cpu


ALIASES = ("legacy_ppo128", "macro_v2_ppo512")
ORIGINAL = "results/q4_rl/bundle-smoke-8006600"
ORIGINS = {
    "legacy_ppo128": ("q4-deep-rl/results/q4_rl/server-pilot-evaluation-001",
                      "runs/train-v1b/training/checkpoint-000060.pt"),
    "macro_v2_ppo512": ("q4-rl-micro-actions/results/q4_rl/server-pair-v2-evaluation-001",
                        "runs/train-pair-v2/macro512/training/checkpoint-000055.pt"),
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/q4_rl/bundle-smoke-rl-supplement-8006600")
    args = parser.parse_args()
    configure_cpu()
    output = (ROOT/args.output).resolve(); output.relative_to(ROOT)
    if output.exists():
        raise ValueError("Fresh output required")
    old = ROOT/ORIGINAL; old_evidence = load(old/"evidence.json")
    for name, digest in old_evidence["files"].items():
        assert file_digest(old/name) == digest, ("original evidence", name)
    old_freeze = load(old/"source_freeze.json")["files"]
    for name, digest in old_freeze.items():
        assert file_digest(ROOT/name) == digest, ("original dependency changed", name)
    with zipfile.ZipFile(old/"source.zip") as archive:
        assert archive.testzip() is None and set(archive.namelist()) == set(old_freeze)
        import hashlib
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name,digest in old_freeze.items())
    requests = load(old/"requests.json")
    assert requests == [dict(seed=8006600, split="train", family="random", source_mode="all_directional"),
                        dict(seed=8006601, split="train", family="minimum_radius", source_mode="all_directional")]
    endpoint_root = ROOT.parent/"q4-rl-micro-attention"
    endpoint_document = endpoint_root/"research/q4_rl/bundle_v3_endpoints.json"
    endpoint_specs = endpoint_root/"research/q4_rl/bundle_v3_evaluation_specs.json"
    assert file_digest(endpoint_specs) == load(endpoint_document)["evaluation_specs_sha256"]
    selected_specs = load(endpoint_specs)
    # bundle_v3_endpoints lists new v3 endpoints only. These two older controls
    # obtain authoritative hashes from their original frozen evaluation manifests.
    provenance = {}
    for alias in ALIASES:
        origin, original_name = ORIGINS[alias]
        folder = ROOT.parent/origin
        manifest = load(folder/"manifest.json"); evidence = load(folder/"evidence.json")
        assert file_digest(folder/"manifest.json") == evidence["manifest_sha256"]
        expected = manifest["artifact_sha256"][original_name]
        model = endpoint_root/"results/q4_rl/bundle-v3-endpoints"/(alias+".pt")
        assert file_digest(model) == expected, alias
        provenance[alias] = dict(sha256=expected, original_checkpoint=original_name,
            original_manifest=origin+"/manifest.json", original_manifest_sha256=file_digest(folder/"manifest.json"),
            original_evidence_sha256=file_digest(folder/"evidence.json"), original_manifest_matches_evidence=True)
    output.mkdir(); (output/"models").mkdir(); (output/"raw").mkdir(); (output/"provenance").mkdir()
    for source, name in ((endpoint_document, "bundle_v3_endpoints.json"),
                         (endpoint_specs, "bundle_v3_evaluation_specs.json"),
                         (old/"evidence.json", "original_rule_evidence.json"),
                         (old/"source_freeze.json", "original_rule_source_freeze.json")):
        shutil.copyfile(source, output/"provenance"/name)
    specs = load(old/"specs.json")
    for alias in ALIASES:
        shutil.copyfile(endpoint_root/"results/q4_rl/bundle-v3-endpoints"/(alias+".pt"), output/"models"/(alias+".pt"))
        origin, _ = ORIGINS[alias]
        shutil.copyfile(ROOT.parent/origin/"manifest.json", output/"provenance"/(alias+"-original-manifest.json"))
        spec = copy.deepcopy(selected_specs[alias])
        target = (output/"models"/(alias+".pt")).relative_to(ROOT).as_posix()
        spec["artifact_files"] = [target]
        spec["policy_factory"]["kwargs"]["checkpoint"] = target
        specs[alias] = spec
    save(output/"specs.json", specs); save(output/"requests.json", requests)
    save(output/"model_provenance.json", dict(models=provenance,
        endpoint_document_note="v3 endpoint document contains the four newly trained endpoints, not these legacy controls; original pilot/v2 frozen manifests supply their verified hashes."))
    dependencies = source_hashes()
    frozen = dict(dependencies)
    for script in ("research/q4_rl/run_bundle_smoke.py", Path(__file__).resolve().relative_to(ROOT).as_posix()):
        frozen[script] = file_digest(ROOT/script)
    save(output/"source_freeze.json", dict(files=frozen, workers=1, numerical_threads=1,
        original_rule_dependencies_unchanged=True, protocol="Same original two TRAIN requests and common bound; all five arms retained"))
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in frozen:
            archive.write(ROOT/name, name)
    rows = []
    for path in sorted(old.glob("*.json.gz")):
        shutil.copyfile(path, output/"raw"/path.name)
        rows.append(json.loads(gzip.decompress(path.read_bytes()))["row"])
    assert len(rows) == 6
    artifacts = {(output/"models"/(alias+".pt")).relative_to(ROOT).as_posix(): provenance[alias]["sha256"] for alias in ALIASES}
    for request in requests:
        for alias in ALIASES:
            assert all(file_digest(ROOT/name) == digest for name,digest in frozen.items())
            record = run_one(request, alias, specs[alias], expected_sources=dependencies, expected_artifacts=artifacts)
            path = output/"raw"/f"{request['seed']}-{alias}.json.gz"
            raw = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            path.write_bytes(gzip.compress(raw, mtime=0)); rows.append(record["row"])
            row = record["row"]
            print(json.dumps({key:row[key] for key in ("seed", "strategy", "successful", "audit_passed", "virtual_time_s", "common_lower_bound_s", "time_over_lower_bound", "failed_clear_count", "process_cpu_s")}), flush=True)
    summary = report_rows(rows, reference="r8", samples=1000)
    summary["scope"] = "Two previously used TRAIN functionality fixtures, two seed clusters and different families; not independent validation, model ranking, generalization evidence or promotion evidence. No case or endpoint dropped."
    summary["record_provenance"] = {"retained_rule_records":6, "new_rl_records":4, "original_directory":ORIGINAL,
        "original_evidence_sha256":file_digest(old/"evidence.json"), "original_rule_dependencies_unchanged":True}
    summary["compute_scope"] = "One CPU thread and sequential runs; wall/CPU measurements include checkpoint loading and may reflect different cache/import states versus retained rules. Virtual billing and bound definitions are identical."
    save(output/"summary.json", summary)
    lines = ["# Scan-bundle functionality supplement with both frozen PPO controls", "",
        summary["scope"], "", "Both prior PPO controls were included before running; the better complete-panel endpoint remains unsettled.", "",
        "| Method | All-clear | Failed clear total | Mean T (s) | P95 T (s) | Sum T / sum L | Mean CPU (s) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ("r8", "legacy_ppo128", "macro_v2_ppo512", "macro_rule", "bundle_rule"):
        value = summary["summaries"][label]
        lines.append(f"| {label} | {value['successful']}/2 | {value['failed_clear_total']} | {value['mean_actual_elapsed_time_s']:.6f} | {value['p95_penalized_time_s']:.6f} | {value['ratio_of_sums']:.6f} | {value['mean_process_cpu_s']:.6f} |")
    lines += ["", "| TRAIN seed / family | L (s) | R8 T/L | legacy PPO128 T/L | macro v2 PPO512 T/L | macro rule T/L | bundle rule T/L |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for request in requests:
        case = {r["strategy"]:r for r in rows if r["seed"] == request["seed"]}
        ratios = " | ".join(f"{case[label]['time_over_lower_bound']:.6f}" if case[label]["time_over_lower_bound"] is not None else "failed" for label in ("r8", *ALIASES, "macro_rule", "bundle_rule"))
        lines.append(f"| {request['seed']} / {request['family']} | {case['r8']['common_lower_bound_s']:.6f} | {ratios} |")
    lines += ["", "Full paired seed-cluster bootstrap intervals, all-clear Wilson intervals, failure records and timing components are in summary.json. Two clusters make these descriptive only; P95 is an interpolation of two observations, not a population-tail estimate.", "",
        summary["compute_scope"], "", "Reproduce the four RL runs from the archived script and manifests; six rule records are reused byte for byte. No checkpoint was trained or selected here.", ""]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    assert all(file_digest(ROOT/name) == digest for name,digest in frozen.items())
    assert all(file_digest(old/name) == digest for name,digest in old_evidence["files"].items())
    with zipfile.ZipFile(output/"source.zip") as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name,digest in frozen.items())
    save(output/"evidence.json", dict(source_archive_crc_and_all_hashes_verified=True,
        original_rules_unchanged=True, source_files=len(frozen), paired_records=len(rows),
        files={path.relative_to(output).as_posix():file_digest(path) for path in sorted(output.rglob("*")) if path.is_file()}))


if __name__ == "__main__":
    main()

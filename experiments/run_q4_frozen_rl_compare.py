"""Paired RL reference using isolated frozen code and the exact state-search cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_round2 import report_rows, write_json
from experiments.run_q4_state_study import digest

REFERENCE = ROOT / "research/q4_joint_visibility/rl_reference"
WORKER = ROOT / "experiments/q4_frozen_rl_worker.py"
RL_ZIP_SHA256 = "86580efae901756eccaeeb48edb6045ca2a9bb8ac9b436ca60cf17fa9179229d"
RL_CHECKPOINT_SHA256 = "3e830b070d5247e3573b3e59b7dfc32d54da101c5aab05cd4effb0dec2b354e8"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_record(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify_archive(archive_path, sources):
    with zipfile.ZipFile(archive_path) as archive:
        for name in list(sources) + archive.namelist():
            path = Path(name)
            if path.anchor or name.startswith(("/", "\\")) or ".." in path.parts or ":" in name or "\\" in name:
                raise ValueError("Unsafe archive member")
        if set(archive.namelist()) != set(sources) or len(archive.namelist()) != len(sources):
            raise ValueError("RL source archive membership mismatch")
        for name, expected in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("RL archive source mismatch: " + name)


def prepare_reference():
    origin = ROOT.parent / "q4-rl-micro-actions/results/q4_rl"
    evaluation = origin / "server-pair-v2-evaluation-001"
    checkpoint = origin / "server-pair-v2-complete-001/macro512/training/checkpoint-000055.pt"
    archive = evaluation / "source.zip"
    if file_hash(archive) != RL_ZIP_SHA256 or file_hash(checkpoint) != RL_CHECKPOINT_SHA256:
        raise ValueError("Upstream frozen reference identity changed")
    manifest = json.loads((evaluation / "manifest.json").read_bytes())
    specs = json.loads((evaluation / "specs.original.json").read_bytes())
    expected_spec = {
        "entrypoint": "q4_rl.controller:run_q4_rl",
        "kwargs": {"max_expansions": 200, "record_transitions": False, "max_decisions": 512},
        "artifact_files": ["runs/train-pair-v2/macro512/training/checkpoint-000055.pt"],
        "policy_factory": {"entrypoint": "q4_rl.network:load_policy", "kwargs": {
            "deterministic": True, "checkpoint": "runs/train-pair-v2/macro512/training/checkpoint-000055.pt"}}}
    if specs["macro_ppo512"] != expected_spec:
        raise ValueError("Upstream learned policy specification changed")
    verify_archive(archive, manifest["source_sha256"])
    reference = dict(role="Fixed strongest evaluated learned Q4 policy at selection time; not claimed optimal",
        source_archive_sha256=RL_ZIP_SHA256, source_sha256=manifest["source_sha256"],
        checkpoint_sha256=RL_CHECKPOINT_SHA256, spec=specs["macro_ppo512"],
        upstream_manifest_sha256=file_hash(evaluation / "manifest.json"),
        upstream_summary_sha256=file_hash(evaluation / "summary.json"),
        upstream_path="q4-rl-micro-actions/results/q4_rl/server-pair-v2-evaluation-001",
        numerical_threads=1, device="cpu", training_enabled=False,
        pairing="Identical explicit environment config and historical LB; no regeneration with RL seed API")
    if REFERENCE.exists():
        if json.loads((REFERENCE / "reference.json").read_bytes()) != reference:
            raise ValueError("Preserve prior RL reference freeze")
    else:
        REFERENCE.mkdir(parents=True)
        shutil.copyfile(archive, REFERENCE / "source.zip")
        shutil.copyfile(checkpoint, REFERENCE / "checkpoint.pt")
        write_json(REFERENCE / "reference.json", reference)
    if (file_hash(REFERENCE / "source.zip") != RL_ZIP_SHA256 or
            file_hash(REFERENCE / "checkpoint.pt") != RL_CHECKPOINT_SHA256):
        raise ValueError("Local frozen artifacts differ")
    return reference


def runtime_root(reference):
    directory = ROOT / "results/tmp/frozen-rl-86580efae901"
    archive_path = REFERENCE / "source.zip"
    verify_archive(archive_path, reference["source_sha256"])
    if not directory.exists():
        directory.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            for name in reference["source_sha256"]:
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    for relative, expected in reference["source_sha256"].items():
        if file_hash(directory / relative) != expected:
            raise ValueError("Extracted frozen source changed")
    return directory


def paired_inputs(state_directory):
    manifest = json.loads((state_directory / "manifest.json").read_bytes())
    freeze = json.loads((state_directory / "freeze.json").read_bytes())
    if freeze["manifest_sha256"] != digest(manifest):
        raise ValueError("State-search manifest differs from freeze")
    records = [read_record(path) for path in sorted((state_directory / "records").glob("compact_baseline-*.json.gz"))]
    if sorted(record["row"]["seed"] for record in records) != sorted(manifest["seeds"]):
        raise ValueError("State reference cases incomplete or duplicated")
    reference_hash = file_hash(REFERENCE / "reference.json")
    items = []
    for record in records:
        if record["evaluation_phase"] != "after_policy_termination":
            raise ValueError("Source evaluation has no completed-policy boundary")
        config, row = record["evaluation"]["ground_truth"], record["row"]
        if digest(config) != row["case_sha256"] or row["strategy"] != "compact_baseline":
            raise ValueError("State case identity mismatch")
        items.append(dict(environment=config, case_sha256=row["case_sha256"],
            common_lower_bound_s=row["common_lower_bound_s"], stage=row["stage"],
            reference_sha256=reference_hash))
    return items, manifest


def run_child(python, source, input_path, output_path, reference_arm):
    command = [str(python), "-B", str(WORKER), "--source-root", str(source),
        "--reference", str(REFERENCE / "reference.json"), "--checkpoint", str(REFERENCE / "checkpoint.pt"),
        "--input", str(input_path), "--output", str(output_path)]
    if reference_arm:
        command.append("--reference-arm")
    with output_path.with_suffix(".log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Frozen RL worker failed; preserve " + str(output_path.with_suffix(".log")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--state-results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reference-arm", action="store_true")
    args = parser.parse_args()
    reference = prepare_reference()
    source = runtime_root(reference)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "source_sha256": RL_ZIP_SHA256,
                          "checkpoint_sha256": RL_CHECKPOINT_SHA256}))
        return
    if not args.state_results or not args.output or not args.python or not 1 <= args.workers <= 4:
        raise ValueError("Need paired state results, new output, Python and 1..4 workers")
    directory = args.output.resolve()
    items, _ = paired_inputs(args.state_results)
    directory.mkdir(parents=True, exist_ok=False)
    freeze = dict(reference_sha256=file_hash(REFERENCE / "reference.json"),
        worker_sha256=file_hash(WORKER), runner_sha256=file_hash(__file__),
        state_manifest_sha256=file_hash(args.state_results / "manifest.json"),
        state_summary_sha256=file_hash(args.state_results / "summary.json"),
        cases=[{k: item[k] for k in ("case_sha256", "common_lower_bound_s", "stage")} for item in items],
        spec=reference["spec"], reference_arm=args.reference_arm,
        role="Frozen explicit same-case RL comparison; no policy training or checkpoint selection here")
    write_json(directory / "freeze.json", freeze)
    shards = [items[i::args.workers] for i in range(args.workers)]
    tasks = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        path = directory / f"environment-input-{i}.json"
        write_json(path, {"worker_sha256": freeze["worker_sha256"], "cases": shard})
        tasks.append((args.python.resolve(), source, path, directory / f"worker-{i}", args.reference_arm))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda task: run_child(*task), tasks))
    records = [read_record(path) for path in sorted(directory.glob("worker-*/*.json.gz"))]
    expected = {item["case_sha256"]: item for item in items}
    if len(records) != len(expected) or {r["row"]["case_sha256"] for r in records} != set(expected):
        raise ValueError("Incomplete or duplicated RL comparison")
    for record in records:
        row = record["row"]
        if (row["common_lower_bound_s"] != expected[row["case_sha256"]]["common_lower_bound_s"] or
                record["policy_reference_sha256"] != freeze["reference_sha256"]):
            raise ValueError("Paired lower bound or reference identity mismatch")
    if file_hash(WORKER) != freeze["worker_sha256"] or file_hash(__file__) != freeze["runner_sha256"]:
        raise ValueError("Outer evaluator changed during comparison")
    state_rows = json.loads((args.state_results / "summary.json").read_bytes())["rows"]
    if file_hash(args.state_results / "summary.json") != freeze["state_summary_sha256"]:
        raise ValueError("State results changed during comparison")
    report = report_rows(sorted(state_rows+[r["row"] for r in records], key=lambda row: (row["seed"], row["strategy"])))
    report["rl_audits_all_passed"] = all(r["audit"]["passed"] for r in records)
    report["freeze_sha256"] = file_hash(directory / "freeze.json")
    write_json(directory / "comparison.json", report)
    print(json.dumps({"summaries": report["summaries"], "rl_audits_all_passed": report["rl_audits_all_passed"]}))


if __name__ == "__main__":
    main()

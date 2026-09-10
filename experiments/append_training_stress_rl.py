"""Append one fixed RL checkpoint to the previously frozen training cases."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from training_stress_reliability import CRITICAL, digest


def append(args):
    original = args.original.resolve()
    output = original / "rl_append"
    output.mkdir(exist_ok=False)
    parent = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    model_bytes = args.checkpoint.read_bytes()
    expected_model = "e602c96224bd6bcfae6e0ad8a217aedf4075809d4901af2a43d5f08df7739c37"
    if hashlib.sha256(model_bytes).hexdigest() != expected_model:
        raise ValueError("Expected approved best002u384 portable checkpoint")
    metadata = json.loads(Path(str(args.checkpoint) + ".json").read_text(encoding="utf-8"))
    assert metadata["output_file_sha256"] == expected_model
    assert metadata["model_tensor_sha256"] == "d86a4bb891e9badf94a9bf9e30cd59058a59c5dc21d75b37a52fa7be9000aac9"
    (output / "checkpoint.pt").write_bytes(model_bytes)
    shutil.copyfile(str(args.checkpoint) + ".json", output / "checkpoint.pt.json")
    # Same serialized doubles and IDs; no trigonometry/generator call here.
    shutil.copyfile(original / "cases.json", output / "cases.json")
    dataset = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert digest(dataset) == parent["cases_sha256"]
    commit = subprocess.check_output(["git", "rev-parse", "b33191d"], cwd=args.repository, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", commit, "src", "pyproject.toml"], cwd=args.repository)
    archive_path = output / "rl_best002_source.zip"
    archive_path.write_bytes(archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        sources = {name: hashlib.sha256(zipped.read(name)).hexdigest()
                   for name in zipped.namelist() if name.endswith(".py")}
    assert {p: sources["src/" + p] for p in CRITICAL} == parent["shared_physical_source_sha256"]
    spec = {"name": "rl_best002_u384", "entrypoint": "research_rl:run_rl_search",
            "kwargs": {"checkpoint": str((output / "checkpoint.pt").resolve()), "device": "cpu",
                       "deterministic": True, "num_threads": 1, "feature_version": "v3"}}
    frozen = {"name": "rl_best002_u384", "commit": commit, "spec": spec,
              "archive": archive_path.name, "archive_sha256": hashlib.sha256(archive).hexdigest(),
              "source_sha256": sources, "model_file_sha256": expected_model,
              "model_tensor_sha256": metadata["model_tensor_sha256"],
              "checkpoint_metadata_defaults": "Old missing architecture/distribution/action-schema metadata uses explicit validated MLP/flat/base defaults; model tensors unmodified."}
    manifest = {**parent, "policies": [frozen], "appended_to": str(original),
                "parent_manifest_sha256": hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest(),
                "checkpoint_origin": str(args.checkpoint.resolve()), "worker_python": str(args.python.resolve())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": parent["case_count"], "cases_sha256": parent["cases_sha256"],
                      "rl_commit": commit, "portable_model_sha256": expected_model}), flush=True)
    runner = Path(__file__).resolve().with_name("training_stress_reliability.py")
    return subprocess.run([str(args.python.resolve()), str(runner), "run", "--output", str(output), "--jobs", "1"],
                          cwd=Path(__file__).resolve().parents[1]).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    raise SystemExit(append(parser.parse_args()))

"""Build an anonymous, hash-locked CPU research package for object storage."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4"


def git(tree, *args):
    return subprocess.check_output(["git", "-c", "safe.directory="+tree.as_posix(), *args], cwd=tree)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_parent(path):
    """Initialization needs policy tensors, not old operator paths or command logs."""
    import torch
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("Use CPU-only PyTorch to package the model")
    assert digest(path.read_bytes()) == PARENT_SHA
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = ("algorithm", "hidden", "architecture", "action_distribution", "action_schema",
            "feature_version", "feature_schema", "model", "git_commit", "torch_version", "source_manifest")
    clean = {k: payload[k] for k in keys if k in payload}
    clean["state"] = {k: payload.get("state", {}).get(k) for k in
                      ("update", "optimizer_steps", "episodes", "attempted_episodes", "next_seed")}
    clean["initialization_source_sha256"] = PARENT_SHA
    clean["compute_policy"] = "CPU-only inference initialization; original optimizer and private paths omitted"
    buffer = io.BytesIO()
    torch.save(clean, buffer)
    data = buffer.getvalue()
    restored = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    assert all(torch.equal(payload["model"][k], restored["model"][k]) for k in payload["model"])
    return data


def build(args):
    tracked = git(ROOT, "ls-files", "-z").decode().split("\0")
    required = ["scripts/autonomy_runtime.py", "scripts/autonomy_job.py", "scripts/build_autonomy_package.py",
                "tests/test_autonomy_runtime.py", "tests/test_deep_rl_prefix_rollouts.py",
                "tests/test_deep_rl_certified_cover.py",
                "tests/test_deep_rl_joint_scan.py", "tests/test_deep_rl_controller.py",
                "tests/test_deep_rl_rollout_improvement.py",
                "scripts/sync_cpu_results.py", "research/autonomy/protocol.json", "research/v1_protocol.json",
                "experiments/__init__.py", "experiments/research_v1_eval.py", "experiments/run_q3_comparison.py",
                "research/theory_v1/audit_eval_bounds.py", "research/theory_v1/certify_bounds.py"]
    names = sorted(set(required) | {p.relative_to(ROOT).as_posix() for p in (ROOT/"src").rglob("*.py")}
                   | {p.relative_to(ROOT).as_posix() for p in (ROOT/"research/autonomy/plans").glob("*.json")})
    if not args.smoke:
        assert all(name in tracked for name in names), "Commit all package source before release"
        assert not git(ROOT, "status", "--porcelain", "--", *names).strip()
    members = {name: (ROOT/name).read_bytes() for name in names}
    # The original trained checkpoint has private operator paths; only the
    # exact policy and public provenance are placed in the exchanged package.
    members["models/parent.pt"] = public_parent(args.checkpoint)
    reference_tree = args.state_tree.resolve()
    protocol=json.loads((ROOT/"research/autonomy/protocol.json").read_text(encoding="utf-8-sig"))
    assert git(reference_tree,"rev-parse","HEAD").decode().strip()==protocol["state_reference_commit"]
    assert digest((reference_tree/args.state_spec).read_bytes())==protocol["state_reference_spec_sha256"]
    reference_spec = json.loads((reference_tree/args.state_spec).read_text(encoding="utf-8-sig"))
    reference_names = [p.relative_to(reference_tree).as_posix() for p in (reference_tree/"src").rglob("*.py")]
    reference_names += ["experiments/__init__.py", "experiments/research_v1_eval.py", "experiments/run_q3_comparison.py"]
    if not args.smoke:
        reference_tracked = git(reference_tree, "ls-files", "-z").decode().split("\0")
        assert all(name in reference_tracked for name in [*reference_names, args.state_spec])
        assert not git(reference_tree, "status", "--porcelain", "--", *reference_names, args.state_spec).strip()
    for name in reference_names:
        members["reference_state/"+name] = (reference_tree/name).read_bytes()
    for name in ("src/simulation/cases.py", "src/simulation/engine.py", "src/simulator_client/rules.py",
                 "src/simulator_client/client.py", "src/simulator_client/state.py", "src/geometry/__init__.py",
                 "src/simulation/__init__.py", "src/localization/omni.py",
                 "experiments/research_v1_eval.py", "experiments/run_q3_comparison.py"):
        assert members[name] == members["reference_state/"+name], "Shared evaluator/physics differs: "+name
    members["reference_state/spec.json"] = (json.dumps(reference_spec, indent=2)+"\n").encode()
    identity = dict(schema=1, source_commit=git(ROOT,"rev-parse","HEAD").decode().strip(),
        state_commit=git(reference_tree,"rev-parse","HEAD").decode().strip(),
        parent_original_sha256=PARENT_SHA, parent_packaged_sha256=digest(members["models/parent.pt"]),
        parent_policy_tensors_preserved=True, smoke_only=args.smoke,
        files={name:digest(value) for name,value in members.items()})
    members["PACKAGE_MANIFEST.json"] = (json.dumps(identity,indent=2)+"\n").encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as stream:
        with tarfile.open(fileobj=stream,mode="w:gz") as archive:
            for name,value in sorted(members.items()):
                info=tarfile.TarInfo("package/"+name);info.size=len(value);info.mode=0o644;info.mtime=0
                archive.addfile(info,io.BytesIO(value))
    manifest_path=args.output.with_suffix(args.output.suffix+".json")
    manifest_path.write_text(json.dumps({**identity,"archive_sha256":digest(args.output.read_bytes())},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"archive":args.output.name,"sha256":digest(args.output.read_bytes()),"files":len(members),"source_commit":identity["source_commit"]}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--state-tree",type=Path,required=True)
    p.add_argument("--state-spec",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    build(p.parse_args())

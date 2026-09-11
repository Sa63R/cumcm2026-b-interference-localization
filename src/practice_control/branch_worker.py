"""One fresh process for one audited, frozen Q3 branch profile.

Only the controller package comes from the main checkout. Solver modules and
their SimulatorClient base class come from the selected frozen source tree.
No simulator connection is made until provenance and imports pass preflight.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


MAIN_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = MAIN_ROOT.parent / "q3-v1-artifacts"
PROFILES = {
    "baseline": {
        "entrypoint": "strategies:run_search",
        "commit": "2f0a486fcdaecd0b03da2bea06ca8f845a9fdbae",
        "freeze_sha256": "4f749ad8613f064920da860b6a17d7b9bf387124c448d621c00d73f39743b9b9",
    },
    "state": {
        "entrypoint": "strategies.relocating_state_search:run_relocating_state_search",
        "commit": "8aa620649b609876b612d307b5be71ab588acc08",
        "freeze_sha256": "9496e9ccedec26c5a3d0e9a18142e715a0681a6b64f205c0a0f390071a9936d8",
    },
    "rl": {
        "entrypoint": "research_rl:run_rl_search",
        "commit": "de7d65b60ec05e4d1098f7b5a2c9c7a99c392adb",
        "freeze_sha256": "93dc2def82a83aea8d87ecf6a0112ef0177fe17d7503cc00de75735efd6132ae",
    },
    "geo": {
        "entrypoint": "strategies.geometric_relocation:run_relocation_search",
        "commit": "3c4f86fdb7409cdcf904d82ba166cb8564de8c7f",
        "freeze_sha256": "1ee40ad1440005c412d6a9c1f4cae4113af6e0905c04f50f6a035cf00c00777e",
    },
}
CHECKPOINT_NAME = "rl-gae095-u512.pt"
CHECKPOINT_SHA256 = "8985b7fc8d1709c5a54bb1aaf8f21df66f56782119eb901163f723a03281734f"
_SOLVER_PACKAGES = {"geometry", "localization", "planning", "simulation", "simulator_client",
                    "strategies", "research_rl", "workflow"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8"))


def _contained(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Frozen artifact resolves outside its expected directory")
    return path


def verify_profile(profile, delivery_root, restored_root):
    """Read and hash local source/configuration artifacts; never load a model here."""
    if profile not in PROFILES:
        raise ValueError("Unknown frozen branch profile")
    expected = PROFILES[profile]
    delivery = Path(delivery_root).resolve(strict=True)
    source = (Path(restored_root) / profile).resolve(strict=True)
    freeze_path = _contained(delivery, f"configs/{profile}/freeze.json")
    freeze_bytes = freeze_path.read_bytes()
    if digest(freeze_bytes) != expected["freeze_sha256"]:
        raise ValueError("Frozen provenance file differs from the audited version")
    freeze = json.loads(freeze_bytes)
    if freeze["git_commit"] != expected["commit"]:
        raise ValueError("Frozen source commit differs from the selected profile")
    identity = freeze["identity"]
    spec_path = _contained(delivery, f"configs/{profile}/spec.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if (spec.get("entrypoint") != expected["entrypoint"]
            or not isinstance(spec.get("kwargs"), dict)
            or not isinstance(spec.get("name"), str)):
        raise ValueError("Profile entrypoint or parameters differ from the reviewed interface")
    if canonical_digest(spec) != identity["spec_sha256"]:
        raise ValueError("Frozen strategy parameters were changed")
    for relative, expected_sha in identity["source_sha256"].items():
        if digest(_contained(source, relative).read_bytes()) != expected_sha:
            raise ValueError(f"Frozen source differs: {relative}")
    expected_sources = {p for p in identity["source_sha256"] if p.startswith("src/") and p.endswith(".py")}
    actual_sources = {p.relative_to(source).as_posix() for p in (source / "src").rglob("*.py")}
    if actual_sources != expected_sources:
        raise ValueError("Frozen source tree contains added or missing Python files")
    kwargs = copy.deepcopy(spec["kwargs"])
    metadata = {
        "profile": profile, "version_scope": "frozen_v1_branch",
        "method_name": spec["name"], "source_commit": expected["commit"],
        "source_root": str(source), "freeze_sha256": expected["freeze_sha256"],
        "spec_sha256": identity["spec_sha256"], "spec": spec,
        "source_file_count": len(identity["source_sha256"]),
        "controller_root": str(MAIN_ROOT), "python_executable": sys.executable,
    }
    if profile == "rl":
        if kwargs.get("checkpoint") != f"results/rl/v1-candidates/{CHECKPOINT_NAME}":
            raise ValueError("Unexpected RL checkpoint reference")
        checkpoint = _contained(delivery, f"models/{CHECKPOINT_NAME}")
        if (digest(checkpoint.read_bytes()) != CHECKPOINT_SHA256
                or identity["checkpoint_sha256"].get("checkpoint") != CHECKPOINT_SHA256):
            raise ValueError("RL checkpoint differs from the frozen model")
        kwargs["checkpoint"] = str(checkpoint)
        metadata["checkpoint_sha256"] = CHECKPOINT_SHA256
    elif "checkpoint" in kwargs or identity["checkpoint_sha256"]:
        raise ValueError("Unexpected model requirement for this profile")
    return source, spec, kwargs, metadata


def _solver_adapter(callback, kwargs):
    def solve(client, *, problem, variant, max_actions):
        if problem != 3:
            raise ValueError("Frozen four-branch methods support Q3 practice only")
        return callback(client, problem=3, max_actions=max_actions, **copy.deepcopy(kwargs))
    return solve


def load_solver(source, spec, kwargs):
    """Bootstrap before runner import; refuse reuse of an already-loaded solver."""
    if any(name.split(".", 1)[0] in _SOLVER_PACKAGES for name in sys.modules):
        raise ValueError("Frozen profiles require a fresh worker process before solver imports")
    # Deliberately omit the frozen repository root: result registration must
    # resolve to main/experiments, not to an old standalone research runner.
    sys.path[:0] = [str(source / "src"), str(MAIN_ROOT), str(MAIN_ROOT / "src")]
    module_name, function_name = spec["entrypoint"].split(":", 1)
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(source / "src"):
        raise ValueError("Strategy import did not resolve to the frozen source tree")
    callback = getattr(module, function_name)
    if not callable(callback):
        raise ValueError("Frozen strategy entrypoint is not callable")
    if kwargs.get("checkpoint"):
        # Load and validate model architecture/features before creating a case.
        from research_rl.network import load_policy
        load_policy(kwargs["checkpoint"], device=kwargs.get("device", "cpu"),
                    deterministic=kwargs.get("deterministic", True))
    from . import runner
    client_module = importlib.import_module("simulator_client")
    if not Path(client_module.__file__).resolve().is_relative_to(source / "src"):
        raise ValueError("HTTP client import did not resolve to the same frozen source tree")
    registrar = importlib.import_module("experiments.register_practice")
    if not Path(registrar.__file__).resolve().is_relative_to(MAIN_ROOT):
        raise ValueError("Result registrar did not resolve to the main controller repository")
    return runner, _solver_adapter(callback, kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one frozen repository-branch method in Q3 practice")
    parser.add_argument("--profile", required=True, choices=tuple(PROFILES))
    parser.add_argument("--delivery-root", type=Path, default=ARTIFACT_ROOT / "v1-delivery")
    parser.add_argument("--restored-root", type=Path, default=ARTIFACT_ROOT / "v1-restored-verification")
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--max-actions", type=int, default=20000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        source, spec, kwargs, metadata = verify_profile(args.profile, args.delivery_root, args.restored_root)
        runner, solver = load_solver(source, spec, kwargs)
        if args.preflight_only:
            print(json.dumps({"preflight": "passed", "simulator_requests_sent": False,
                              "method_metadata": metadata}, ensure_ascii=False))
            return 0
        runner.validate_run(3, "baseline", 1, args.max_actions, args.robot_id)
        if args.simulator_dir is None:
            raise ValueError("--simulator-dir is required to run practice")
        sim_dir = args.simulator_dir.resolve(strict=True)
        if not (sim_dir / "jammers-simulator-full.exe").is_file():
            raise ValueError("simulator-dir must contain the original installed executable")
        output = args.output or MAIN_ROOT / "results/practice_batches" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-frozen-" + args.profile)
        from .bridge import PracticeBridge
        with runner.controller_lock(sim_dir / ".practice-control/controller.lock"):
            with PracticeBridge(args.debug_port) as bridge:
                record = runner.run_once(
                    bridge, problem=3, robot_id=args.robot_id, variant="baseline",
                    max_actions=args.max_actions, output=output, simulator_dir=sim_dir,
                    solver=solver, method_label=f"frozen-v1/{args.profile}/{spec['name']}",
                    method_metadata=metadata,
                )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return 0 if record["completed"] else 1
    except Exception as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

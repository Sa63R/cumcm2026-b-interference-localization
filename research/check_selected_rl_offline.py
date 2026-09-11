"""Check the restored selected source/model and one already-used training seed.

This handoff check is separate from the frozen algorithm. It has no test-split
option, training update, simulator HTTP request, or official-test operation.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args()
    source, freeze = args.source.resolve(), args.freeze.resolve()
    audit_script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    os.chdir(source)
    sys.path[:0] = [str(source / "src"), str(source)]
    import numpy as np
    import torch
    from experiments import run_q3_practice
    from experiments.research_v1_eval import identity, read_json, run_case
    from research_rl.network import load_policy
    from research_rl.portable_checkpoint import model_tensor_digest
    from simulation import random_scenario
    from simulator_client import SimulatorClient

    torch.set_num_threads(1)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert commit == "de7d65b60ec05e4d1098f7b5a2c9c7a99c392adb"
    spec_path = Path("research/v1-candidate-rl-gae095-u512.json")
    spec, protocol, registered = map(read_json, [spec_path, Path("research/v1_protocol.json"), freeze])
    observed_identity = identity(spec, protocol)
    assert registered["git_commit"] == commit and observed_identity == registered["identity"]
    assert spec["kwargs"]["checkpoint"] == "results/rl/v1-candidates/rl-gae095-u512.pt"
    checkpoint = Path(spec["kwargs"]["checkpoint"])
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checksum == "8985b7fc8d1709c5a54bb1aaf8f21df66f56782119eb901163f723a03281734f"
    attempts = []

    def forbidden(*a, **kw):
        attempts.append("network/client attempted")
        raise AssertionError("Offline handoff check cannot connect to a simulator")

    with patch.object(SimulatorClient, "__init__", forbidden), \
            patch("socket.create_connection", forbidden), patch("socket.socket.connect", forbidden), \
            patch("urllib.request.OpenerDirector.open", forbidden):
        policy = load_policy(checkpoint, device="cpu", deterministic=True)
        record = run_case(random_scenario(3, 100121), spec, protocol)
        assert record["row"]["successful"] and record["row"]["failed_clear_count"] == 0
        output = io.StringIO()
        with redirect_stdout(output):
            assert run_q3_practice.main(["--spec", str(spec_path), "--dry-run"]) == 0
        dry = json.loads(output.getvalue())
    assert not attempts and dry["simulator_requests_sent"] is False
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ancestry, state = [], payload["state"]
    while state:
        ancestry.append({k: state[k] for k in ("update", "episodes", "next_seed")})
        state = state.get("initialization", {}).get("source_state")
    assert [r["episodes"] for r in ancestry] == [16384, 12288, 12288, 16384]
    result = dict(schema=1, scope="offline restored-source compatibility; one already-used training seed",
                  git_commit=commit, checkpoint_sha256=checksum,
                  tensor_sha256=model_tensor_digest(policy.model.state_dict()),
                  freeze_file_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
                  identity_exact=True, identity=observed_identity, ancestry_newest_first=ancestry,
                  ancestry_policy_episodes=sum(r["episodes"] for r in ancestry),
                  parameters=sum(p.numel() for p in policy.model.parameters()),
                  feature_version=policy.feature_version, hidden=policy.model.hidden,
                  architecture=policy.architecture, distribution=policy.action_distribution,
                  action_schema=policy.action_schema, python=platform.python_version(),
                  torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                  num_threads=1, row=record["row"], network_attempts=attempts,
                  practice_dry_run=True, weight_unchanged=True, audit_script_sha256=audit_script_sha)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

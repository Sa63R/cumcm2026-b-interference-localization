"""Explicit, content-bound loading of completed micro BC weights only."""
import hashlib
import io
from pathlib import Path

import torch

from .micro_network import validate_checkpoint


INITIALIZATION_TYPE = "frozen-micro-bc-weights-v1"


def initialization_binding(sha256):
    if (not isinstance(sha256, str) or len(sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in sha256)):
        raise ValueError("initialization SHA256 must contain exactly 64 hexadecimal characters")
    return {"type": INITIALIZATION_TYPE, "sha256": sha256.lower()}


def load_micro_warmstart(path, sha256, *, expected_metadata=None):
    """Check the exact bytes before decoding; never restore optimizer or RNG."""
    binding = initialization_binding(sha256)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("micro warmstart SHA256 mismatch")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    state, config = saved.get("state"), saved.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("micro initialization requires a completed BC transaction")
    target = config.get("warmstart_episodes")
    completed = state.get("warmstart_completed")
    if (type(target) is not int or target <= 0 or type(completed) is not int
            or completed != target or type(state.get("ppo_batches")) is not int
            or state["ppo_batches"] != 0 or "pending_batch" not in state
            or state["pending_batch"] is not None
            or state.get("episodes") != target):
        raise ValueError("micro initialization requires complete BC, zero PPO and no pending batch")
    if expected_metadata is not None and saved["network"] != expected_metadata:
        raise ValueError("micro initialization network metadata differs from requested architecture/hidden")
    if any(value.device.type != "cpu" for value in saved["model"].values()):
        raise ValueError("micro initialization model must contain CPU tensors")
    return saved

"""Explicit G3 completed-BC weights only; provenance is self-contained on resume."""
import hashlib
import io
from pathlib import Path

import torch


INITIALIZATION_TYPE = "frozen-g3-memory-bc-weights-v1"


def initialization_binding(sha256):
    if (not isinstance(sha256, str) or len(sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in sha256)):
        raise ValueError("initialization SHA256 must contain exactly 64 hexadecimal characters")
    return {"type": INITIALIZATION_TYPE, "sha256": sha256.lower()}


def validate_initialization_binding(binding):
    if (not isinstance(binding, dict) or set(binding) != {"type", "sha256"}
            or binding != initialization_binding(binding.get("sha256"))):
        raise ValueError("invalid G3 weights-only initialization binding; cross-schema reuse is forbidden")
    return dict(binding)


def load_memory_warmstart(path, sha256, *, expected_metadata=None):
    """Verify exact bytes, schema and complete BC transaction without restoring state."""
    from .memory_network import validate_checkpoint
    binding = initialization_binding(sha256)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("G3 memory warmstart SHA256 mismatch")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    state, config = saved.get("state"), saved.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("G3 initialization requires a completed BC transaction")
    target, completed = config.get("warmstart_episodes"), state.get("warmstart_completed")
    if (type(target) is not int or target <= 0 or type(completed) is not int
            or completed != target or type(state.get("ppo_batches")) is not int
            or state["ppo_batches"] != 0 or "pending_batch" not in state
            or state["pending_batch"] is not None or type(state.get("episodes")) is not int
            or state["episodes"] != target):
        raise ValueError("G3 initialization requires complete BC, zero PPO and no pending batch")
    if expected_metadata is not None and saved["network"] != expected_metadata:
        raise ValueError("G3 initialization network metadata differs from requested architecture/hidden")
    if any(value.device.type != "cpu" for value in saved["model"].values()):
        raise ValueError("G3 initialization model must contain CPU tensors")
    return saved

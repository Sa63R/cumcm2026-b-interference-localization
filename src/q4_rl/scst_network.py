"""SCST checkpoint identity over the unchanged public 13/50 micro architecture."""
from pathlib import Path

import torch

from . import micro_network
from .micro_network import (CONTROLLER_ENTRYPOINT, GLOBAL_DIM, CANDIDATE_DIM,
                            configure_cpu, feature_schema, pack_observations)
from .network import TorchPolicy
from .micro_initialization import initialization_binding, load_micro_warmstart

ARCHITECTURE = "q4-micro-scst-candidate-mlp-v1"
CHECKPOINT_VERSION = "q4-micro-scst-cpu-v1"
ALGORITHM = "self-critical-sequence-training-v1"
OBJECTIVE = {"gamma": 1.0, "cost_unit_s": 1000.0, "failure_penalty_s": 360000.0,
             "baseline": "same_scene_current_greedy", "entropy_coefficient": 0.0,
             "reduction": "mean_trajectory_sum_log_probability", "updates_per_batch": 1}


class SCSTCandidateActorCritic(micro_network.MicroCandidateActorCritic):
    """Value head is preserved for exact initialization, but receives no SCST loss."""
    def metadata(self):
        return {**super().metadata(), "architecture": ARCHITECTURE}


def model_from_metadata(metadata):
    if (metadata.get("architecture") != ARCHITECTURE or
            metadata.get("global_dim") != GLOBAL_DIM or metadata.get("candidate_dim") != CANDIDATE_DIM):
        raise ValueError("SCST architecture/public dimension mismatch")
    return SCSTCandidateActorCritic(hidden=metadata["hidden"])


def validate_checkpoint(saved):
    if (not isinstance(saved, dict) or saved.get("version") != CHECKPOINT_VERSION or
            saved.get("algorithm") != ALGORITHM or saved.get("controller_entrypoint") != CONTROLLER_ENTRYPOINT or
            saved.get("feature_schema") != feature_schema() or saved.get("device") != "cpu" or
            saved.get("objective") != OBJECTIVE):
        raise ValueError("SCST checkpoint algorithm/controller/objective/schema mismatch")
    if saved.get("config", {}).get("max_decisions") != 512:
        raise ValueError("SCST checkpoint must retain the frozen 512-decision controller")


def initialize_micro_warmstart(path, expected_sha256):
    """Explicit, hash-bound migration of weights only; never import PPO optimizer."""
    configure_cpu()
    saved = load_micro_warmstart(path, expected_sha256)
    if saved["config"].get("max_decisions") != 512:
        raise ValueError("Initialization requires a completed 512-decision imitation-only micro warmstart")
    original = micro_network.model_from_metadata(saved["network"])
    original.load_state_dict(saved["model"])
    model = SCSTCandidateActorCritic(hidden=original.hidden)
    model.load_state_dict(original.state_dict())
    return model, initialization_binding(expected_sha256)


class RolloutPolicy(TorchPolicy):
    def __init__(self, model, *, deterministic):
        super().__init__(model, deterministic=deterministic)

    def __call__(self, global_features, candidate_features):
        action = super().__call__(global_features, candidate_features)
        if self.deterministic:
            self.records.clear()  # Greedy evidence keeps physical requests, not unused training tensors.
        return action


def load_policy(checkpoint, *, deterministic=True):
    configure_cpu()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    model.eval()
    return RolloutPolicy(model, deterministic=deterministic)

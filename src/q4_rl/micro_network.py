"""Strictly versioned CPU model for Q4 G1 micro decisions.

The tensor operations are shared with the frozen candidate MLP. Checkpoints
have distinct architecture, controller and complete feature semantics; a macro
checkpoint is never implicitly resized or migrated.
"""
from pathlib import Path

from .network import (CandidateActorCritic, TorchPolicy, configure_cpu,
                      pack_observations as _pack_observations)
import torch


ARCHITECTURE = "q4-micro-candidate-mlp-v1"
CHECKPOINT_VERSION = "q4-micro-ppo-cpu-v1"
CONTROLLER_ENTRYPOINT = "q4_rl.micro_controller:run_q4_micro"
FEATURE_SCHEMA_VERSION = "q4-micro-g1-v1"
GLOBAL_DIM, CANDIDATE_DIM = 13, 50
OBJECTIVE = {"gamma": 1.0, "lambda": 1.0, "cost_unit_s": 1000.0,
             "failure_penalty_s": 360000.0}


def feature_schema():
    from . import micro_controller as controller
    if (controller.FEATURE_SCHEMA_VERSION != FEATURE_SCHEMA_VERSION or
            controller.GLOBAL_DIM != GLOBAL_DIM or controller.CANDIDATE_DIM != CANDIDATE_DIM):
        raise ValueError("micro controller version/dimension contract changed")
    return {"version": FEATURE_SCHEMA_VERSION,
            "global_features": list(controller.GLOBAL_FEATURE_NAMES),
            "candidate_features": list(controller.CANDIDATE_FEATURE_NAMES)}


class MicroCandidateActorCritic(CandidateActorCritic):
    def __init__(self, hidden=64):
        super().__init__(GLOBAL_DIM, CANDIDATE_DIM, hidden)

    def metadata(self):
        return {**super().metadata(), "architecture": ARCHITECTURE}


def model_from_metadata(metadata):
    if (metadata.get("architecture") != ARCHITECTURE or
            metadata.get("global_dim") != GLOBAL_DIM or
            metadata.get("candidate_dim") != CANDIDATE_DIM):
        raise ValueError("unsupported micro network architecture/dimensions; no implicit macro migration")
    return MicroCandidateActorCritic(hidden=metadata["hidden"])


def pack_observations(records):
    return _pack_observations(records, global_dim=GLOBAL_DIM, candidate_dim=CANDIDATE_DIM)


def validate_checkpoint(saved):
    if (not isinstance(saved, dict) or saved.get("version") != CHECKPOINT_VERSION or
            saved.get("feature_schema") != feature_schema() or
            saved.get("controller_entrypoint") != CONTROLLER_ENTRYPOINT):
        raise ValueError("micro checkpoint version/controller/public feature semantics mismatch")
    if saved.get("device") != "cpu" or saved.get("objective") != OBJECTIVE:
        raise ValueError("micro checkpoint CPU/undiscounted objective contract mismatch")


def load_policy(checkpoint, *, deterministic=True):
    configure_cpu()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    model.eval()
    return TorchPolicy(model, deterministic=deterministic)

"""Independent CPU MLP/checkpoint contract for public scan-bundle decisions."""
from pathlib import Path

import torch

from .network import (CandidateActorCritic, TorchPolicy, configure_cpu,
                      pack_observations as _pack_observations)
from .bundle_controller import (GLOBAL_DIM, CANDIDATE_DIM, FEATURE_SCHEMA_VERSION,
                                feature_schema)


ARCHITECTURE = "q4-scan-bundle-candidate-mlp-v1"
CHECKPOINT_VERSION = "q4-scan-bundle-ppo-cpu-v1"
CONTROLLER_ENTRYPOINT = "q4_rl.bundle_controller:run_q4_bundle"
OBJECTIVE = {"gamma": 1.0, "lambda": 1.0, "cost_unit_s": 1000.0,
             "failure_penalty_s": 360000.0}


class BundleCandidateActorCritic(CandidateActorCritic):
    def __init__(self, hidden=64):
        super().__init__(GLOBAL_DIM, CANDIDATE_DIM, hidden)

    def metadata(self):
        return {**super().metadata(), "architecture": ARCHITECTURE}


def architecture_name(metadata):
    if (not isinstance(metadata, dict)
            or set(metadata) != {"architecture", "global_dim", "candidate_dim", "hidden"}
            or metadata.get("architecture") != ARCHITECTURE
            or metadata.get("global_dim") != GLOBAL_DIM or metadata.get("candidate_dim") != CANDIDATE_DIM
            or type(metadata.get("hidden")) is not int or metadata["hidden"] <= 0):
        raise ValueError("scan-bundle network metadata mismatch; cross-schema migration is unsupported")
    return "mlp"


def make_model(architecture="mlp", *, hidden=64):
    if architecture != "mlp":
        raise ValueError("scan-bundle currently supports its own MLP only")
    return BundleCandidateActorCritic(hidden=hidden)


def model_from_metadata(metadata):
    architecture_name(metadata)
    return BundleCandidateActorCritic(hidden=metadata["hidden"])


def pack_observations(records):
    return _pack_observations(records, global_dim=GLOBAL_DIM, candidate_dim=CANDIDATE_DIM)


def validate_checkpoint(saved):
    if (not isinstance(saved, dict) or saved.get("version") != CHECKPOINT_VERSION
            or saved.get("controller_entrypoint") != CONTROLLER_ENTRYPOINT
            or saved.get("feature_schema") != feature_schema()
            or saved.get("device") != "cpu" or saved.get("objective") != OBJECTIVE):
        raise ValueError("scan-bundle checkpoint version/controller/schema/CPU/objective mismatch")
    if saved.get("config", {}).get("architecture", "mlp") != architecture_name(saved.get("network")):
        raise ValueError("scan-bundle training configuration and network architecture differ")
    if saved.get("config", {}).get("initialization"):
        raise ValueError("scan-bundle does not support cross-schema warmstart initialization")


def load_policy(checkpoint, *, deterministic=True):
    configure_cpu()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    model.eval()
    return TorchPolicy(model, deterministic=deterministic)

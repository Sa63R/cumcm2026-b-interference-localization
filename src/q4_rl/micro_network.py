"""Strictly versioned CPU model for Q4 G1 micro decisions.

The tensor operations are shared with the frozen candidate MLP. Checkpoints
have distinct architecture, controller and complete feature semantics; a macro
checkpoint is never implicitly resized or migrated.
"""
from pathlib import Path

from .network import (CandidateActorCritic, TorchPolicy, configure_cpu,
                      pack_observations as _pack_observations)
import torch
from torch import nn


ARCHITECTURE = "q4-micro-candidate-mlp-v1"
INDUCED_ARCHITECTURE = "q4-micro-candidate-induced-v1"
INDUCED_SETTINGS = dict(inducing_points=16, attention_heads=2, attention_dim=16,
                        attention_blocks=1, attention_refinement="residual")
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


class InducedSetBlock(nn.Module):
    """16 latent slots exchange information with candidates in O(16N) space.

    Adapted from the macro induced16/dim16/residual experiment. No positions,
    identity embeddings, extra observations or dropout are introduced.
    """
    def __init__(self, hidden):
        super().__init__()
        width = INDUCED_SETTINGS["attention_dim"]
        self.inducing = nn.Parameter(torch.empty(16, width))
        nn.init.normal_(self.inducing, std=width**-.5)
        self.input_projection = nn.Identity() if hidden == width else nn.Linear(hidden, width)
        self.output_projection = nn.Identity() if hidden == width else nn.Linear(width, hidden)
        self.to_latents = nn.MultiheadAttention(width, 2, dropout=0., batch_first=True)
        self.to_candidates = nn.MultiheadAttention(width, 2, dropout=0., batch_first=True)
        self.latent_norm = nn.LayerNorm(width)
        self.latent_ff = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.latent_output_norm = nn.LayerNorm(width)

    def forward(self, encoded, mask):
        encoded = encoded.masked_fill(~mask[..., None], 0.)
        projected = self.input_projection(encoded)
        inducing = self.inducing.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        message, _ = self.to_latents(inducing, projected, projected,
                                    key_padding_mask=~mask, need_weights=False)
        latents = self.latent_norm(inducing+message)
        latents = self.latent_output_norm(latents+self.latent_ff(latents))
        message, _ = self.to_candidates(projected, latents, latents, need_weights=False)
        return (encoded+self.output_projection(message)).masked_fill(~mask[..., None], 0.)


class MicroInducedActorCritic(MicroCandidateActorCritic):
    def __init__(self, hidden=64):
        super().__init__(hidden)
        self.set_blocks = nn.ModuleList([InducedSetBlock(hidden)])

    def metadata(self):
        return {**super().metadata(), "architecture": INDUCED_ARCHITECTURE, **INDUCED_SETTINGS}

    def forward(self, global_features, candidate_features, mask):
        if (global_features.device.type != "cpu" or candidate_features.device.type != "cpu"
                or mask.device.type != "cpu" or next(self.parameters()).device.type != "cpu"):
            raise ValueError("Q4 training and inference are CPU-only")
        if mask.dtype != torch.bool or mask.ndim != 2 or not bool(mask.any(1).all()):
            raise ValueError("every observation needs at least one legal candidate")
        if (candidate_features.ndim != 3 or global_features.ndim != 2 or
                candidate_features.shape[:2] != mask.shape or
                global_features.shape != (mask.shape[0], self.global_dim) or
                candidate_features.shape[2] != self.candidate_dim):
            raise ValueError("feature dimensions do not match model")
        encoded = self.encoder(candidate_features.masked_fill(~mask[..., None], 0.))
        for block in self.set_blocks:
            encoded = block(encoded, mask)
        # Preserve the original mean/max pooling, actor and critic heads.
        mean = (encoded*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
        context = self.context(torch.cat((global_features, mean, maximum), dim=-1))
        joint = torch.cat((encoded, context[:, None, :].expand(-1, encoded.shape[1], -1)), -1)
        return self.actor(joint).squeeze(-1).masked_fill(~mask, -torch.inf), self.critic(context).squeeze(-1)


def architecture_name(metadata):
    """Validate exact metadata without constructing a model or advancing RNG."""
    if (not isinstance(metadata, dict) or metadata.get("global_dim") != GLOBAL_DIM
            or metadata.get("candidate_dim") != CANDIDATE_DIM):
        raise ValueError("unsupported micro network dimensions; no implicit macro migration")
    architecture = metadata.get("architecture")
    required = {"architecture", "global_dim", "candidate_dim", "hidden"}
    if architecture == INDUCED_ARCHITECTURE:
        required |= INDUCED_SETTINGS.keys()
        if any(metadata.get(k) != v for k, v in INDUCED_SETTINGS.items()):
            raise ValueError("incomplete or mismatched induced16 attention metadata")
    elif architecture != ARCHITECTURE:
        raise ValueError("unsupported micro network architecture")
    if set(metadata) != required or type(metadata["hidden"]) is not int or metadata["hidden"] <= 0:
        raise ValueError("incomplete or unexpected micro network metadata")
    return "induced" if architecture == INDUCED_ARCHITECTURE else "mlp"


def make_model(architecture="mlp", *, hidden=64):
    if architecture == "mlp":
        return MicroCandidateActorCritic(hidden=hidden)
    if architecture == "induced":
        return MicroInducedActorCritic(hidden=hidden)
    raise ValueError("unknown micro architecture")


def model_from_metadata(metadata):
    return make_model(architecture_name(metadata), hidden=metadata["hidden"])


def pack_observations(records):
    return _pack_observations(records, global_dim=GLOBAL_DIM, candidate_dim=CANDIDATE_DIM)


def validate_checkpoint(saved):
    if (not isinstance(saved, dict) or saved.get("version") != CHECKPOINT_VERSION or
            saved.get("feature_schema") != feature_schema() or
            saved.get("controller_entrypoint") != CONTROLLER_ENTRYPOINT):
        raise ValueError("micro checkpoint version/controller/public feature semantics mismatch")
    if saved.get("device") != "cpu" or saved.get("objective") != OBJECTIVE:
        raise ValueError("micro checkpoint CPU/undiscounted objective contract mismatch")
    if saved.get("config", {}).get("architecture", "mlp") != architecture_name(saved.get("network")):
        raise ValueError("checkpoint training configuration and network architecture differ")


def load_policy(checkpoint, *, deterministic=True):
    configure_cpu()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    model.eval()
    return TorchPolicy(model, deterministic=deterministic)

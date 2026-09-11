"""CPU-only, permutation-equivariant candidate actor and invariant critic.

The model sees only controller-provided public features. Physical feature scales
are fixed in the controller; no validation statistics, scene ids or lower bounds
are accepted here. Padded candidates cannot affect either output.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import math

import torch
from torch import nn
from torch.distributions import Categorical


ARCHITECTURE = "q4-candidate-mlp-v1"
CHECKPOINT_VERSION = "q4-ppo-cpu-v1"


def configure_cpu():
    """One compute thread per learner/worker; never initialize a CUDA device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before the first parallel work.
        if torch.get_num_interop_threads() != 1:
            raise


def feature_schema():
    from .controller import (GLOBAL_FEATURE_NAMES, CANDIDATE_FEATURE_NAMES,
                             FEATURE_SCHEMA_VERSION)
    return {"version": FEATURE_SCHEMA_VERSION,
            "global_features": list(GLOBAL_FEATURE_NAMES),
            "candidate_features": list(CANDIDATE_FEATURE_NAMES)}


class CandidateActorCritic(nn.Module):
    def __init__(self, global_dim=10, candidate_dim=16, hidden=64):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (global_dim, candidate_dim, hidden)):
            raise ValueError("network dimensions must be positive integers")
        self.global_dim, self.candidate_dim, self.hidden = global_dim, candidate_dim, hidden
        self.encoder = nn.Sequential(nn.Linear(candidate_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(global_dim + 2 * hidden, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def metadata(self):
        return {"architecture": ARCHITECTURE, "global_dim": self.global_dim,
                "candidate_dim": self.candidate_dim, "hidden": self.hidden}

    def forward(self, global_features, candidate_features, mask):
        if global_features.device.type != "cpu" or candidate_features.device.type != "cpu":
            raise ValueError("Q4 training and inference are CPU-only")
        if mask.dtype != torch.bool or mask.ndim != 2 or not bool(mask.any(1).all()):
            raise ValueError("every observation needs at least one legal candidate")
        if (candidate_features.ndim != 3 or global_features.ndim != 2 or
                candidate_features.shape[:2] != mask.shape or
                global_features.shape != (mask.shape[0], self.global_dim) or
                candidate_features.shape[2] != self.candidate_dim):
            raise ValueError("feature dimensions do not match model")
        encoded = self.encoder(candidate_features)
        mean = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
        context = self.context(torch.cat((global_features, mean, maximum), dim=-1))
        joint = torch.cat((encoded, context[:, None, :].expand(-1, encoded.shape[1], -1)), -1)
        logits = self.actor(joint).squeeze(-1).masked_fill(~mask, -torch.inf)
        return logits, self.critic(context).squeeze(-1)


def pack_observations(records, *, global_dim=10, candidate_dim=16):
    if not records:
        raise ValueError("cannot pack an empty observation batch")
    maximum = max(len(row["candidate_features"]) for row in records)
    if maximum == 0:
        raise ValueError("empty legal action set")
    context = torch.zeros((len(records), global_dim), dtype=torch.float32)
    candidates = torch.zeros((len(records), maximum, candidate_dim), dtype=torch.float32)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool)
    for i, row in enumerate(records):
        g, c = row["global_features"], row["candidate_features"]
        if len(g) != global_dim or not c or any(len(v) != candidate_dim for v in c):
            raise ValueError("invalid observation feature dimensions")
        if not all(math.isfinite(float(v)) for v in g) or not all(
                math.isfinite(float(v)) for candidate in c for v in candidate):
            raise ValueError("non-finite public features")
        context[i] = torch.tensor(g, dtype=torch.float32)
        candidates[i, :len(c)] = torch.tensor(c, dtype=torch.float32)
        mask[i, :len(c)] = True
    return context, candidates, mask


class TorchPolicy:
    """The callback returns only an index; optional training details stay local."""
    def __init__(self, model, *, deterministic=True):
        configure_cpu()
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("CPU model required")
        self.model, self.deterministic = model, bool(deterministic)
        self.records = []

    @torch.no_grad()
    def __call__(self, global_features, candidate_features):
        row = {"global_features": list(global_features),
               "candidate_features": [list(x) for x in candidate_features]}
        logits, value = self.model(*pack_observations(
            [row], global_dim=self.model.global_dim, candidate_dim=self.model.candidate_dim))
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if self.deterministic else distribution.sample()
        row.update(action_index=int(action.item()), log_prob=float(distribution.log_prob(action).item()),
                   value=float(value.item()))
        self.records.append(row)
        return row["action_index"]


def model_from_metadata(metadata):
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported Q4 network architecture")
    return CandidateActorCritic(metadata["global_dim"], metadata["candidate_dim"], metadata["hidden"])


def load_policy(checkpoint, *, deterministic=True):
    """Load our own tensor/primitive checkpoint without arbitrary pickle globals."""
    configure_cpu()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if saved.get("version") != CHECKPOINT_VERSION or saved.get("feature_schema") != feature_schema():
        raise ValueError("checkpoint version or public feature semantics mismatch")
    model = model_from_metadata(saved["network"])
    model.load_state_dict(saved["model"])
    model.eval()
    return TorchPolicy(model, deterministic=deterministic)

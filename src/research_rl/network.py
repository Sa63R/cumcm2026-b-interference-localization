"""Masked set actor/critic and batched PPO tensors (PyTorch optional elsewhere)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .controller import ALGORITHM_VERSIONS, CONTEXT_DIM, FEATURE_DIM, FEATURE_DIMS, feature_schema


class CandidateActorCritic(nn.Module):
    """Permutation-equivariant actor, invariant critic over legal candidates.

    Every action is chosen by the trained network. The teacher index is absent
    from forward inputs. The mean/max pooled context communicates the remaining
    task set; this compact architecture does not claim to reproduce AlphaGo.
    """
    def __init__(self, hidden=96, feature_dim=FEATURE_DIM):
        super().__init__()
        self.hidden = hidden
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.context_encoder = nn.Sequential(
            nn.Linear(CONTEXT_DIM + 2 * hidden, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, features, context, mask):
        encoded = self.encoder(features)
        mean = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = encoded.masked_fill(~mask[..., None], -1e9).amax(1)
        pooled = self.context_encoder(torch.cat((context, mean, maximum), dim=-1))
        joint = torch.cat((encoded, pooled[:, None, :].expand(-1, encoded.shape[1], -1)), dim=-1)
        logits = self.actor(joint).squeeze(-1).masked_fill(~mask, -1e9)
        return logits, self.critic(pooled).squeeze(-1)


def pack_observations(records, device="cpu"):
    maximum = max(len(r["features"]) for r in records)
    feature_dim = len(records[0]["features"][0])
    features = np.zeros((len(records), maximum, feature_dim), dtype=np.float32)
    mask = np.zeros((len(records), maximum), dtype=np.bool_)
    for row, record in enumerate(records):
        count = len(record["features"])
        features[row, :count] = record["features"]
        mask[row, :count] = True
    context = np.asarray([r["context"] for r in records], dtype=np.float32)
    return (torch.as_tensor(features, device=device), torch.as_tensor(context, device=device),
            torch.as_tensor(mask, device=device))


class TorchPolicy:
    def __init__(self, model, *, device="cpu", deterministic=True, teacher=False):
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.teacher = teacher
        self.feature_version = next(v for v, dim in FEATURE_DIMS.items() if dim == model.feature_dim)

    @torch.no_grad()
    def __call__(self, features, context, teacher):
        tensors = pack_observations([dict(features=features, context=context)], self.device)
        logits, value = self.model(*tensors)
        distribution = Categorical(logits=logits)
        action = (torch.tensor([teacher], device=self.device) if self.teacher else
                  logits.argmax(-1) if self.deterministic else distribution.sample())
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


@lru_cache(maxsize=8)
def _load_cached(path, modified_ns, device, deterministic):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    versions = {algorithm: version for version, algorithm in ALGORITHM_VERSIONS.items()}
    version = versions.get(checkpoint.get("algorithm"))
    if version is None:
        raise ValueError("Unsupported checkpoint algorithm version")
    if ((version != "v1" or "feature_schema" in checkpoint)
            and checkpoint.get("feature_schema") != feature_schema(version)):
        raise ValueError("checkpoint feature semantics do not match this implementation")
    model = CandidateActorCritic(checkpoint["hidden"], FEATURE_DIMS[version]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return TorchPolicy(model, device=device, deterministic=deterministic)


def load_policy(checkpoint, *, device="cpu", deterministic=True):
    path = Path(checkpoint).resolve()
    return _load_cached(str(path), path.stat().st_mtime_ns, device, deterministic)

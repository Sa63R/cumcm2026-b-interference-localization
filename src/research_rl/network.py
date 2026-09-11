"""Masked set actor/critic and batched PPO tensors (PyTorch optional elsewhere)."""

from functools import lru_cache
from pathlib import Path

from .cpu_runtime import require_cpu

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .controller import ALGORITHM_VERSIONS, CONTEXT_DIM, FEATURE_DIM, FEATURE_DIMS, feature_schema
from .distributions import (adjusted_logits, validate_distribution, checkpoint_distribution,
                            sample_probe_diagnostics)
from .action_sets import ANYPOINT_SCHEMAS, validate_action_schema, checkpoint_action_schema


def architecture_spec(name="mlp", layers=1, heads=4):
    if name == "mlp":
        return {"version": 1, "name": "mlp"}
    if name != "attention":
        raise ValueError("architecture must be mlp or attention")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers not in (1, 2):
        raise ValueError("attention layers must be 1 or 2")
    if isinstance(heads, bool) or not isinstance(heads, int) or not 1 <= heads <= 8:
        raise ValueError("attention heads must be an integer in 1..8")
    return {"version": 1, "name": "attention", "layers": layers, "heads": heads,
            "residual_gate": "zero_initialized", "position_encoding": "none", "dropout": 0.0}


def validate_architecture(spec):
    if spec is None:
        return architecture_spec()
    if not isinstance(spec, dict):
        raise ValueError("architecture metadata must be a dictionary")
    expected = architecture_spec(spec.get("name"), spec.get("layers", 1), spec.get("heads", 4))
    if spec != expected:
        raise ValueError("unknown architecture metadata/version")
    return expected


def architecture_from_args(args):
    """Keep existing paired-training callers with older Namespaces compatible."""
    return architecture_spec(getattr(args, "architecture", "mlp"),
                             getattr(args, "attention_layers", 1),
                             getattr(args, "attention_heads", 4))


def checkpoint_architecture(payload):
    spec = payload.get("architecture")
    relation_keys = [key for key in payload.get("model", {}) if key.startswith("relations.")]
    if spec is None and relation_keys:
        raise ValueError("attention checkpoint is missing architecture metadata")
    result = validate_architecture(spec)
    if relation_keys and result["name"] == "mlp":
        raise ValueError("attention parameters contradict MLP architecture metadata")
    if result["name"] == "attention":
        valid_prefixes = tuple(f"relations.{i}." for i in range(result["layers"]))
        if any(not key.startswith(valid_prefixes) for key in relation_keys):
            raise ValueError("attention parameters contradict declared layer count")
    return result


class GatedSetAttention(nn.Module):
    """Content-only self-attention, with trainable zero-initialized residual gates.

    No positional index embeddings: candidate permutation changes only output
    order. Key padding masks prevent fake candidate rows from affecting real rows.
    At initialization the map is the identity; gates receive gradients on the
    first backward pass and branch weights receive gradients once gates open.
    """
    def __init__(self, hidden, heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.Tanh(),
                                         nn.Linear(2 * hidden, hidden))
        self.attention_gate = nn.Parameter(torch.zeros(()))
        self.feedforward_gate = nn.Parameter(torch.zeros(()))

    def forward(self, encoded, mask):
        normalized = self.attention_norm(encoded)
        delta, _ = self.attention(normalized, normalized, normalized,
                                  key_padding_mask=~mask, need_weights=False)
        encoded = encoded + self.attention_gate * delta
        return encoded + self.feedforward_gate * self.feedforward(self.feedforward_norm(encoded))


class ZeroFeatureResidual(nn.Module):
    """Bias-free linear map constructed without consuming random numbers."""
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden, input_dim))

    def forward(self, features):
        return nn.functional.linear(features, self.weight)


class CandidateActorCritic(nn.Module):
    """Permutation-equivariant actor, invariant critic over legal candidates.

    Every action is chosen by the trained network. The teacher index is absent
    from forward inputs. The mean/max pooled context communicates the remaining
    task set; this compact architecture does not claim to reproduce AlphaGo.
    """
    def __init__(self, hidden=96, feature_dim=FEATURE_DIM, architecture=None, action_distribution=None,
                 action_schema=None):
        super().__init__()
        self.hidden = hidden
        self.feature_dim = feature_dim
        self.architecture = validate_architecture(architecture)
        self.action_distribution = validate_distribution(action_distribution)
        self.action_schema = validate_action_schema(action_schema)
        if feature_dim == FEATURE_DIMS["v4"] and (
                self.architecture["name"] != "mlp" or self.action_distribution["name"] != "flat"
                or self.action_schema["name"] != "base"):
            raise ValueError("v4 route-debt ablation requires MLP, flat distribution and base actions")
        if self.action_distribution["name"] != "flat" and feature_dim != 60:
            raise ValueError("group distribution requires v3 feature semantics")
        if self.action_schema["name"] != "base" and feature_dim != 60:
            raise ValueError("candidate extension requires v3 feature semantics")
        if self.action_schema["name"] in ANYPOINT_SCHEMAS and (
                self.architecture["name"] != "mlp" or self.action_distribution["name"] != "flat"):
            raise ValueError("anypoint scan ablation requires MLP and flat distribution")
        if self.architecture["name"] == "attention" and hidden % self.architecture["heads"]:
            raise ValueError("hidden width must be divisible by attention heads")
        input_dim = 60 if feature_dim == FEATURE_DIMS["v4"] else feature_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.context_encoder = nn.Sequential(
            nn.Linear(CONTEXT_DIM + 2 * hidden, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)
        # Create the new branches AFTER the entire legacy network, preserving
        # the legacy initialization stream and original state_dict key names.
        self.relations = nn.ModuleList()
        if self.architecture["name"] == "attention":
            # Training constructs parameters on CPU before .to(device). Keep
            # the branch's original random weights but do not shift the outer
            # stream used by matched MLP/attention minibatch permutations.
            with torch.random.fork_rng(devices=[]):
                self.relations.extend(GatedSetAttention(hidden, self.architecture["heads"])
                                      for _ in range(self.architecture["layers"]))
        self.route_adapter = None
        if feature_dim == FEATURE_DIMS["v4"]:
            # Preserve the old 60-column GEMM shape and accumulation order.
            # A full 76-column GEMM with zero columns is only algebraically equal.
            self.route_adapter = ZeroFeatureResidual(feature_dim - input_dim, hidden)

    def forward(self, features, context, mask):
        if self.route_adapter is None:
            encoded = self.encoder(features)
        else:
            # Contiguous old prefix also preserves its input stride and kernel.
            legacy = self.encoder[0](features[..., :60].contiguous())
            added = self.route_adapter(features[..., 60:].contiguous())
            encoded = self.encoder[3](self.encoder[2](self.encoder[1](legacy + added)))
        for relation in self.relations:
            encoded = relation(encoded, mask)
        mean = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = encoded.masked_fill(~mask[..., None], -1e9).amax(1)
        pooled = self.context_encoder(torch.cat((context, mean, maximum), dim=-1))
        joint = torch.cat((encoded, pooled[:, None, :].expand(-1, encoded.shape[1], -1)), dim=-1)
        logits = self.actor(joint).squeeze(-1).masked_fill(~mask, -1e9)
        logits = adjusted_logits(logits, features, mask, self.action_distribution)
        return logits, self.critic(pooled).squeeze(-1)


def pack_observations(records, device="cpu"):
    require_cpu(device)
    if not records or any(len(r["features"]) == 0 for r in records):
        raise ValueError("each observation needs at least one legal candidate")
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
    def __init__(self, model, *, device="cpu", deterministic=True, teacher=False,
                 capture_diagnostics=False):
        require_cpu(device)
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.teacher = teacher
        self.architecture = model.architecture
        self.action_distribution = model.action_distribution
        self.action_schema = model.action_schema
        self.feature_version = next(v for v, dim in FEATURE_DIMS.items() if dim == model.feature_dim)
        self.capture_diagnostics = capture_diagnostics and self.feature_version in {"v3", "v4"}
        self.center_mask = None
        self.last_probe_diagnostics = None

    def prepare_candidates(self, candidates, regions):
        if not self.capture_diagnostics:
            return
        # Legal geometric metadata for logging only; never a forward input.
        centers = {channel: regions[channel].enclosing_disk().center
                   for channel in {c.channel for c in candidates if c.kind == "probe"}}
        self.center_mask = [c.kind == "probe" and
            (c.point.x - centers[c.channel][0]) ** 2 +
            (c.point.y - centers[c.channel][1]) ** 2 <= 1e-12 for c in candidates]

    @torch.no_grad()
    def __call__(self, features, context, teacher):
        tensors = pack_observations([dict(features=features, context=context)], self.device)
        logits, value = self.model(*tensors)
        distribution = Categorical(logits=logits)
        action = (torch.tensor([teacher], device=self.device) if self.teacher else
                  logits.argmax(-1) if self.deterministic else distribution.sample())
        if self.capture_diagnostics:
            if self.center_mask is None or len(self.center_mask) != len(features):
                raise ValueError("diagnostic policy requires current legal candidate metadata")
            self.last_probe_diagnostics = sample_probe_diagnostics(
                logits[0].cpu().numpy(), features, self.center_mask, int(action.item()))
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
    model = CandidateActorCritic(checkpoint["hidden"], FEATURE_DIMS[version],
                                 checkpoint_architecture(checkpoint),
                                 checkpoint_distribution(checkpoint),
                                 checkpoint_action_schema(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return TorchPolicy(model, device=device, deterministic=deterministic)


def load_policy(checkpoint, *, device="cpu", deterministic=True):
    require_cpu(device)
    path = Path(checkpoint).resolve()
    return _load_cached(str(path), path.stat().st_mtime_ns, device, deterministic)

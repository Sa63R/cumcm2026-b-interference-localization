"""CPU candidate-set GRU with separate candidate and sequence padding masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .cpu_runtime import require_cpu
from .controller import FEATURE_DIMS, feature_schema, ALGORITHM_VERSIONS
from .network import (CandidateActorCritic, TorchPolicy, pack_observations,
                      checkpoint_architecture)
from .distributions import checkpoint_distribution
from .action_sets import checkpoint_action_schema

ALGORITHM = "q3-candidate-memory-gru-ppo-v1"


def memory_spec(memory_hidden=64):
    if type(memory_hidden) is not int or not 1 <= memory_hidden <= 256:
        raise ValueError("memory_hidden must be an integer in 1..256")
    return dict(version=1, cell="GRU", layers=1, hidden=memory_hidden,
                input="v3-masked-candidate-pooled-context", actor_critic="shared",
                readout="zero-initialized-bias-free-residual", dropout=0.0,
                sequence="whole-episode-BPTT", reset="zero-at-every-episode-start",
                previous_actions="inferred-only-through-subsequent-public-observation",
                padding="distinct-time-and-candidate-masks")


class RecurrentCandidateActorCritic(CandidateActorCritic):
    def __init__(self, hidden=96, memory_hidden=64):
        require_cpu()
        super().__init__(hidden, FEATURE_DIMS["v3"])
        self.memory_spec = memory_spec(memory_hidden)
        self.memory_hidden = memory_hidden
        self.memory = nn.GRU(hidden, memory_hidden, num_layers=1)
        self.memory_readout = nn.Linear(memory_hidden, hidden, bias=False)
        nn.init.zeros_(self.memory_readout.weight)

    def zero_state(self, batch_size=1):
        return next(self.parameters()).new_zeros((1, batch_size, self.memory_hidden))

    def _encode(self, features, context, mask):
        if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Expected [batch,candidate,feature] and bool candidate mask")
        encoded = self.encoder(features)
        mean = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = encoded.masked_fill(~mask[..., None], -1e9).amax(1)
        pooled = self.context_encoder(torch.cat((context, mean, maximum), dim=-1))
        return encoded, pooled

    def _heads(self, encoded, pooled, memory, mask):
        enriched = pooled + self.memory_readout(memory)
        joint = torch.cat((encoded, enriched[:, None, :].expand(-1, encoded.shape[1], -1)), dim=-1)
        logits = self.actor(joint).squeeze(-1).masked_fill(~mask, -1e9)
        return logits, self.critic(enriched).squeeze(-1)

    def forward_step(self, features, context, mask, hidden=None, episode_start=None):
        if not mask.any(1).all():
            raise ValueError("Every live step needs a legal candidate")
        encoded, pooled = self._encode(features, context, mask)
        hidden = self.zero_state(len(features)) if hidden is None else hidden
        if hidden.shape != (1, len(features), self.memory_hidden):
            raise ValueError("Recurrent hidden-state shape mismatch")
        if episode_start is not None:
            if episode_start.shape != (len(features),) or episode_start.dtype != torch.bool:
                raise ValueError("episode_start must be one bool per environment")
            hidden = hidden * (~episode_start)[None, :, None]
        memory, next_hidden = self.memory(pooled[None], hidden)
        logits, values = self._heads(encoded, pooled, memory[0], mask)
        return logits, values, next_hidden

    def forward_sequence(self, features, context, candidate_mask, time_mask):
        """Each batch column is one complete, chronological episode from zero.

        Packed recurrence never sees padding, so padded states cannot propagate
        to any valid step. Gradients cross every real step of each episode.
        """
        if features.ndim != 4 or time_mask.shape != features.shape[:2] or time_mask.dtype != torch.bool:
            raise ValueError("Expected time-major episodes and a bool time mask")
        time, batch, candidates, width = features.shape
        if candidate_mask.shape != features.shape[:3] or candidate_mask.dtype != torch.bool:
            raise ValueError("Candidate mask shape/dtype mismatch")
        lengths = time_mask.sum(0)
        expected = torch.arange(time)[:, None] < lengths[None, :]
        if not lengths.gt(0).all() or not torch.equal(time_mask, expected):
            raise ValueError("Time mask must contain a nonempty contiguous episode prefix")
        if not torch.equal(candidate_mask.any(-1), time_mask):
            raise ValueError("Exactly live steps must contain legal candidates")
        encoded, pooled = self._encode(features.reshape(time * batch, candidates, width),
                                       context.reshape(time * batch, -1),
                                       candidate_mask.reshape(time * batch, candidates))
        packed = pack_padded_sequence(pooled.reshape(time, batch, -1), lengths.cpu(), enforce_sorted=False)
        sequence, final_hidden = self.memory(packed, self.zero_state(batch))
        memory, _ = pad_packed_sequence(sequence, total_length=time)
        logits, values = self._heads(encoded, pooled, memory.reshape(time * batch, -1),
                                     candidate_mask.reshape(time * batch, candidates))
        return logits.reshape(time, batch, candidates), values.reshape(time, batch), final_hidden

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use forward_step with explicit state or forward_sequence with complete episodes")


def pack_episodes(episodes):
    require_cpu()
    if not episodes or any(not episode for episode in episodes):
        raise ValueError("Each sequence must be a nonempty complete episode")
    steps = max(map(len, episodes))
    count = len(episodes)
    candidates = max(len(r["features"]) for episode in episodes for r in episode)
    features = np.zeros((steps, count, candidates, FEATURE_DIMS["v3"]), dtype=np.float32)
    context = np.zeros((steps, count, 12), dtype=np.float32)
    masks = np.zeros((steps, count, candidates), dtype=np.bool_)
    time_mask = np.zeros((steps, count), dtype=np.bool_)
    fields = {name: np.zeros((steps, count), dtype=np.int64 if name == "action" else np.float32)
              for name in ("action", "log_prob", "advantage", "return")}
    for b, episode in enumerate(episodes):
        for t, record in enumerate(episode):
            n = len(record["features"])
            if n < 1 or record.get("episode_start") is not (t == 0) or record.get("terminal") is not (t == len(episode) - 1):
                raise ValueError("Sequence must have exactly one start and terminal, with legal candidates")
            if not 0 <= record["action"] < n:
                raise ValueError("Logged action is outside the legal candidate set")
            features[t, b, :n] = record["features"]
            context[t, b] = record["context"]
            masks[t, b, :n] = True
            time_mask[t, b] = True
            for name, values in fields.items():
                values[t, b] = record[name]
    return dict(features=torch.from_numpy(features), context=torch.from_numpy(context),
                candidate_mask=torch.from_numpy(masks), time_mask=torch.from_numpy(time_mask),
                **{name: torch.from_numpy(value) for name, value in fields.items()})


def initialize_from_mlp(model, payload):
    if (payload.get("algorithm") != ALGORITHM_VERSIONS["v3"]
            or payload.get("feature_schema") != feature_schema("v3")
            or payload.get("hidden") != model.hidden
            or checkpoint_architecture(payload)["name"] != "mlp"
            or checkpoint_distribution(payload)["name"] != "flat"
            or checkpoint_action_schema(payload)["name"] != "base"):
        raise ValueError("Warm start requires an exact v3/base/flat MLP of the same hidden width")
    reference = CandidateActorCritic(model.hidden, FEATURE_DIMS["v3"])
    reference.load_state_dict(payload["model"], strict=True)
    weights = model.state_dict()
    weights.update(reference.state_dict())
    model.load_state_dict(weights, strict=True)
    nn.init.zeros_(model.memory_readout.weight)


class RecurrentPolicy(TorchPolicy):
    def __init__(self, model, *, deterministic=True):
        super().__init__(model, deterministic=deterministic, capture_diagnostics=False)
        self.reset()

    def reset(self):
        self.hidden = None
        self.steps = 0

    @torch.no_grad()
    def __call__(self, features, context, teacher):
        tensors = pack_observations([dict(features=features, context=context)])
        logits, value, self.hidden = self.model.forward_step(*tensors, hidden=self.hidden,
            episode_start=torch.tensor([self.steps == 0], dtype=torch.bool))
        self.steps += 1
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if self.deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


def model_from_checkpoint(payload):
    if (payload.get("algorithm") != ALGORITHM or payload.get("feature_schema") != feature_schema("v3")
            or payload.get("memory_spec") != memory_spec(payload.get("memory_hidden", 64))
            or checkpoint_architecture(payload)["name"] != "mlp"
            or checkpoint_distribution(payload)["name"] != "flat"
            or checkpoint_action_schema(payload)["name"] != "base"):
        raise ValueError("Unknown recurrent checkpoint algorithm or observation/action/memory semantics")
    model = RecurrentCandidateActorCritic(payload["hidden"], payload["memory_hidden"])
    model.load_state_dict(payload["model"], strict=True)
    return model


def load_recurrent_policy(checkpoint, *, device="cpu", deterministic=True):
    require_cpu(device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    # Never cache a mutable policy hidden state across episodes.
    model = model_from_checkpoint(payload).eval()
    return RecurrentPolicy(model, deterministic=deterministic)

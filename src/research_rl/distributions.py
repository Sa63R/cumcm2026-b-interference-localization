"""Versioned action-probability ablation; physical candidates stay unchanged."""

import math

import numpy as np
import torch


def distribution_spec(alpha=0):
    if isinstance(alpha, bool) or alpha not in (0, 1):
        raise ValueError("group alpha must be exactly 0 or 1")
    if alpha == 0:
        return {"version": 1, "name": "flat"}
    return {"version": 1, "name": "group_cardinality", "alpha": 1,
            "groups": "v3-cover-position-or-source-channel"}


def validate_distribution(spec):
    if spec is None:
        return distribution_spec()
    if not isinstance(spec, dict):
        raise ValueError("action distribution metadata must be a dictionary")
    expected = distribution_spec(0 if spec.get("name") == "flat" else spec.get("alpha"))
    if spec != expected:
        raise ValueError("unknown action distribution metadata/version")
    return expected


def distribution_from_args(args):
    return distribution_spec(getattr(args, "group_alpha", 0))


def checkpoint_distribution(payload):
    spec = payload.get("action_distribution")
    args = payload.get("args", {})
    if spec is None and args.get("group_alpha", 0) != 0:
        raise ValueError("grouped checkpoint is missing action distribution metadata")
    result = validate_distribution(spec)
    if "group_alpha" in args and result != distribution_spec(args["group_alpha"]):
        raise ValueError("action distribution contradicts checkpoint arguments")
    return result


def group_sizes(features, mask):
    """Exact group cardinalities from the existing v3 60-feature schema.

    Cover groups share exactly the encoded point; source groups share the
    discrete channel. Equality uses no candidate index, so it is equivariant.
    Pairwise comparison is intentionally simple/auditable; max v3 N is small.
    """
    if features.shape[-1] != 60:
        raise ValueError("group distribution requires v3 feature semantics")
    cover = features[..., 0] > 0.5
    same_point = ((features[:, :, None, 4] == features[:, None, :, 4]) &
                  (features[:, :, None, 5] == features[:, None, :, 5]))
    channels = (features[..., 45] * 20).round().long()
    same_source = channels[:, :, None] == channels[:, None, :]
    same_group = ((cover[:, :, None] & cover[:, None, :] & same_point) |
                  (~cover[:, :, None] & ~cover[:, None, :] & same_source))
    return (same_group & mask[:, :, None] & mask[:, None, :]).sum(-1)


def adjusted_logits(logits, features, mask, spec):
    if spec["name"] == "flat":
        # Preserve legacy outputs and RNG trajectories exactly.
        return logits
    sizes = group_sizes(features, mask)
    return (logits - sizes.clamp_min(1).to(logits.dtype).log()).masked_fill(~mask, -1e9)


def empty_probe_diagnostics():
    return dict(source_groups=0, entropy_sum=0.0, normalized_entropy_sum=0.0,
                center_available_groups=0, center_probability_sum=0.0,
                selected_probes=0, selected_center_probes=0)


def sample_probe_diagnostics(logits, features, center_mask, action):
    """Behavior-policy conditional entropy, from the actual sampled forward.

    Only source groups with >=2 probe options enter entropy averages. The center
    flags come from legal enclosing disks and never enter the actor/critic.
    They distinguish a true center from a fallback teacher option numbered zero.
    """
    result = empty_probe_diagnostics()
    groups = {}
    for index, row in enumerate(features):
        if row[1] > 0.5:
            groups.setdefault(round(row[45] * 20), []).append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = np.asarray(logits)[indices].astype(np.float64)
        values -= values.max()
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        entropy = -sum(float(p) * math.log(max(float(p), 1e-300)) for p in probabilities)
        result["source_groups"] += 1
        result["entropy_sum"] += entropy
        result["normalized_entropy_sum"] += entropy / math.log(len(indices))
        centers = [bool(center_mask[index]) for index in indices]
        if any(centers):
            result["center_available_groups"] += 1
            result["center_probability_sum"] += float(probabilities[centers].sum())
    if features[action][1] > 0.5:
        result["selected_probes"] = 1
        result["selected_center_probes"] = int(center_mask[action])
    return result


def merge_probe_diagnostics(records):
    totals = empty_probe_diagnostics()
    for record in records:
        for key in totals:
            totals[key] += record.get(key, 0)
    groups, centers = totals["source_groups"], totals["center_available_groups"]
    return dict(totals,
                conditional_entropy=totals["entropy_sum"] / groups if groups else None,
                normalized_conditional_entropy=totals["normalized_entropy_sum"] / groups if groups else None,
                center_conditional_probability=totals["center_probability_sum"] / centers if centers else None)

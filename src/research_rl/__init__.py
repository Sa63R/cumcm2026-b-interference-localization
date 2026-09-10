"""Independent Q3 deep reinforcement learning research; no Q4 changes."""

from .controller import DeepRLSearch


def run_rl_search(client, *, problem=3, max_actions=20000, checkpoint=None,
                  device="cpu", deterministic=True, max_decisions=256,
                  policy=None, max_active_probes=6, num_threads=1, feature_version=None):
    """Observation-only evaluator entry point; policy files are trusted artifacts.

    ``checkpoint`` is loaded once per process/path/mtime by the network module.
    A supplied callable ``policy(features, context, teacher)`` permits tests
    without installing torch. No simulator truth is accepted by this API.
    """
    if problem not in (3, "q3"):
        raise ValueError("deep RL is Q3 only")
    if policy is None:
        if checkpoint is None:
            raise ValueError("checkpoint or explicit policy is required")
        import torch
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        torch.set_num_threads(num_threads)
        from .network import load_policy
        policy = load_policy(checkpoint, device=device, deterministic=deterministic)
    if feature_version is None:
        feature_version = getattr(policy, "feature_version", "v2")
    if getattr(policy, "feature_version", feature_version) != feature_version:
        raise ValueError("policy and requested feature versions differ")
    if feature_version == "v3":
        from .joint_scan import JointScanRLSearch
        controller = JointScanRLSearch
    else:
        controller = DeepRLSearch
    return controller(client, policy, max_actions=max_actions,
                        max_decisions=max_decisions,
                        max_active_probes=max_active_probes,
                        feature_version=feature_version).run()


__all__ = ["DeepRLSearch", "run_rl_search"]

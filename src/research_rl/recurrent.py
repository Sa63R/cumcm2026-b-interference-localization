"""Fresh per-episode recurrent policy on unchanged v3/base candidate actions."""
from .cpu_runtime import require_cpu
from .action_sets import controller_for, action_schema
from .recurrent_network import ALGORITHM, load_recurrent_policy, RecurrentPolicy


def run_recurrent_search(client, *, problem=3, max_actions=20000, checkpoint=None,
                         device="cpu", deterministic=True, max_decisions=256,
                         max_active_probes=6, num_threads=1, policy=None, recorder=None,
                         action_deadline_epoch=None):
    require_cpu(device)
    if problem not in (3, "q3") or type(num_threads) is not int or num_threads < 1:
        raise ValueError("Only Q3 and positive CPU thread counts are supported")
    import torch
    torch.set_num_threads(num_threads)
    if policy is None:
        if checkpoint is None:
            raise ValueError("checkpoint or recurrent policy required")
        policy = load_recurrent_policy(checkpoint, device=device, deterministic=deterministic)
    if not isinstance(policy, RecurrentPolicy):
        raise ValueError("Expected a resettable RecurrentPolicy")
    policy.reset()
    try:
        report = controller_for("v3", action_schema("base"))(
            client, policy, max_actions=max_actions, max_decisions=max_decisions,
            max_active_probes=max_active_probes, feature_version="v3", recorder=recorder,
            action_deadline_epoch=action_deadline_epoch).run()
        report.learning["algorithm"] = ALGORITHM
        report.learning["memory_spec"] = policy.model.memory_spec
        report.learning["memory_steps"] = policy.steps
        report.learning["memory_reset_at_episode_start"] = True
        return report
    finally:
        policy.reset()

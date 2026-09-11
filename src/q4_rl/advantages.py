"""Optional actor GAE; gamma is one and critic targets remain full MC costs."""
import math


def validate_gae_lambda(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("GAE lambda must be finite and inside [0, 1]")
    return float(value)


def checkpoint_objective(legacy_objective, config):
    """Keep legacy metadata exact; make nondefault actor/critic semantics explicit."""
    gae_lambda = validate_gae_lambda(config.get("gae_lambda", 1.))
    if gae_lambda == 1.:
        return dict(legacy_objective)
    return {**legacy_objective, "lambda": gae_lambda,
            "actor_advantage": "gae", "critic_target": "undiscounted_mc"}


def attach_actor_advantages(records, gae_lambda):
    """Attach to exactly one complete executed episode, never across episodes.

    Costs already include fallback. Only the separately recorded terminal failure
    adjustment is added. No reward, return, observation or action is overwritten.
    Lambda one deliberately does nothing: PPO uses its exact legacy float32
    returns-minus-old-value operation, including its original rounding order.
    """
    gae_lambda = validate_gae_lambda(gae_lambda)
    if gae_lambda == 1.:
        return
    if not records:
        return
    from .train import COST_UNIT_S
    if not records[-1].get("terminal") or any(row.get("terminal") for row in records[:-1]):
        raise ValueError("actor GAE requires exactly one complete terminal episode")
    next_value, advantage = 0., 0.
    pending = []
    for index in range(len(records)-1, -1, -1):
        row = records[index]
        value, cost = float(row["value"]), float(row["cost_s"])
        penalty = float(row.get("terminal_penalty_s", 0.))
        if (not all(math.isfinite(x) for x in (value, cost, penalty))
                or cost < 0 or penalty < 0 or (index != len(records)-1 and penalty != 0)):
            raise ValueError("invalid GAE old value or actual/terminal cost")
        # Match the fixed training cost unit; no fitted normalization or duration discount.
        reward = -(cost / COST_UNIT_S + penalty / COST_UNIT_S)
        advantage = reward + next_value - value + gae_lambda * advantage
        if not math.isfinite(advantage):
            raise ValueError("non-finite actor advantage")
        pending.append((row, advantage))
        next_value = value
    for row, advantage in pending:
        row["actor_advantage"] = advantage

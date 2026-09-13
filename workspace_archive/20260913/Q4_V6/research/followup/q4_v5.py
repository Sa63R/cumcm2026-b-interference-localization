"""Public entry points for the evaluated literature-inspired research solver.

Default: hierarchical Rao-Blackwellised local rollout + posterior shared sensing.
Global: full history-conditioned, all-source V4 continuation; more expensive.
Neither entry claims a published benchmark SOTA or per-case policy improvement.
"""
from __future__ import annotations
from dataclasses import replace
from q4_local_rollout import LocalRolloutConfig,solve_local_rollout
from q4_rollout import RolloutConfig,solve_rollout
from q4_v4_solver import V4Config


def default_config() -> LocalRolloutConfig:
    """Frozen on 18 development cases, before the new holdout was generated."""
    return LocalRolloutConfig(base=V4Config(),scenes=24,position_draws=128,min_gain=2.,
        max_decisions=40,max_free_probes=40,seed=67541,total_planning_seconds=120.,
        dynamic_coverage=False,rb_sharing=True,share_threshold=10.)


def global_config() -> RolloutConfig:
    return RolloutConfig(base=V4Config(),max_decisions=24,scenes=12,validation_scenes=6,
        position_draws=96,risk_weight=.01,stderr_weight=.1,min_gain=.3,
        decision_seconds=15.,rich_candidates=False,validate_winner=True)


def solve_v5(device,config:LocalRolloutConfig|None=None) -> dict:
    """device must expose position/channel/move/detect/clear, not hidden data."""
    return solve_local_rollout(device,config or default_config())


def solve_v5_global(device,config:RolloutConfig|None=None) -> dict:
    """Research alternative; actual full closed-loop rollout, not oracle routing."""
    return solve_rollout(device,config or global_config())

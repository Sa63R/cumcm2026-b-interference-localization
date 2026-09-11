"""Observation-only conditional worlds and public clients for Fresh Q3 rollout.

This deliberately reuses the audited ``q3_belief`` mathematics and
``simulation.q3_branch`` accounting instead of creating a second posterior.
``FreshQ3.history`` already uses their action-history schema, including failed
clear attempts. The nominal model is an approximation for action selection,
never an absence or termination certificate:

* N is uniform on 10..16 before conditioning; channel subsets include the
  1 / choose(20, N) factor and shared count constraint after conditioning.
* Position/radius pairs use finite pools weighted by compatible radius length;
  no-signal requires radius strictly below the observed source distance.
* Bearings use the same conservative 1.005-degree compatibility band as
  FreshQ3's CandidateRegion, not an identified official error likelihood.
* Successfully cleared historical sources remain in each initial world, but
  the branch's public cleared state removes them from all future feedback.
* Historical feedback is anchored. Future error is keyed by the world seed,
  channel, and exact coordinates (with signed zero normalized), independently
  of candidate evaluation order.

Sample the worlds ONCE per decision and reuse that list for every candidate.
Catch BeliefSamplingError (including BeliefSamplingTimeout) to select the
frozen baseline action without modifying any geometric completion evidence.
The caller must separately copy its FreshQ3 policy state/continuation; a public
client fork cannot reconstruct pending policy decisions from client state.
"""

from collections.abc import Sequence

from simulation.cases import Scenario
from simulation.engine import MemoryClient
from simulation.q3_branch import make_q3_branch
from simulator_client.state import ClientState
from .q3_belief import (BeliefSamplingError, BeliefSamplingTimeout,
                        sample_worlds as _sample_worlds)


def sample_worlds(history: Sequence[dict], *, count: int, seed: int,
                  deadline: float | None = None) -> list[Scenario]:
    """Return compatible hypothetical worlds using only legal action history.

    ``deadline`` is an absolute ``time.perf_counter()`` deadline. Cancellation
    raises rather than returning a partial list or declaring a channel absent.
    Neither the live client, its private exchange callback, nor evaluator data
    is accepted by this interface.
    """
    return _sample_worlds(history, count=count, seed=seed, deadline=deadline)


def make_branch(scenario: Scenario, public_state: ClientState,
                history: Sequence[dict]) -> MemoryClient:
    """Restore an independent public client in one sampled hypothetical world.

    Position, measuring channel, cumulative costs, action counters, observed
    source states, and remaining virtual allowance are copied without replay.
    This client remains active after the last hypothetical source is removed;
    the continuation must obtain its own certificate and explicitly exit.
    The caller's real deadline remains its responsibility and is not extended.
    """
    return make_q3_branch(scenario, public_state, history)


__all__ = ["BeliefSamplingError", "BeliefSamplingTimeout", "sample_worlds", "make_branch"]

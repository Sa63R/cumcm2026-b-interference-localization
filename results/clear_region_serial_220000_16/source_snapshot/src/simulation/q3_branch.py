"""Observation-preserving Q3 continuation in a caller-supplied hypothesis.

This module has no access to a live simulator or evaluation data. The caller is
responsible for sampling a scenario consistent with the complete observations;
validation here covers the public session state and history format. Historical
feedback is anchored at exact coordinates, not regenerated from a new seed.
"""

from collections.abc import Sequence
import copy
from dataclasses import asdict
import math
import time

from simulator_client.state import ClientState, Position, SourceState, TimeBreakdown
from .cases import Scenario
from .engine import LocalResearchSimulator, MemoryClient


_LOCAL_REAL_BUDGET_S = 1200.0


def _finite(value, name, *, minimum=0.0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _channel(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
        raise ValueError("Channels must be integers in 1..20")
    return value


def _feedback_anchors(history, state):
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("history must be a sequence of action-history dictionaries")
    anchors = {}
    removed = set()
    for item in history:
        if not isinstance(item, dict) or item.get("action") not in {"measure", "clear"}:
            raise ValueError("history must use SearchResult.action_history format")
        channel = _channel(item.get("channel"))
        position = Position.coerce(item.get("position"))
        if "virtual_time_s" in item:
            timestamp = _finite(item["virtual_time_s"], "history virtual_time_s")
            if timestamp > state.virtual_time_s + 1e-5:
                raise ValueError("Historical action is later than the supplied state")
        result = item.get("result")
        if item["action"] == "clear":
            if result not in {"success", "no_target_in_range"}:
                raise ValueError("Invalid historical clear result")
            if result == "success":
                source = state.sources.get(channel)
                if source is None or source.status != "cleared" or channel in removed:
                    raise ValueError("Successful clear history contradicts supplied state")
                removed.add(channel)
            continue
        if result not in {"direction", "near", "no_signal"}:
            raise ValueError("Invalid historical measure result")
        feedback = {"measure_result": result}
        if result == "direction":
            bearing = _finite(item.get("bearing_deg"), "bearing_deg")
            if bearing >= 360:
                raise ValueError("bearing_deg must be in [0, 360)")
            feedback["svd_deg"] = bearing
        if channel in removed:
            if result != "no_signal":
                raise ValueError("History observes a source after successful removal")
            # A post-removal absence is not a property of the active source.
            continue
        key = (channel, position.x, position.y)
        if key in anchors and anchors[key] != feedback:
            raise ValueError("Repeated active-source measurement has conflicting feedback")
        anchors[key] = feedback
    return anchors


class _AnchoredQ3Simulator(LocalResearchSimulator):
    def __init__(self, scenario, anchors, **kwargs):
        super().__init__(scenario, **kwargs)
        self._anchors = anchors

    def _execute(self, path, payload):
        response = super()._execute(path, payload)
        if path == "/measure" and payload["channel"] not in self._cleared:
            point = payload["position"]
            anchor = self._anchors.get((payload["channel"], point["x"], point["y"]))
            if anchor is not None:
                response.pop("svd_deg", None)
                response.update(anchor)
                self._history[-1]["response"] = response.copy()
        return response


def make_q3_branch(scenario: Scenario, state: ClientState,
                   history: Sequence[dict]) -> MemoryClient:
    """Fork an ACTIVE public client without replaying or rebilling its past.

    ``scenario`` is a hypothetical world supplied by the belief sampler, never a
    real engine. Its sources and reception radii stay fixed throughout this
    branch. Public observations, position, channel, cumulative virtual time,
    budget, and action counters are copied. A fresh local 1200-second monotonic
    deadline replaces the caller's absolute real deadline; it cannot extend the
    real session. Historical anchors retain feedback, never historical times.

    The returned client has the normal explicit ``exit()`` protocol. It does not
    automatically end when the hypothesis's last source is removed. No policy
    should inspect its private exchange callback or hypothetical engine.
    """
    if not isinstance(scenario, Scenario) or scenario.problem != 3:
        raise ValueError("make_q3_branch requires a Question 3 Scenario")
    if not isinstance(state, ClientState) or state.session != "active":
        raise ValueError("make_q3_branch requires an active ClientState")
    Position.coerce(state.position)
    _channel(state.current_channel)
    virtual_time = _finite(state.virtual_time_s, "virtual_time_s")
    maximum = _finite(state.max_virtual_duration_s, "max_virtual_duration_s")
    if maximum == 0:
        raise ValueError("max_virtual_duration_s must be positive")
    if (isinstance(state.accepted_actions, bool)
            or not isinstance(state.accepted_actions, int) or state.accepted_actions < 0):
        raise ValueError("accepted_actions must be a nonnegative integer")
    if not isinstance(state.time_breakdown, TimeBreakdown) or not isinstance(state.sources, dict):
        raise ValueError("Invalid public time breakdown or source states")
    components = asdict(state.time_breakdown)
    for name, value in components.items():
        _finite(value, name)
    # The public client sums unrounded movement; the engine rounds each action
    # to microseconds. Allow that accumulated difference without rebilling it.
    tolerance = 1e-5 + state.accepted_actions * 0.5e-6
    if abs(sum(components.values()) - virtual_time) > tolerance:
        raise ValueError("Time breakdown is inconsistent with cumulative virtual time")
    hypothetical_channels = {source.channel for source in scenario.sources}
    cleared = set()
    for channel, source in state.sources.items():
        _channel(channel)
        if not isinstance(source, SourceState) or source.status not in {"unknown", "detected", "cleared"}:
            raise ValueError("Invalid public source state")
        if source.status != "unknown" and channel not in hypothetical_channels:
            raise ValueError("Hypothesis omits an observed source channel")
        if source.status == "cleared":
            cleared.add(channel)
    anchors = _feedback_anchors(history, state)
    clock = time.monotonic
    simulator = _AnchoredQ3Simulator(
        copy.deepcopy(scenario), anchors, max_virtual_duration_s=maximum,
        max_real_duration_s=_LOCAL_REAL_BUDGET_S, clock=clock)
    simulator._session = "active"
    simulator._started = clock()
    simulator._position = Position.coerce(state.position)
    simulator._channel = state.current_channel
    simulator._microseconds = round(virtual_time * 1_000_000)
    simulator._components_us = {name: round(value * 1_000_000)
                                for name, value in components.items()}
    # Public movement accumulates unrounded distances. Reconcile the private
    # integer ledger to the authoritative total, while retaining the exact
    # supplied public breakdown below.
    simulator._components_us["movement_s"] += (
        simulator._microseconds - sum(simulator._components_us.values()))
    simulator._cleared = cleared
    client = simulator.client()
    client.state = copy.deepcopy(state)
    client.state.position = simulator._position
    client.state.real_deadline = simulator._started + _LOCAL_REAL_BUDGET_S
    client._sequence = state.accepted_actions
    return client

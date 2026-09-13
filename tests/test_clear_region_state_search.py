"""Paired development cases and clearance proofs rebuilt from real responses."""

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.state import Position
from strategies.clear_region_state_search import run_clear_region_state_search
from strategies.relocating_state_search import run_relocating_state_search
from strategies.state_search import StateSearchConfig
from tests.test_strategy import ObservationOnlyClient


# These are fresh development seeds, not any earlier final evaluation set.
DEVELOPMENT_SEEDS = (220000, 220001)
FROZEN_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/state_search_candidate_relocating_cover_v1.json"
)
FROZEN_CONFIG = asdict(StateSearchConfig.parse(
    json.loads(FROZEN_CONFIG_PATH.read_text())["kwargs"]["config"]
))


@pytest.fixture(scope="module", params=DEVELOPMENT_SEEDS, ids=lambda seed: f"dev-{seed}")
def paired_runs(request):
    scenario = random_scenario(3, request.param)
    runs = {}
    for mode in ("original", "disabled", "nearest"):
        simulator = LocalResearchSimulator(scenario)
        client = simulator.client()
        restricted = ObservationOnlyClient(client)
        kwargs = {"config": FROZEN_CONFIG.copy(), "enabled": True,
                  "max_active_probes": 6}
        if mode == "original":
            report = run_relocating_state_search(restricted, **kwargs)
        else:
            report = run_clear_region_state_search(
                restricted, clear_mode=mode, history_silence=False, **kwargs
            )
        # Truth becomes available only after the policy has finished and exited.
        runs[mode] = (report, simulator, client)
    return runs


def assert_accepted_trace_and_cost_prefixes(report, simulator, client):
    """Recompute charged time and verify every reported action against its receipt."""
    actual = simulator.observation_history()
    assert actual[0]["action"] == "/enter"
    assert actual[-1]["action"] == "/exit"
    assert all(item["response"]["accepted"] is True for item in actual)
    assert actual[0]["response"]["virtual_time_s"] == 0
    assert client.state.session == "exited" and client.pending_request is None
    assert report.error is None and report.exit_error is None
    assert len(report.action_history) + 2 == len(actual) == report.accepted_actions

    previous, current_channel = Position(0, 0), 1
    charged_us = 0
    components_us = dict(movement_s=0, switching_s=0, detection_s=0,
                         optical_s=0, removal_s=0)
    for item, receipt in zip(report.action_history, actual[1:-1]):
        assert receipt["action"] == "/" + item["action"]
        assert receipt["position"] == dict(zip(("x", "y"), item["position"]))
        assert receipt["channel"] == item["channel"]
        response = receipt["response"]
        result_key = "measure_result" if item["action"] == "measure" else "clear_result"
        assert item["result"] == response[result_key]
        if item["result"] == "direction":
            assert item["bearing_deg"] == response["svd_deg"]

        point = Position(*item["position"])
        costs = {"movement_s": round(previous.distance_to(point) / 5 * 1_000_000)}
        if item["action"] == "measure":
            costs.update(detection_s=5_000_000,
                         switching_s=int(current_channel != item["channel"]) * 1_000_000)
            current_channel = item["channel"]
        else:
            costs.update(optical_s=3_000_000,
                         removal_s=int(item["result"] == "success") * 2_000_000)
        for name, amount in costs.items():
            components_us[name] += amount
        charged_us += sum(costs.values())
        assert item["virtual_time_s"] == response["virtual_time_s"] == charged_us / 1_000_000
        previous = point

    evaluation = simulator.evaluation()
    assert report.virtual_time_s == evaluation["virtual_time_s"] == charged_us / 1_000_000
    assert actual[-1]["response"]["virtual_time_s"] == report.virtual_time_s
    assert report.measurement_count == evaluation["measurement_count"]
    for name, amount in components_us.items():
        assert evaluation["time_breakdown_s"][name] == amount / 1_000_000
        # Client component estimates retain sub-microsecond movement precision;
        # the independent physical engine rounds each submitted movement.
        tolerance = len(report.action_history) * 0.5e-6 + 1e-9 if name == "movement_s" else 0
        assert math.isclose(report.time_breakdown[name], amount / 1_000_000,
                            rel_tol=0, abs_tol=tolerance)


def test_disabled_clearance_refinement_exactly_preserves_frozen_relocation(paired_runs):
    original, original_simulator, original_client = paired_runs["original"]
    disabled, disabled_simulator, disabled_client = paired_runs["disabled"]
    assert original.action_history == disabled.action_history
    assert original.virtual_time_s == disabled.virtual_time_s
    assert original.time_breakdown == disabled.time_breakdown
    assert original.accepted_actions == disabled.accepted_actions
    assert original.completion_reason == disabled.completion_reason
    assert disabled.strategy_parameters["clear_region_log"] == []
    assert_accepted_trace_and_cost_prefixes(original, original_simulator, original_client)
    assert_accepted_trace_and_cost_prefixes(disabled, disabled_simulator, disabled_client)


def test_nearest_clearance_is_certified_by_real_observation_prefix(paired_runs):
    report, simulator, client = paired_runs["nearest"]
    assert_accepted_trace_and_cost_prefixes(report, simulator, client)
    evaluation = simulator.evaluation()
    assert evaluation["all_cleared"] and evaluation["failed_clear_count"] == 0
    assert report.completion_certified_under_model
    assert report.cleared_channels == evaluation["cleared_channels"]
    assert report.unresolved_channels == []

    # Deliberately do not read source_estimates, clear_region_log or inferred
    # constraints. Each safety witness uses only earlier physical responses.
    regions, near = {}, {}
    certified_count = 0
    for item in report.action_history:
        channel, point = item["channel"], Position(*item["position"])
        if item["action"] == "measure":
            region = regions.setdefault(channel, OmniCandidateRegion())
            if item["result"] == "direction":
                region.observe(point, item["bearing_deg"])
            elif item["result"] == "no_signal":
                region.observe_no_signal(point)
            elif item["result"] == "near":
                near[channel] = point
        elif item["phase"] == "certified_clear":
            region = regions[channel]
            assert region.vertices and region.observations
            worst = max(point.distance_to(Position(*vertex)) for vertex in region.vertices)
            assert worst <= 19.9 + 1e-7
            assert item["result"] == "success"
            certified_count += 1
        elif item["phase"] == "near_clear":
            assert channel in near
            assert point.distance_to(near[channel]) + 5 <= 19.9 + 1e-7
            assert item["result"] == "success"
            certified_count += 1
    assert certified_count == report.clear_attempt_count == evaluation["cleared_total"]

"""Artificial observation archives; no scenarios, simulator, database or HTTP."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from types import SimpleNamespace

import pytest

from experiments.round2_derived_silence_audit import (
    exact_relative_certificate, observation_audit,
)
from experiments.round2_posthoc_audit import load_helpers


@pytest.fixture(scope="module")
def legacy():
    return load_helpers()[1]


def empty_record():
    return {"history": [{"index": 0, "action": "/enter", "channel": None,
                         "position": {"x": 0., "y": 0.},
                         "response": {"accepted": True, "virtual_time_s": 0.}}],
            "summary": {"action_history": [], "completion_certified_under_model": False,
                        "coverage_points_visited": 0, "strategy_parameters": {
                            "derived_silence_config": {"enabled": True, "relative_silence": True,
                                                       "stop_scan_at_16": True},
                            "derived_scan_audit": [], "inferred_no_signal_constraints": []}}}


def add_action(record, channel, position, result, *, bearing=None, phase="coverage", action="measure"):
    ordinal = len(record["summary"]["action_history"]) + 1
    response = {"accepted": True, "virtual_time_s": float(ordinal*5),
                "measure_result" if action == "measure" else "clear_result": result}
    report = {"action": action, "channel": channel, "position": list(position), "result": result,
              "virtual_time_s": response["virtual_time_s"], "phase": phase}
    if bearing is not None:
        response["svd_deg"] = report["bearing_deg"] = bearing
    record["history"].append({"index": ordinal, "action": "/"+action, "channel": channel,
                              "position": dict(zip(("x", "y"), position)), "response": response})
    record["summary"]["action_history"].append(report)


def scan_log(position, start, end, known, *, measured=(), relative=(), old=(), clear=(), count=(), credited=False):
    return {"position": list(position), "after_actual_action_count_before": start,
            "after_actual_action_count_after": end, "known_channels_before": list(known),
            "known_channels_after": list(known), "actual_measured_channels": list(measured),
            "relative_silence_skipped": list(relative), "old_silence_skipped": list(old),
            "clear_certified_skipped": list(clear), "count_skipped": list(count),
            "completed": True, "certified_by_count": len(known) == 16,
            "credited_geometric_station": credited, "physical_action_count": end-start}


def relative_record(legacy):
    record = empty_record()
    n, q = (1100., 0.), (1200., 0.)
    add_action(record, 1, n, "no_signal", phase="prior")
    add_action(record, 1, (100., 0.), "direction", bearing=180., phase="prior")
    region = legacy.OmniCandidateRegion()
    region.observe_no_signal(n)
    region.observe((100., 0.), 180.)
    lower, _ = exact_relative_certificate(region.vertices, q, n)
    # Independent conservative enclosure, deliberately far from tight rounding.
    claimed = float(lower)-1.
    upper = math.nextafter(math.dist(q, n), math.inf)
    required = math.nextafter(1e-5*upper, math.inf)
    event = {"inference_kind": "inferred_no_signal", "method": "relative_actual_negative",
             "channel": 1, "position": list(q), "physical_measurement": False,
             "after_actual_action_count": 2, "virtual_time_s": 10.,
             "witness_action_ordinal": 1, "witness_kind": "prior_actual_measure_no_signal",
             "actual_negative_position": list(n), "witness_virtual_time_s": 5.,
             "outer_region_vertex_count": len(region.vertices), "margin_m": 1e-5,
             "affine_lower_bound_m2": claimed, "required_affine_margin_m2": required,
             "query_negative_distance_upper_m": upper,
             "signed_bisector_distance_lower_m": math.nextafter(claimed/upper, -math.inf)}
    parameters = record["summary"]["strategy_parameters"]
    parameters["inferred_no_signal_constraints"] = [event]
    for channel in range(2, 21):
        add_action(record, channel, q, "no_signal")
    parameters["derived_scan_audit"] = [scan_log(q, 2, 21, [1], measured=range(2, 21),
                                                relative=[1], credited=True)]
    record["summary"]["coverage_points_visited"] = 1
    return record


def cap_record():
    record = empty_record()
    for channel in range(1, 17):
        add_action(record, channel, (0., 0.), "near", phase="prior")
    known = list(range(1, 17))
    count = [{"channel": c, "after_actual_action_count": 16, "known_channels": known.copy()}
             for c in range(17, 21)]
    record["summary"]["strategy_parameters"]["derived_scan_audit"] = [scan_log(
        (1200., 0.), 16, 16, known, clear=[16, *range(1, 16)], count=count)]
    return record


def test_exact_fraction_certificate_and_strict_boundary():
    square = [(-10., -10.), (10., -10.), (10., 10.), (-10., 10.)]
    lower, norm2 = exact_relative_certificate(square, (1200., 0.), (1100., 0.))
    assert lower == Fraction(114000) and norm2 == 10000
    for bad in ((1100., 0.), (1000., 0.)):
        with pytest.raises(ValueError):
            exact_relative_certificate(square, bad, (1100., 0.))
    with pytest.raises(ValueError, match="margin"):
        exact_relative_certificate([(0., 0.)], (2e-5, 0.), (0., 0.))


@pytest.mark.parametrize("vertices", [[(0., 0.)], [(0., 0.), (1., 1.)],
                                      [(1e12, 1e12), (1e12+1, 1e12+1)]])
def test_exact_certificate_accepts_degenerate_and_large_binary64_inputs(vertices):
    shift = vertices[0][0]
    exact_relative_certificate(vertices, (shift+1200., shift), (shift+1100., shift))


def test_valid_relative_inference_passes_and_never_creates_real_absence_credit(legacy):
    record = relative_record(legacy)
    original = deepcopy(record)
    result = observation_audit(record, {}, legacy)
    assert result["relative_inferred_silence_verified"] == 1
    assert result["legacy_inferred_silence_verified"] == 0
    assert result["inferred_coverage_credits"] == result["relative_inferred_coverage_credits"] == 0
    assert result["absence"][0]["real_negative_measurements"] == 1
    assert result["derived_scan_totals"]["completed_geometric_scans"] == 1
    assert result["legacy_without_relative_constraints_passed"]
    assert not result["terminal_certified"]
    assert record == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value,error", [
    ("witness_action_ordinal", 3, "earlier actual"),
    ("witness_action_ordinal", True, "earlier actual"),
    ("witness_action_ordinal", 2, "same-channel"),
    ("witness_kind", "inferred_no_signal", "same-channel"),
    ("actual_negative_position", [1101., 0.], "position differs"),
    ("witness_virtual_time_s", 11., "Witness virtual time"),
    ("virtual_time_s", 11., "Inference virtual time"),
    ("physical_measurement", True, "claims a physical"),
    ("outer_region_vertex_count", 1234, "vertex count"),
    ("affine_lower_bound_m2", 1e30, "enclosure"),
    ("affine_lower_bound_m2", float("nan"), "Nonfinite"),
    ("required_affine_margin_m2", 0., "enclosure"),
    ("query_negative_distance_upper_m", 99., "enclosure"),
    ("signed_bisector_distance_lower_m", 1e20, "enclosure"),
    ("margin_m", 0., "safety margin"),
])
def test_relative_certificate_metadata_mutations_rejected(legacy, field, value, error):
    record = relative_record(legacy)
    record["summary"]["strategy_parameters"]["inferred_no_signal_constraints"][0][field] = value
    with pytest.raises(ValueError, match=error):
        observation_audit(record, {}, legacy)


def test_wrong_channel_real_negative_cannot_witness_relative_silence(legacy):
    record = relative_record(legacy)
    record["history"][1]["channel"] = record["summary"]["action_history"][0]["channel"] = 2
    with pytest.raises(ValueError, match="same-channel"):
        observation_audit(record, {}, legacy)


def test_query_on_wrong_side_is_rejected_by_geometry_not_candidate_claim(legacy):
    record = relative_record(legacy)
    parameters = record["summary"]["strategy_parameters"]
    parameters["inferred_no_signal_constraints"][0]["position"] = [1000., 0.]
    parameters["derived_scan_audit"][0]["position"] = [1000., 0.]
    for action, reported in zip(record["history"][3:], record["summary"]["action_history"][2:]):
        action["position"]["x"] = reported["position"][0] = 1000.
    with pytest.raises(ValueError, match="positive distance margin"):
        observation_audit(record, {}, legacy)


def test_new_certificate_cannot_be_disguised_as_old_1500_rule(legacy):
    record = relative_record(legacy)
    parameters = record["summary"]["strategy_parameters"]
    parameters["inferred_no_signal_constraints"][0]["method"] = "polygon_edges"
    scan = parameters["derived_scan_audit"][0]
    scan["old_silence_skipped"], scan["relative_silence_skipped"] = [1], []
    with pytest.raises(ValueError, match=">1500m"):
        observation_audit(record, {}, legacy)


def test_valid_old_1500_inference_remains_in_legacy_replay(legacy):
    record = relative_record(legacy)
    parameters = record["summary"]["strategy_parameters"]
    parameters["inferred_no_signal_constraints"][0].update(method="polygon_edges", position=[2000., 0.])
    scan = parameters["derived_scan_audit"][0]
    scan.update(old_silence_skipped=[1], relative_silence_skipped=[], position=[2000., 0.])
    for action, report in zip(record["history"][3:], record["summary"]["action_history"][2:]):
        action["position"]["x"] = report["position"][0] = 2000.
    result = observation_audit(record, {}, legacy)
    assert result["legacy_inferred_silence_verified"] == result["inferred_silence_verified"] == 1
    assert result["relative_inferred_silence_verified"] == 0
    assert result["absence"][0]["real_negative_measurements"] == 1


def test_negative_derived_point_cannot_replace_the_real_witness(legacy):
    record = relative_record(legacy)
    event = record["summary"]["strategy_parameters"]["inferred_no_signal_constraints"][0]
    event["actual_negative_position"] = event["position"].copy()
    with pytest.raises(ValueError, match="witness position"):
        observation_audit(record, {}, legacy)


def test_legacy_failure_is_never_overridden_by_new_certificate(legacy):
    def reject(record, cache):
        assert record["summary"]["strategy_parameters"]["inferred_no_signal_constraints"] == []
        raise ValueError("legacy refuses real clear/absence proof")
    wrapped = SimpleNamespace(OmniCandidateRegion=legacy.OmniCandidateRegion,
                              polygon_distance=legacy.polygon_distance, observation_audit=reject)
    with pytest.raises(ValueError, match="legacy refuses"):
        observation_audit(relative_record(legacy), {}, wrapped)


def test_public_16_skip_is_prefix_justified_and_all_skipped_scan_does_not_move(legacy):
    record = cap_record()
    result = observation_audit(record, {}, legacy)
    assert result["cleared"] == 0 and result["detected"] == 16
    assert not result["terminal_certified"]
    assert result["derived_scan_totals"]["count_cap_skips"] == 4
    assert result["derived_scan_totals"]["no_physical_action_scans"] == 1
    assert result["derived_scan_checks"][0]["physical_action_count"] == 0
    assert result["derived_scan_totals"]["completed_geometric_scans"] == 0


def test_16th_channel_discovered_during_scan_allows_only_later_unknown_skips(legacy):
    record = empty_record()
    for channel in range(1, 16):
        add_action(record, channel, (0., 0.), "near", phase="prior")
    add_action(record, 16, (1200., 0.), "near")
    known = list(range(1, 17))
    skipped = [{"channel": c, "after_actual_action_count": 16, "known_channels": known.copy()}
               for c in range(17, 21)]
    scan = scan_log((1200., 0.), 15, 16, list(range(1, 16)), measured=[16],
                    clear=[15, *range(1, 15)], count=skipped)
    scan.update(known_channels_after=known, certified_by_count=True)
    record["summary"]["strategy_parameters"]["derived_scan_audit"] = [scan]
    result = observation_audit(record, {}, legacy)
    assert result["derived_scan_checks"][0]["physical_action_count"] == 1
    assert result["derived_scan_checks"][0]["certified_by_count"]
    assert not result["derived_scan_checks"][0]["credited_geometric_station"]


def test_interrupted_scan_has_only_a_traversal_prefix_and_no_station_credit(legacy):
    record = empty_record()
    add_action(record, 1, (100., 0.), "no_signal")
    scan = scan_log((100., 0.), 0, 1, [], measured=[1])
    scan["completed"] = False
    record["summary"]["strategy_parameters"]["derived_scan_audit"] = [scan]
    result = observation_audit(record, {}, legacy)
    assert result["physical_actions"] == 1
    assert result["derived_scan_totals"]["completed_geometric_scans"] == 0
    scan["actual_measured_channels"] = [2]
    with pytest.raises(ValueError, match="traversal"):
        observation_audit(record, {}, legacy)


def test_actual_prefix_region_is_not_replaced_with_archive_ground_truth(legacy):
    class Unreadable:
        def __deepcopy__(self, memo):
            raise AssertionError("Hidden payload must not be traversed")
    record = relative_record(legacy)
    record["ground_truth"] = Unreadable()
    record["scenario"] = Unreadable()
    assert observation_audit(record, {}, legacy)["relative_inferred_silence_verified"] == 1


@pytest.mark.parametrize("mutation,error", [
    ("visit", "Coverage visit"), ("credit", "unearned geometric"),
    ("physical", "physical action count"), ("relocation", "unexecuted/incomplete"),
    ("future_count", "Count skip"), ("known", "known-channel list"),
    ("measured", "multiple action/skip"), ("missing_scan", "missing its scan"),
])
def test_fake_physical_or_scan_cap_logs_fail_closed(legacy, mutation, error):
    record = relative_record(legacy) if mutation == "missing_scan" else cap_record()
    params = record["summary"]["strategy_parameters"]
    scan = params["derived_scan_audit"][0]
    if mutation == "visit":
        record["summary"]["coverage_points_visited"] = 1
    elif mutation == "credit":
        scan["credited_geometric_station"] = True
    elif mutation == "physical":
        scan["physical_action_count"] = 1
    elif mutation == "relocation":
        params["relocation_log"] = [{"after_actual_action_count": 16,
                                     "executed_discovery_stations": [[1200., 0.]]}]
    elif mutation == "future_count":
        scan["count_skipped"][0]["after_actual_action_count"] = 15
    elif mutation == "known":
        scan["known_channels_before"] = list(range(1, 16))
    elif mutation == "measured":
        scan["actual_measured_channels"] = [16]
    else:
        params["derived_scan_audit"] = []
    with pytest.raises(ValueError, match=error):
        observation_audit(record, {}, legacy)


def test_later_16th_detection_does_not_justify_earlier_count_skips(legacy):
    record = cap_record()
    record["history"][16]["response"]["measure_result"] = "no_signal"
    record["summary"]["action_history"][15]["result"] = "no_signal"
    add_action(record, 16, (0., 0.), "near", phase="later")
    scan = record["summary"]["strategy_parameters"]["derived_scan_audit"][0]
    scan["known_channels_before"] = scan["known_channels_after"] = list(range(1, 16))
    scan["clear_certified_skipped"] = list(range(1, 16))
    scan["count_skipped"].insert(0, {"channel": 16, "after_actual_action_count": 16,
                                      "known_channels": list(range(1, 17))})
    with pytest.raises(ValueError, match="16 distinct actual known"):
        observation_audit(record, {}, legacy)


def test_clear_certified_skip_needs_prior_near_or_mec(legacy):
    record = cap_record()
    record["history"][1]["response"].update(measure_result="direction", svd_deg=0.)
    record["summary"]["action_history"][0].update(result="direction", bearing_deg=0.)
    with pytest.raises(ValueError, match="near/MEC"):
        observation_audit(record, {}, legacy)


def test_real_physical_row_cannot_explicitly_claim_it_was_inferred(legacy):
    record = cap_record()
    record["history"][1]["physical_measurement"] = False
    with pytest.raises(ValueError, match="inferred physical"):
        observation_audit(record, {}, legacy)


def test_disabled_identity_runs_original_auditor_without_new_logs(legacy):
    record = empty_record()
    record["summary"]["strategy_parameters"]["derived_silence_config"]["enabled"] = False
    add_action(record, 1, (0., 0.), "near")
    add_action(record, 1, (0., 0.), "success", action="clear", phase="near_clear")
    result = observation_audit(record, {}, legacy)
    expected = legacy.observation_audit(record, {})
    assert all(result[key] == value for key, value in expected.items())
    assert result["relative_inferred_silence_verified"] == 0


def test_clear_does_not_retune_before_the_next_physical_scan(legacy):
    record = empty_record()
    add_action(record, 1, (0., 0.), "near", phase="prior")
    add_action(record, 5, (0., 0.), "direction", bearing=0., phase="prior")
    add_action(record, 1, (0., 0.), "success", action="clear", phase="near_clear")
    order = [5, 2, 3, 4, *range(6, 21)]
    for channel in order:
        add_action(record, channel, (0., 0.), "direction" if channel == 5 else "no_signal",
                   bearing=0. if channel == 5 else None)
    scan = scan_log((0., 0.), 3, 22, [1, 5], measured=order, credited=True)
    record["summary"]["strategy_parameters"]["derived_scan_audit"] = [scan]
    record["summary"]["coverage_points_visited"] = 1
    result = observation_audit(record, {}, legacy)
    assert result["derived_scan_checks"][0]["physical_action_count"] == 19
    assert result["derived_scan_checks"][0]["credited_geometric_station"]

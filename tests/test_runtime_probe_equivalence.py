"""Differential checks against committed v1, without new performance worlds."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType

import pytest

from localization.omni import OmniCandidateRegion
from planning.probe_candidates import geometry_candidates
from planning.radius_probe import ProbeGeometryMemo, choose_radius_probe
from simulation.engine import MemoryClient
from simulator_client.state import Position
from strategies.refined_state_search import RefinedStateSearch


ROOT = Path(__file__).resolve().parents[1]
FROZEN = "760af8230a8e5bda6050663b19d871a9701c9829"


def committed_module(path, name):
    # The real archived implementation is the oracle, not a mirrored scoring
    # formula. git show reads local committed bytes only; it never fetches.
    source = subprocess.check_output(["git", "show", f"{FROZEN}:{path}"], cwd=ROOT)
    module = ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    module.frozen_sha256 = hashlib.sha256(source).hexdigest()
    exec(compile(source, f"{FROZEN}:{path}", "exec"), module.__dict__)
    return module


ORIGINAL = committed_module("src/planning/radius_probe.py", "planning._original_runtime_probe")
OLD_REFINED = committed_module("src/strategies/refined_state_search.py", "strategies._original_runtime_refined")
OLD_REFINED.choose_radius_probe = ORIGINAL.choose_radius_probe
BASE = json.loads((ROOT / "experiments/state_search_candidate_derived_silence_combined_v1.json").read_text())["kwargs"]["config"]


def without_runtime(log):
    if isinstance(log, dict):
        return {k: without_runtime(v) for k, v in log.items() if k not in {"runtime_s", "total_runtime_s", "program_runtime_s"}}
    if isinstance(log, list):
        return [without_runtime(v) for v in log]
    return log


def public_region(kind):
    region = OmniCandidateRegion()
    if kind == "negative_only":
        return region.observe_no_signal((1600., 0.))
    if kind == "wrapped":
        return region.observe((0., 0.), 359.999).observe_no_signal((1600., 0.))
    if kind == "negative":
        return region.observe((0., 0.), 0.).observe_no_signal((1600., 0.))
    if kind == "extreme":
        return region.observe((-1800., -0.), 0.).observe_no_signal((-100., 0.))
    if kind == "tight":
        region.observe((0., 0.), 0.)
        region.observe((700., -500.), math.degrees(math.atan2(500., 100.)))
        return region
    return region.observe((0., 0.), 0.)


@pytest.mark.parametrize("kind", ["positive", "negative", "negative_only", "wrapped", "extreme", "tight"])
@pytest.mark.parametrize("weight", [0., .5, 3.])
def test_two_stages_preserve_every_score_and_search_counter(kind, weight):
    region = public_region(kind)
    current = Position(-1800., -0.) if kind == "extreme" else Position(0., 0.)
    first = 359.999 if kind == "wrapped" else 0.
    observed = {(round(current.x, 6), round(current.y, 6))}
    memo = ProbeGeometryMemo()
    old_point, old_log = ORIGINAL.choose_radius_probe(region, current, first, observed, weight)
    actual, actual_log = choose_radius_probe(region, current, first, observed, weight, geometry_memo=memo)
    assert actual == old_point
    assert without_runtime(actual_log) == without_runtime(old_log)
    if old_point is not None:
        extra, _ = geometry_candidates(region, mode="axis_quantile", old_best=old_point)
        extra += extra[:3] + [old_point, old_point]  # exact duplicate rejection
        expected = ORIGINAL.choose_radius_probe(region, current, first, observed, weight, extra_points=extra)
        actual = choose_radius_probe(region, current, first, observed, weight, extra_points=extra, geometry_memo=memo)
        assert actual[0] == expected[0]
        assert without_runtime(actual[1]) == without_runtime(expected[1])


def test_empty_input_preserves_original_exception():
    region = public_region("positive").observe_no_signal((0., 0.))
    assert not region.vertices
    for function, kwargs in ((ORIGINAL.choose_radius_probe, {}),
                             (choose_radius_probe, {"geometry_memo": ProbeGeometryMemo()})):
        with pytest.raises(ValueError, match="empty sets"):
            function(region, Position(0., 0.), 0., set(), **kwargs)


def test_empty_posterior_is_cached_without_changing_infinite_tie(monkeypatch):
    region = public_region("positive")
    original_copy = OmniCandidateRegion.copy
    calls = []
    def empty_copy(self):
        calls.append(1)
        copy = original_copy(self)
        def empty_observe(*_):
            copy.vertices = ()
            copy._circle = None
            return copy
        copy.observe = empty_observe
        return copy
    monkeypatch.setattr(OmniCandidateRegion, "copy", empty_copy)
    expected = ORIGINAL.choose_radius_probe(region, Position(0., 0.), 0., set())
    memo = ProbeGeometryMemo()
    actual = choose_radius_probe(region, Position(0., 0.), 0., set(), geometry_memo=memo)
    before = len(calls)
    again = choose_radius_probe(region, Position(0., 0.), 0., set(), geometry_memo=memo)
    assert len(calls) == before
    assert expected[0] == actual[0] == again[0]
    assert without_runtime(expected[1]) == without_runtime(actual[1]) == without_runtime(again[1])
    assert set(memo._values.values()) == {None}


@pytest.mark.parametrize("change", ["copy", "vertices", "positive", "negative", "error", "reception", "negative_only"])
def test_reuse_after_region_change_cannot_retain_stale_posterior(change):
    region = public_region("negative_only" if change == "negative_only" else "positive")
    memo = ProbeGeometryMemo()
    choose_radius_probe(region, Position(0., 0.), 0., set(), geometry_memo=memo)
    old_signature = memo._signature
    memo._values[("impossible",)] = 999.
    if change == "copy":
        region = region.copy()
    elif change == "vertices":
        region.vertices = region.vertices[1:]
        region._circle = None
    elif change == "positive":
        region.observe((0., -200.), 20.)
    elif change in {"negative", "negative_only"}:
        region.observe_no_signal((1600., 300.))
    elif change == "error":
        region.error_deg = 1.1
    else:
        region.reception_radius = 1450.
    expected = ORIGINAL.choose_radius_probe(region, Position(0., 0.), 0., set())
    actual = choose_radius_probe(region, Position(0., 0.), 0., set(), geometry_memo=memo)
    assert ("impossible",) not in memo._values
    assert change == "copy" or memo._signature != old_signature
    assert actual[0] == expected[0]
    assert without_runtime(actual[1]) == without_runtime(expected[1])


def test_cache_keys_preserve_signed_zero_and_source_is_unchanged():
    region = public_region("positive")
    region.enclosing_disk()
    before = deepcopy(region.__dict__)
    memo = ProbeGeometryMemo()
    memo.bind(region)
    memo.posterior_radius(Position(0., 0.), 0.)
    memo.posterior_radius(Position(-0., 0.), 0.)
    memo.posterior_radius(Position(0., 0.), -0.)
    assert len(memo._values) == 3
    assert all(v is None or type(v) is float for v in memo._values.values())
    assert region.__dict__ == before


def test_second_stage_performs_strictly_fewer_real_geometry_updates(monkeypatch):
    region = public_region("positive")
    original = OmniCandidateRegion.copy
    calls = []
    def count(self):
        calls.append(1)
        return original(self)
    monkeypatch.setattr(OmniCandidateRegion, "copy", count)
    old, _ = ORIGINAL.choose_radius_probe(region, Position(0., 0.), 0., set())
    extra, _ = geometry_candidates(region, mode="axis_quantile", old_best=old)
    expected = ORIGINAL.choose_radius_probe(region, Position(0., 0.), 0., set(), extra_points=extra)
    uncached_count = len(calls)
    calls.clear()
    memo = ProbeGeometryMemo()
    choose_radius_probe(region, Position(0., 0.), 0., set(), geometry_memo=memo)
    actual = choose_radius_probe(region, Position(0., 0.), 0., set(), extra_points=extra, geometry_memo=memo)
    assert 0 < len(calls) < uncached_count
    assert without_runtime(actual[1]) == without_runtime(expected[1])


class PhysicalFeedback:
    """One hand-constructed source, used only by the protocol fixture."""
    def __init__(self):
        self.position, self.channel, self.us = (0., 0.), 1, 0

    def __call__(self, path, fields):
        response = {"accepted": True, "real_timestamp_ms": 1, "virtual_time_s": self.us / 1e6}
        if path == "/enter":
            return response | {"max_virtual_duration_s": 360000., "max_real_duration_s": 1200., "remaining_real_duration_s": 1200.}
        if path == "/exit":
            return response | {"exit_reason": "user_exit"}
        p = fields["position"]
        point = (p["x"], p["y"])
        self.us += round(math.dist(self.position, point) / 5 * 1e6) + 5000000
        self.position = point
        distance = math.dist(point, (800., 40.))
        if path == "/measure":
            self.us += 1000000 * (self.channel != fields["channel"])
            self.channel = fields["channel"]
            result = "no_signal" if distance > 1000. else "near" if distance <= 5. else "direction"
            response["measure_result"] = result
            if result == "direction":
                response["svd_deg"] = math.degrees(math.atan2(40. - point[1], 800. - point[0])) % 360
        else:
            response["clear_result"] = "success" if distance <= 20. else "failure"
        response["virtual_time_s"] = self.us / 1e6
        return response


def test_real_feedback_probe_and_clear_match_frozen_resolver():
    reports = []
    for cls in (OLD_REFINED.RefinedStateSearch, RefinedStateSearch):
        client = MemoryClient(PhysicalFeedback(), clock=lambda: 0.)
        client.enter()
        search = cls(client, 10000, 6, BASE, "axis_quantile")
        search._perform("measure", Position(0., 0.), 1, "fixture_first_actual_bearing")
        assert search._resolve(1)
        assert search.report.measurement_count >= 2
        reports.append((search.report.action_history, without_runtime(search.probe_log),
                        client.state.virtual_time_s, search.regions[1].vertices))
    assert reports[0] == reports[1]

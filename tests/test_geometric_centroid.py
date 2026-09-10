import math

import pytest

from localization.omni import OmniCandidateRegion
from simulation import LocalResearchSimulator, random_scenario
from strategies.efficient import EfficientSearch
from strategies.geometric_centroid import GeometricCentroidSearch, run_centroid_search, safe_centroid_point
from tests.test_strategy import ObservationOnlyClient


def test_area_centroid_segment_is_clamped_and_independently_safe():
    region = OmniCandidateRegion()
    region.vertices = ((0., 0.), (1500., -30.), (1500., 30.))
    result = safe_centroid_point(region)
    assert result['raw_centroid'] == pytest.approx((1000., 0.))
    assert 0 < result['segment_fraction'] < 1
    assert all(math.dist(result['position'], v) <= 1000.-1e-6 for v in region.vertices)
    assert result['position'][1] == 0
    assert result['position'][0] > result['mec_center'][0]


def test_first_probe_changes_while_scheduling_destination_stays_original():
    case = random_scenario(3, 103001)
    baseline = EfficientSearch(LocalResearchSimulator(case).client(), 10000, 6, None)
    candidate = GeometricCentroidSearch(LocalResearchSimulator(case).client(), 10000, 6)
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    for search in (baseline, candidate):
        search.regions[1] = region.copy()
        search.first_bearings[1] = 0.
    assert baseline._target(1) == candidate._target(1)
    assert baseline._next_probe(1, 0) != candidate._next_probe(1, 0)
    assert baseline._next_probe(1, 1) == candidate._next_probe(1, 1)


def test_new_first_probe_retains_complete_zero_failure_legal_search():
    sim = LocalResearchSimulator(random_scenario(3, 103001))
    report = run_centroid_search(ObservationOnlyClient(sim.client()))
    assert sim.observation_history()[-1]['action'] == '/exit'
    evaluation = sim.evaluation()
    assert evaluation['all_cleared'] and report.completion_certified_under_model
    assert evaluation['failed_clear_count'] == 0
    assert report.virtual_time_s == evaluation['virtual_time_s']
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s, abs=1e-4)
    assert report.centroid_first_probes
    assert all(p['maximum_vertex_distance_m'] <= 1000.-1e-6 for p in report.centroid_first_probes)


def test_q4_rejected_before_enter():
    sim = LocalResearchSimulator(random_scenario(3, 103001))
    with pytest.raises(ValueError, match='Q3-only'):
        run_centroid_search(ObservationOnlyClient(sim.client()), problem=4)
    assert sim.observation_history() == []

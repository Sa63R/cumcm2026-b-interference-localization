"""Public numeric fixtures only; no scenarios, simulator or private data."""
import math

import numpy as np
import pytest

from q4_rl import negative_memory as module
from q4_rl.negative_memory import NegativeObservationMemory, fixed_bank, visibility


def observe(memory, position, channel=1, result="no_signal", accepted=True, action="measure"):
    return memory.observe(action=action, position=position, channel=channel,
                          result=result, accepted=accepted)


def bank_index(x=0., y=0., radius=1000., heading=0., omni=False):
    rows = fixed_bank()
    angle = math.radians(heading)
    mask = ((rows[:, 0] == x) & (rows[:, 1] == y) & (rows[:, 2] == radius)
            & np.isclose(rows[:, 3], math.cos(angle), atol=1e-14)
            & np.isclose(rows[:, 4], math.sin(angle), atol=1e-14)
            & (rows[:, 5] == int(omni)))
    indices = np.flatnonzero(mask)
    assert len(indices) == 1
    return indices[0]


def test_bank_fixed_physical_support_is_small_and_immutable():
    bank = fixed_bank()
    assert bank.shape == (3051, 6)
    assert np.unique(bank[:, :2], axis=0).shape == (113, 2)
    assert set(bank[:, 2]) == {1000., 1250., 1500.}
    assert np.hypot(bank[:, 0], bank[:, 1]).max() == 1800.
    with pytest.raises(ValueError):
        bank[0, 0] = 99.


@pytest.mark.parametrize("point", [(0., 0.), (1000., 0.), (1500., 0.), (0., 1000.),
                                  (-1., 0.), (1800., -300.), (2800., 0.)])
def test_numpy_visibility_matches_scalar_public_physics(point):
    expected = []
    for x, y, radius, cosine, sine, omni in fixed_bank():
        dx, dy = point[0]-x, point[1]-y
        distance = math.hypot(dx, dy)
        expected.append(distance <= radius and
                        (omni == 1 or cosine*dx+sine*dy >= -1e-12*max(1., distance)))
    np.testing.assert_array_equal(visibility(point), expected)


def test_closed_radius_sector_boundary_heading_wrap_and_near_backside():
    east = bank_index()
    assert visibility((1000., 0.))[east]
    assert not visibility((1000.+1e-6, 0.))[east]
    assert visibility((0., 1000.))[east]  # Included 90-degree sector edge.
    assert not visibility((-1., 0.))[east]  # <=5m does not override direction.
    southeast = bank_index(heading=315.)
    assert visibility((100., 0.))[southeast]  # Sector wraps through 0 degrees.
    assert visibility((2800., 0.))[bank_index(x=1800.)]
    memory = NegativeObservationMemory()
    observe(memory, (-1., 0.))
    assert memory.compatible[0, east]
    assert not memory.compatible[0, bank_index(omni=True)]


def test_only_accepted_real_negative_measurements_filter_and_channels_are_isolated():
    memory = NegativeObservationMemory()
    before = memory.compatible.copy()
    assert not observe(memory, (0., 0.), accepted=False)
    assert not memory.history
    observe(memory, (0., 0.), result="no_target_in_range", action="clear")
    observe(memory, (0., 0.), result="direction", channel=2)
    np.testing.assert_array_equal(memory.compatible, before)
    observe(memory, (100., 0.))
    np.testing.assert_array_equal(memory.compatible[0], ~visibility((100., 0.)))
    np.testing.assert_array_equal(memory.compatible[1:], before[1:])
    assert not memory.score_candidates([(2, (0., 0.))])[0, 7]
    with pytest.raises(TypeError):
        memory.observe(action="measure", position=(0., 0.), channel=1,
                       result="no_signal", accepted=True, case_seed=8000000)


def test_repetition_has_no_extra_mass_or_count_and_replay_is_exact():
    memory = NegativeObservationMemory()
    observe(memory, (0., 0.))
    before = memory.compatible.copy()
    assert not observe(memory, (0., 0.))
    np.testing.assert_array_equal(memory.compatible, before)
    assert len(memory.negative_points[0]) == 1
    observe(memory, (1000., 0.))
    observe(memory, (0., 1000.), channel=2)
    observe(memory, (0., 0.), channel=3, result="near")
    queries = [(1, (0., 0.)), (1, (-1000., 0.)), (2, (100., 200.)), (3, (0., 0.))]
    replayed = NegativeObservationMemory.replay(memory.history)
    np.testing.assert_array_equal(replayed.compatible, memory.compatible)
    np.testing.assert_array_equal(replayed.score_candidates(queries), memory.score_candidates(queries))
    assert memory.score_candidates([(1, (0., 0.))])[0, 1] == 0.


def test_negative_spatial_history_changes_scores_with_equal_observation_counts():
    left, right = NegativeObservationMemory(), NegativeObservationMemory()
    observe(left, (-900., 0.))
    observe(right, (900., 0.))
    query = [(1, (-900., 0.))]
    a, b = left.score_candidates(query)[0], right.score_candidates(query)[0]
    assert a[4] == b[4]
    assert a[1] == 0. and b[1] > 0.
    assert a[5] == 0. and b[5] > 0.


def test_empty_existence_library_is_not_channel_absence_proof():
    memory = NegativeObservationMemory()
    for point in np.unique(fixed_bank()[:, :2], axis=0):
        observe(memory, point)
    assert not memory.compatible[0].any()
    features = memory.score_candidates([(1, (0., 0.))])[0]
    np.testing.assert_array_equal(features[:4], [0., 0., 0., 1.])
    assert features[7] == 1.  # Absence remains possible, not established.
    assert np.isfinite(features).all()
    assert not hasattr(memory, "completion_certified")


def test_clear_success_retires_channel_without_learning_from_post_clear_silence():
    memory = NegativeObservationMemory()
    observe(memory, (0., 0.), action="clear", result="success")
    before = memory.compatible.copy()
    assert not observe(memory, (0., 0.))
    np.testing.assert_array_equal(memory.compatible, before)
    assert not memory.negative_points[0]
    with pytest.raises(ValueError, match="live channels"):
        memory.score_candidates([(1, (0., 0.))])


def test_candidate_order_chunking_and_bounded_cache_do_not_change_scores(monkeypatch):
    memory = NegativeObservationMemory()
    for c in range(1, 21):
        observe(memory, (c*40., -c*20.), channel=c)
    queries = [(i % 20+1, (float(i*13), float(i*-7))) for i in range(160)]
    expected = memory.score_candidates(queries)
    monkeypatch.setattr(module, "SCORE_CHUNK_SIZE", 7)
    np.testing.assert_array_equal(memory.score_candidates(list(reversed(queries)))[::-1], expected)
    assert len(memory._visibility_cache) <= module.CACHE_POINTS
    assert np.isfinite(expected).all() and (expected >= 0.).all() and (expected <= 1.).all()
    assert memory.score_candidates([]).shape == (0, module.FEATURE_DIM)


def test_existing_micro_feature_schema_remains_unchanged():
    from q4_rl import micro_controller
    assert micro_controller.FEATURE_SCHEMA_VERSION == "q4-micro-g1-v1"
    assert (micro_controller.GLOBAL_DIM, micro_controller.CANDIDATE_DIM) == (13, 50)

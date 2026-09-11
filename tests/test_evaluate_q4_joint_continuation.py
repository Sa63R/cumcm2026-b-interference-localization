"""Pure gate algebra, not generated or saved development/held-out cases."""
import pytest

from experiments.evaluate_q4_joint_continuation import candidate_passes, choose_candidate, CANDIDATES


def good():
    return dict(all_clear=True, mean_saved_s=10., saving_ci95_s=[1., 19.],
                mean_reduction_fraction=.005, p95_ratio=1.05)


@pytest.mark.parametrize("field,value", [("mean_saved_s", 0.), ("all_clear", False), ("p95_ratio", 1.050001)])
def test_development_needs_strict_random_saving_and_safety(field, value):
    random, stress = good(), good()
    random[field] = value
    assert not candidate_passes(random, stress, "development")


def test_negative_stress_mean_rejects_even_if_random_is_good():
    random, stress = good(), good()
    stress["mean_saved_s"] = -1e-9
    assert not candidate_passes(random, stress, "development")
    stress["mean_saved_s"] = 0.
    assert candidate_passes(random, stress, "development")


@pytest.mark.parametrize("field,value", [("saving_ci95_s", [0., 20.]), ("mean_reduction_fraction", .004999)])
def test_independent_requires_positive_interval_and_half_percent(field, value):
    random, stress = good(), good()
    random[field] = value
    assert candidate_passes(random, stress, "development")
    assert not candidate_passes(random, stress, "confirmation")


def test_five_second_tie_is_strict_and_cannot_select_failed_simple_arm():
    simple, dynamic = CANDIDATES
    assert choose_candidate(list(CANDIDATES), {simple: 100., dynamic: 95.1}) == simple
    assert choose_candidate(list(CANDIDATES), {simple: 100., dynamic: 95.}) == dynamic
    assert choose_candidate([dynamic], {simple: 90., dynamic: 95.1}) == dynamic
    assert choose_candidate([], {}) is None


def test_unknown_or_duplicated_candidate_rejected():
    with pytest.raises(ValueError):
        choose_candidate([CANDIDATES[0], CANDIDATES[0]], {CANDIDATES[0]: 100.})
    with pytest.raises(ValueError):
        choose_candidate(["unexpected"], {"unexpected": 100.})

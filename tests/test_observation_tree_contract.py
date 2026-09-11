"""Exact artificial oracles for the observation-tree review contract.

These test the numerical contract, NOT a candidate tree implementation.
No repository strategy, simulator, dataset, sampled scenario or external I/O.
An implementation adapter must later produce the same results independently.
"""

from fractions import Fraction as F

import pytest


PRIOR = (F(1, 2), F(1, 2))
COSTS = {"L": (F(0), F(10)), "R": (F(10), F(0))}


def posterior(prior, likelihood):
    weights = tuple(p * likelihood[i] for i, p in enumerate(prior))
    mass = sum(weights)
    if not mass:
        raise ValueError("zero-probability observation has no posterior")
    return mass, tuple(w / mass for w in weights)


def expected_cost(weights, action):
    return sum(weights[i] * COSTS[action][i] for i in range(2))


def choose_nonclairvoyant(weights):
    action = min(COSTS, key=lambda a: (expected_cost(weights, a), a))
    return action, expected_cost(weights, action)


def ess(weights):
    return sum(weights) ** 2 / sum(w * w for w in weights)


def test_same_observation_requires_one_shared_action_not_worldwise_minimum():
    _, weights = posterior(PRIOR, (F(1), F(1)))
    action, legal_value = choose_nonclairvoyant(weights)
    clairvoyant_value = sum(PRIOR[i] * min(c[i] for c in COSTS.values()) for i in range(2))
    assert action == "L" and legal_value == 5
    assert clairvoyant_value == 0 and clairvoyant_value != legal_value


def test_informative_two_child_tree_has_exact_value_three_not_one():
    likelihoods = ((F(4, 5), F(1, 5)), (F(1, 5), F(4, 5)))
    result, actions = F(1), []  # complete first macro costs exactly one
    for likelihood in likelihoods:
        mass, weights = posterior(PRIOR, likelihood)
        action, value = choose_nonclairvoyant(weights)
        actions.append(action)
        result += mass * value
    assert actions == ["L", "R"]
    assert result == 3


def test_independent_conditional_evaluation_cannot_reuse_training_optimum():
    action, training_value = choose_nonclairvoyant((F(1), F(0)))
    evaluation_value = expected_cost(PRIOR, action)
    assert training_value == 0 and evaluation_value == 5


def test_predictive_representatives_must_not_be_probability_weighted_twice():
    masses, values = (F(9, 10), F(1, 10)), (F(0), F(10))
    correct = sum(p * v for p, v in zip(masses, values))
    twice = sum(p * p * v for p, v in zip(masses, values)) / sum(p * p for p in masses)
    # An exact 9:1 multiset represents these predictive probabilities.
    equal_representatives = sum([F(0)] * 9 + [F(10)]) / 10
    assert equal_representatives == correct == 1
    assert twice == F(5, 41) and twice != correct


def test_complete_macro_history_can_cancel_last_observation_preference():
    first, second = (F(9, 10), F(1, 10)), (F(1, 10), F(9, 10))
    _, whole = posterior(PRIOR, tuple(first[i] * second[i] for i in range(2)))
    _, last_only = posterior(PRIOR, second)
    assert whole == PRIOR and last_only != whole


def test_duplicate_fixed_key_is_not_independent_evidence():
    likelihood = (F(9, 10), F(1, 10))
    _, once = posterior(PRIOR, likelihood)
    _, repeated_wrong = posterior(PRIOR, tuple(p ** 100 for p in likelihood))
    assert once == (F(9, 10), F(1, 10))
    assert repeated_wrong != once
    # A real implementation must apply likelihood once to 100 repeated keys.


def test_one_shared_reception_radius_gives_joint_probability_point_two():
    joint = F(1200 - 1100, 1500 - 1000)
    independently_redrawn_wrong = F(1500 - 1100, 500) * F(1200 - 1000, 500)
    assert joint == F(1, 5)
    assert independently_redrawn_wrong == F(8, 25) != joint


def test_split_and_duplicate_support_cannot_inflate_effective_sample_size():
    assert ess((F(1),) * 4) == 4
    assert ess((F(1), F(0), F(0), F(0))) == 1
    assert ess((F(1),) * 2) == 2  # never claim ESS>=4 on a 2-world split
    copied_world_ids = ["same-physical-world"] * 4
    assert len(set(copied_world_ids)) == 1


def test_inconsistent_observation_has_no_invented_normalization():
    with pytest.raises(ValueError, match="no posterior"):
        posterior(PRIOR, (F(0), F(0)))

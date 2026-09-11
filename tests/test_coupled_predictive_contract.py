"""Exact finite probability contracts; no Q3 cases, policy or simulator imports."""
from fractions import Fraction as F
from itertools import product


STATES = (0, 1)


def observation_probability(z, w):
    return F(3,4) if z == w else F(1,4)


def posterior(w, z):
    # Equal prior and symmetric observation kernel imply P(Z)=1/2.
    return observation_probability(z,w)


def tail_cost(world, action):
    return F(8) * (world != action)


def expectation(distribution):
    assert sum(p for p,_ in distribution) == 1
    return sum(p*x for p,x in distribution)


def variance(distribution):
    mean = expectation(distribution)
    return sum(p*(x-mean)**2 for p,x in distribution)


def test_conditional_independence_factorization_and_exact_coupled_value():
    terms = []
    for w,z,t in product(STATES,repeat=3):
        joint = F(1,2)*observation_probability(z,w)*posterior(t,z)
        # Given z, selecting on the fresh T leaves the evaluator W posterior.
        assert joint/F(1,2) == posterior(w,z)*posterior(t,z)
        terms.append((joint,F(2)+tail_cost(w,t)))
    assert expectation(terms) == 5


def test_fresh_conditional_evaluation_has_exactly_same_expectation():
    terms = []
    for w,z,t,u in product(STATES,repeat=4):
        joint = F(1,2)*observation_probability(z,w)*posterior(t,z)*posterior(u,z)
        terms.append((joint,F(2)+tail_cost(u,t)))
    assert expectation(terms) == 5


def test_copying_evaluator_world_into_training_is_clairvoyant_and_biased():
    illegal = [(F(1,2)*observation_probability(z,w),F(2)+tail_cost(w,w))
               for w,z in product(STATES,repeat=2)]
    assert expectation(illegal) == 2
    assert expectation(illegal) != 5


def test_independent_but_unconditioned_evaluation_is_also_wrong():
    terms = [(F(1,2)*observation_probability(z,w)*posterior(t,z)*F(1,2),
              F(2)+tail_cost(u,t)) for w,z,t,u in product(STATES,repeat=4)]
    assert expectation(terms) == 6


def test_unbiased_random_training_policy_need_not_be_bayes_optimal():
    map_policy = [(F(1,2)*observation_probability(z,w),F(2)+tail_cost(w,z))
                  for w,z in product(STATES,repeat=2)]
    assert expectation(map_policy) == 4
    assert expectation(map_policy) < 5


def test_independent_inexact_training_still_evaluates_its_own_policy_value():
    # q(T=0|z)=1 ignores the posterior, but does not peek at evaluation W.
    # Its own policy value is 6, not the exact-posterior learner's value 5.
    terms = [(F(1,2)*observation_probability(z,w),F(2)+tail_cost(w,0))
             for w,z in product(STATES,repeat=2)]
    actual_fixed_action_value = expectation([(F(1,2),F(2)+tail_cost(w,0)) for w in STATES])
    assert expectation(terms) == actual_fixed_action_value == 6


def test_common_worlds_can_reduce_or_increase_difference_variance():
    a = (F(1),F(11))
    for b,expected_coupled in (((F(2),F(12)),F(0)),((F(11),F(1)),F(100))):
        common = [(F(1,2),a[w]-b[w]) for w in STATES]
        independent = [(F(1,4),a[w]-b[u]) for w,u in product(STATES,repeat=2)]
        assert expectation(common) == expectation(independent)
        assert variance(independent) == 50
        assert variance(common) == expected_coupled


def test_root_minimum_remains_optimistic_even_with_unbiased_candidate_means():
    a,b = (F(1),F(11)),(F(11),F(1))
    assert expectation([(F(1,2),a[w]) for w in STATES]) == 6
    assert expectation([(F(1,2),b[w]) for w in STATES]) == 6
    sample_minimum = expectation([(F(1,2),min(a[w],b[w])) for w in STATES])
    assert sample_minimum == 1
    assert sample_minimum < 6

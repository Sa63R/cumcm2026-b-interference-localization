"""Small mathematical certificates, independent of simulator seeds or Torch."""

import importlib.util
import math
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "information_cost_certificate", Path(__file__).with_name("information_cost_certificate.py"))
certificate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(certificate)


def test_exact_variable_stopping_tree_needs_no_extra_stopping_entropy():
    for count in (2, 17, 100):
        record = certificate.blind_clear_tree(count)
        assert record["chain_information_nats"] == pytest.approx(math.log(count))
        assert record["expected_trials"] == pytest.approx((count + 1) / 2)
        assert record["expected_successes"] == pytest.approx(1)
        assert record["chain_information_nats"] <= record["clear_information_upper_nats"]


def test_fixed_success_count_strengthens_log2_count_bound():
    assert certificate.clear_information_bound(13, 0) == 0
    for failures in (0.01, 0.5, 1, 13, 100):
        tighter = certificate.clear_information_bound(13, failures)
        assert tighter <= (13 + failures) * math.log(2) + 1e-12
        assert certificate.clear_information_bound(26, 2 * failures) == pytest.approx(2 * tighter)


def test_weighted_entropy_nodes_satisfy_jensen_clear_bound():
    # Independent nonuniform node examples, not the relaxation optimizer.
    nodes = [(0.2, 0.01), (0.7, 0.2), (0.6, 0.9), (1.0, 1.0)]
    mass = sum(weight for weight, _ in nodes)
    successes = sum(weight * p for weight, p in nodes)
    information = sum(weight * certificate.entropy([p, 1-p]) for weight, p in nodes)
    assert information <= certificate.clear_information_bound(successes, mass - successes)


def test_primal_dual_and_grid_checks():
    assert certificate.verify()


def test_quantized_bound_does_not_assume_iid_noise():
    loose = certificate.cost_certificate(certificate.QUANTIZED_CAPACITY)
    tight = certificate.cost_certificate(certificate.IID_CAPACITY)
    assert loose["extra_cost_per_source_s"] == pytest.approx(4.288194131991073)
    assert tight["extra_cost_per_source_s"] == pytest.approx(8.603532831922573)
    assert loose["extra_cost_per_source_s"] < tight["extra_cost_per_source_s"]
    assert math.log(36000) > certificate.IID_CAPACITY


def test_zero_failure_first_measure_constrained_kind_capacity():
    result = certificate.zero_failure_first_measure()
    visible = result["max_first_visible_probability"]
    near = result["max_first_near_probability"]
    # Brute-force feasible kind probabilities as a separate boundary check.
    for i in range(101):
        for j in range(11):
            near_p = near * j / 10
            direction_p = (visible - near_p) * i / 100
            value = (certificate.entropy([1-near_p-direction_p, near_p, direction_p])
                     + direction_p * math.log(180))
            assert value <= result["first_measure_capacity_nats"] + 1e-12
    assert result["expected_measures_per_source_lower"] > 2
    assert result["scope"] == "zero_failed_clear_policy_class_only"

"""Idealized-prior expectation certificates; never an actual benchmark bound.

Standard-library only. No simulator, benchmark, checkpoint, or seed is read.
See EXPECTED_INFORMATION_IDEAL_MODEL.md for assumptions and proofs.
"""

import argparse
import json
import math
from pathlib import Path


REQUIRED_PER_SOURCE = math.log(8100)
IID_CAPACITY = math.log(182)
QUANTIZED_CAPACITY = math.log(36002)


def entropy(probabilities):
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("probabilities must be in [0,1]")
    return -sum(p * math.log(p) for p in probabilities if p)


def clear_information_bound(sources, expected_failures):
    """Perspective of binary entropy, with exactly sources successes."""
    if sources <= 0 or expected_failures < 0:
        raise ValueError("positive source count and nonnegative failures required")
    total = sources + expected_failures
    # This form is numerically stable when failures are small relative to N.
    return (sources * math.log1p(expected_failures / sources)
            + expected_failures * math.log1p(sources / expected_failures)
            if expected_failures else 0.0)


def cost_certificate(capacity):
    """Primal/dual equality for these constants; counts are expectations."""
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be positive and finite")
    a = 3 * capacity / 5
    failure_ratio = 1 / math.expm1(a)
    clear_info = clear_information_bound(1, failure_ratio)
    measure_ratio = (REQUIRED_PER_SOURCE - clear_info) / capacity
    if measure_ratio < 0:
        raise ValueError("this closed-form positive-measure optimum is outside its regime")
    dual = 5 / capacity * (REQUIRED_PER_SOURCE + math.log1p(-math.exp(-a)))
    primal = 5 * measure_ratio + 3 * failure_ratio
    return dict(capacity_nats=capacity,
                required_nats_per_source=REQUIRED_PER_SOURCE,
                failure_expectation_per_source_at_relaxation_optimum=failure_ratio,
                measure_expectation_per_source_at_relaxation_optimum=measure_ratio,
                clear_information_nats_per_source=clear_info,
                extra_cost_per_source_s=dual,
                primal_cost_s=primal,
                primal_dual_gap_s=abs(primal - dual),
                simpler_log2_linear_extra_cost_s=5 / capacity * (REQUIRED_PER_SOURCE - math.log(2)))


def blind_clear_tree(worlds):
    """Uniform finite worlds, guess in order, stop on first success.

    This is a generic feedback-tree check, not a full competition scenario.
    The stopping time reveals the world, but that information is already in
    the success/failure history and must not be counted a second time.
    """
    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds < 1:
        raise ValueError("worlds must be a positive integer")
    information = 0.0
    expected_trials = 0.0
    expected_successes = 0.0
    for left in range(worlds, 0, -1):
        reach = left / worlds
        probability = 1 / left
        information += reach * entropy([probability, 1 - probability])
        expected_trials += reach
        expected_successes += reach * probability
    return dict(worlds=worlds, full_history_information_nats=math.log(worlds),
                chain_information_nats=information,
                expected_trials=expected_trials,
                expected_successes=expected_successes,
                expected_failures=expected_trials - 1,
                clear_information_upper_nats=clear_information_bound(1, expected_trials - 1))


def zero_failure_first_measure():
    """Only for independent Uniform[1000,1500] radii and zero-failure policies."""
    visible = (1000**2 + 1000 * 1500 + 1500**2) / (3 * 1800**2)
    near = 5**2 / 1800**2
    probabilities = [1 - visible, near, visible - near]
    first_capacity = entropy(probabilities) + (visible - near) * math.log(180)
    measures = 1 + (REQUIRED_PER_SOURCE - first_capacity) / IID_CAPACITY
    return dict(scope="zero_failed_clear_policy_class_only",
                max_first_visible_probability=visible,
                max_first_near_probability=near,
                first_measure_capacity_nats=first_capacity,
                expected_measures_per_source_lower=measures,
                extra_cost_per_source_lower_s=5 * measures)


def verify():
    # Mixed continuous/discrete reference output is a probability measure;
    # every exact-input response kernel has the same KL to this reference.
    assert math.isclose(2 / 182 + 360 / 364, 1, abs_tol=1e-15)
    assert math.isclose(math.log(1 / (1 / 182)), IID_CAPACITY, abs_tol=1e-14)
    assert math.isclose(math.log((1 / 2) / (1 / 364)), IID_CAPACITY, abs_tol=1e-14)
    # This concrete zero-noise finite-angle channel violates log182, showing
    # why a bounded-error assumption alone cannot imply the iid capacity.
    assert math.log(36000) > IID_CAPACITY
    for count in (1, 2, 3, 8, 100, 1000):
        tree = blind_clear_tree(count)
        assert math.isclose(tree["chain_information_nats"], math.log(count), abs_tol=2e-12)
        assert math.isclose(tree["expected_successes"], 1, abs_tol=2e-13)
        assert tree["chain_information_nats"] <= tree["clear_information_upper_nats"] + 1e-12
    for capacity in (1, 2, IID_CAPACITY, QUANTIZED_CAPACITY, 20):
        result = cost_certificate(capacity)
        assert result["primal_dual_gap_s"] < 1e-12
        a = 3 * capacity / 5
        envelope = -math.log1p(-math.exp(-a))
        for failure_ratio in [0.0] + [10 ** (i / 20) for i in range(-160, 121)]:
            g = clear_information_bound(1, failure_ratio)
            assert g - a * failure_ratio <= envelope + 1e-12
            measures = max(0.0, (REQUIRED_PER_SOURCE - g) / capacity)
            assert 5 * measures + 3 * failure_ratio >= result["extra_cost_per_source_s"] - 1e-12
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    verify()
    regimes = {"new_point_iid_uniform_continuous_bearing": cost_certificate(IID_CAPACITY),
               "centidegree_output_arbitrary_noise_same_uniform_location_prior":
                   cost_certificate(QUANTIZED_CAPACITY)}
    for result in regimes.values():
        result["source_count_examples"] = [
            dict(source_count=n, extra_cost_lower_s=n * result["extra_cost_per_source_s"],
                 occupied_nonmovement_lower_s=n * (5 + result["extra_cost_per_source_s"]))
            for n in (10, 13, 16)]
    payload = dict(scope="idealized_prior_expectation_only",
                   actual_benchmark_lower_bound=False,
                   per_instance_lower_bound=False,
                   success_requirement="almost_sure_all_clear; do not condition on successful runs",
                   source_count_conditioning="fixed true occupied channels C, then optionally average at fixed N",
                   all_numeric_checks_passed=True, regimes=regimes,
                   zero_failure_restricted_corollary=zero_failure_first_measure(),
                   stopped_feedback_tree_checks=[blind_clear_tree(n) for n in (2, 8, 100)])
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()

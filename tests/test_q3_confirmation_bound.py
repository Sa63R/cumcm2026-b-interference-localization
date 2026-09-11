"""Proof validation regressions; no simulator and no numerical optimizer."""

from copy import deepcopy
from fractions import Fraction
import json
import math

import pytest

from experiments.q3_confirmation_bound import (
    CERTIFICATE_PATH, POINTS, TARGET_LENGTH, area_length_lower_bound,
    confirmation_length_lower_bound, empty_channel_action_lower_bound,
    guarantee_time_lower_bound, pi_lower_bound, sqrt_upper,
    verify_certificate, verify_order,
)


@pytest.fixture(scope="module")
def certificate():
    return json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))


def test_pi_area_proof_uses_conservative_rationals():
    assert Fraction("3.14159265358979323846") < pi_lower_bound()
    assert pi_lower_bound() < Fraction("3.14159265358979323847")
    assert area_length_lower_bound() == pi_lower_bound() * 1120
    assert float(area_length_lower_bound()) == pytest.approx(3518.583772020568, abs=1e-10)


@pytest.mark.parametrize("value", [Fraction(), Fraction(1), Fraction(2), Fraction(1, 3),
                                  Fraction(123456789012345678901, 998877665544332211)])
def test_sqrt_is_exactly_conservative(value):
    bound = sqrt_upper(value)
    assert bound * bound >= value
    if bound:
        assert (bound - Fraction(1, 10 ** 12)) ** 2 < value


def test_sqrt_rejects_bad_parameters():
    with pytest.raises(ValueError):
        sqrt_upper(Fraction(-1))
    with pytest.raises(ValueError):
        sqrt_upper(Fraction(1), 0)


def test_integer_geometry_and_free_endpoint_formula():
    assert all(x * x + y * y <= 1800 ** 2 for x, y in POINTS)
    record = {"order": list(range(6)), "edge_dual_numerators": [[1, 0] for _ in range(6)]}
    # Constant duals telescope to q_6.x; min over the last disk is 900-1000.
    assert verify_order(record, 1) == -100
    record["edge_dual_numerators"] = [[0, 0] for _ in range(6)]
    assert verify_order(record, 1) == 0


def test_all_720_duals_verified_without_solver(certificate):
    result = verify_certificate(certificate)
    assert result["verified_orders"] == math.factorial(6)
    assert Fraction(result["minimum_verified_dual_m"]) >= TARGET_LENGTH
    assert Fraction(result["certified_length_m"]) == Fraction("4425.6")


def test_missing_order_rejected(certificate):
    damaged = deepcopy(certificate)
    damaged["certificates"].pop()
    with pytest.raises(ValueError, match="720"):
        verify_certificate(damaged)


def test_duplicate_order_rejected(certificate):
    damaged = deepcopy(certificate)
    damaged["certificates"][1] = deepcopy(damaged["certificates"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        verify_certificate(damaged)


def test_infeasible_dual_rejected(certificate):
    damaged = deepcopy(certificate)
    damaged["certificates"][0]["edge_dual_numerators"][0] = [damaged["dual_scale"] + 1, 0]
    with pytest.raises(ValueError, match="norm greater"):
        verify_certificate(damaged)


def test_feasible_but_insufficient_dual_rejected(certificate):
    damaged = deepcopy(certificate)
    damaged["certificates"][0]["edge_dual_numerators"] = [[0, 0] for _ in range(6)]
    with pytest.raises(ValueError, match="fails the target"):
        verify_certificate(damaged)


def test_geometry_changes_rejected(certificate):
    damaged = deepcopy(certificate)
    damaged["origin"] = [1, 0]
    with pytest.raises(ValueError, match="geometry"):
        verify_certificate(damaged)


def test_claimed_summary_is_not_trusted(certificate):
    damaged = deepcopy(certificate)
    damaged["verification_summary"] = {"minimum_verified_dual_decimal_m": 1e100}
    assert verify_certificate(damaged) == verify_certificate(certificate)


def test_area_fallback_and_invalid_file(tmp_path):
    missing = tmp_path / "missing.json"
    fallback = confirmation_length_lower_bound(14, certificate_path=missing)
    assert Fraction(fallback) <= area_length_lower_bound()
    assert fallback == pytest.approx(3518.583772020568)
    damaged = tmp_path / "damaged.json"
    damaged.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        confirmation_length_lower_bound(14, certificate_path=damaged)


def test_replaced_proof_is_reverified(tmp_path, certificate):
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")
    assert confirmation_length_lower_bound(14, certificate_path=path) == pytest.approx(4425.6)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        confirmation_length_lower_bound(14, certificate_path=path)


def test_guarantee_bound_uses_max_and_exempts_sixteen():
    assert confirmation_length_lower_bound(14) == pytest.approx(4425.6)
    assert Fraction(confirmation_length_lower_bound(14)) <= TARGET_LENGTH
    assert confirmation_length_lower_bound(16) == 0
    assert guarantee_time_lower_bound(14, 0) == pytest.approx(1135.12)
    assert Fraction(guarantee_time_lower_bound(14, 0)) <= Fraction("1135.12")
    assert guarantee_time_lower_bound(14, 5000) == pytest.approx(1250)
    assert guarantee_time_lower_bound(16, 5000) == pytest.approx(1080)
    assert empty_channel_action_lower_bound() == 30


@pytest.mark.parametrize("n", [0, 9, 17, 14.0, True])
def test_invalid_source_counts_rejected(n):
    with pytest.raises(ValueError, match="source count"):
        confirmation_length_lower_bound(n)

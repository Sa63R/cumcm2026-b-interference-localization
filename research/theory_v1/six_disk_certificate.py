"""Exact-arithmetic consequences of the published six-disk covering theorem.

This does NOT prove Bezdek's geometric theorem or modify benchmark bounds.
No scenarios, simulator, or experiment results are read. All arc inequalities
below have rational certificates; the covering theorem is an external premise.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path


def asin_upper(x: F, terms: int = 32) -> F:
    """Positive Taylor sum + tail using every remaining coefficient <= 1."""
    assert 0 < x < 1 and terms > 0
    partial = sum((F(math.comb(2*k, k), 4**k * (2*k+1)) * x**(2*k+1)
                   for k in range(terms)), F(0))
    return partial + x**(2*terms+1)/(1-x*x)


def atan_sum(x: F, terms: int) -> F:
    return sum(((-1)**k * x**(2*k+1)/(2*k+1)
                for k in range(terms)), F(0))


def rational_arc_certificate() -> dict:
    # Alternating series: an even number of terms is a lower bound; odd upper.
    # Machin's exact identity: pi = 16 atan(1/5) - 4 atan(1/239).
    pi_lower = 16*atan_sum(F(1, 5), 32) - 4*atan_sum(F(1, 239), 33)
    alpha_upper = F(117806195, 100000000)
    beta_upper = F(2222268, 100000000)
    circumference_lower = F(628318530, 100000000)
    checks = {
        'large_arc_upper_verified': 2*asin_upper(F(5, 9)) < alpha_upper,
        'small_arc_upper_verified': 2*asin_upper(F(1, 90)) < beta_upper,
        'circumference_lower_verified': circumference_lower < 2*pi_lower,
    }
    assert all(checks.values())
    return dict(alpha_upper=str(alpha_upper), beta_upper=str(beta_upper),
                circumference_lower=str(circumference_lower), **checks)


def empty_channel_relaxation(initial_channel: bool) -> dict:
    arc = rational_arc_certificate()
    alpha, beta, perimeter = map(F, (arc['alpha_upper'], arc['beta_upper'],
                                    arc['circumference_lower']))
    rows = []
    # m >= 8 already costs >= 40, so cannot improve a 33/34 second candidate.
    for m in range(8):
        f_arc = max(0, math.ceil((perimeter-m*alpha)/beta))
        f = max(f_arc, 7-m)  # Six enlarged disks cannot cover the arena.
        switch = int(m > 0 and not initial_channel)
        rows.append(dict(measurements=m, failed_clears=f,
                         entry_switch_s=switch, cost_s=5*m+3*f+switch))
    return dict(initial_channel=initial_channel,
                minimum_s=min(row['cost_s'] for row in rows),
                candidates=rows, exact_rational_arc_certificate=arc,
                relaxation_minimizer_is_not_a_feasible_cover_claim=True)


def certificate() -> dict:
    # This comparison is exact, but 1.7988... is imported from the published
    # theorem; this script is not an interval proof of that theorem's constant.
    normalized_lower = 1/F('1.7989')
    gap = normalized_lower-F(5, 9)
    assert gap > 0
    return dict(
        source_url='https://annalesm.elte.hu/annales27-1984/Annales_1984_T-XXVII.pdf',
        download_sha256='3cfad214b81f4541c3c26f92814e922bc48e34e1f18b71c8e508fe08a8f1ea9b',
        bytes=3400008, pdf_page_count=272, printed_pages_equal_pdf_pages=True,
        six_disk_theorem_statement_page=150, proof_reference_page=151,
        full_original_proof_read=False,
        status='published theorem statement; proof referred to 1979 thesis; corroborated 2005/2016',
        unit_covering_radius_lower_from_reported_decimal_prefix=str(normalized_lower),
        requested_radius='5/9', rational_gap=str(gap),
        constant_comparison_depends_on_published_theorem_not_independent_root_isolation=True,
        mixed_cover_integer_relaxation=[empty_channel_relaxation(v) for v in (True, False)],
        zero_failed_clear_minimum_measurements=7,
        requires_robust_all_legal_scenes_policy_and_N_below_16=True,
        does_not_modify_frozen_benchmark_bounds=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(certificate(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report+'\n', encoding='utf-8')
    print(report)

# Q3 paired final_stress report

A single partition report is not full final-suite acceptance.

| Method | Success | Failed clears | Mean s | P95 s | Savings | 95% CI saved s |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 28/28 | 0 | 3627.40 | 4881.47 | reference | - |
| state | 28/28 | 0 | 3046.34 | 4489.26 | 16.02% | 500.65 / 661.82 |
| rl | 28/28 | 0 | 3164.37 | 4687.37 | 12.76% | 323.06 / 590.48 |
| geo | 28/28 | 0 | 3254.15 | 4396.52 | 10.29% | 264.99 / 476.47 |

Means, P95 and paired comparisons retain failed cases with the predeclared penalty.

Recorded wall time may include concurrent load; use a serial benchmark for speed claims.

## Worst paired regressions

- state: q3-v1-stress-narrow_strip-810013 (-151.04 s); q3-v1-stress-minimum_radius-810000 (-275.52 s); q3-v1-stress-boundary-810022 (-279.57 s)
- rl: q3-v1-stress-positive_error-810003 (+384.36 s); q3-v1-stress-boundary-810022 (+123.11 s); q3-v1-stress-alternating_error-810026 (+24.59 s)
- geo: q3-v1-stress-minimum_radius-810000 (+430.25 s); q3-v1-stress-alternating_error-810026 (+174.75 s); q3-v1-stress-narrow_strip-810006 (-184.78 s)

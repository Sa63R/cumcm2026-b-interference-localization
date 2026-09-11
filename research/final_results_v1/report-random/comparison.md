# Q3 paired final_random report

A single partition report is not full final-suite acceptance.

| Method | Success | Failed clears | Mean s | P95 s | Savings | 95% CI saved s |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 256/256 | 0 | 3361.66 | 3843.91 | reference | - |
| state | 256/256 | 0 | 3081.41 | 3520.79 | 8.34% | 259.47 / 301.17 |
| rl | 256/256 | 0 | 3149.79 | 3624.61 | 6.30% | 187.04 / 237.12 |
| geo | 256/256 | 0 | 3214.96 | 3604.87 | 4.36% | 125.79 / 167.35 |

Means, P95 and paired comparisons retain failed cases with the predeclared penalty.

Recorded wall time may include concurrent load; use a serial benchmark for speed claims.

## Worst paired regressions

- state: q3-random-800015 (+125.08 s); q3-random-800217 (+96.41 s); q3-random-800011 (+81.96 s)
- rl: q3-random-800081 (+365.45 s); q3-random-800037 (+357.83 s); q3-random-800074 (+349.98 s)
- geo: q3-random-800102 (+267.73 s); q3-random-800072 (+255.97 s); q3-random-800082 (+246.12 s)

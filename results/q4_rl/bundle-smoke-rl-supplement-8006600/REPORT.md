# Scan-bundle functionality supplement with both frozen PPO controls

Two previously used TRAIN functionality fixtures, two seed clusters and different families; not independent validation, model ranking, generalization evidence or promotion evidence. No case or endpoint dropped.

Both prior PPO controls were included before running; the better complete-panel endpoint remains unsettled.

| Method | All-clear | Failed clear total | Mean T (s) | P95 T (s) | Sum T / sum L | Mean CPU (s) |
|---|---:|---:|---:|---:|---:|---:|
| r8 | 2/2 | 167 | 8825.893787 | 9769.087689 | 4.162574 | 0.117188 |
| legacy_ppo128 | 2/2 | 167 | 9320.219460 | 10492.453205 | 4.395713 | 0.585938 |
| macro_v2_ppo512 | 2/2 | 64 | 8651.114084 | 9005.431310 | 4.080142 | 1.250000 |
| macro_rule | 2/2 | 167 | 9140.487650 | 9814.725342 | 4.310946 | 0.437500 |
| bundle_rule | 2/2 | 496 | 12185.680310 | 12487.290686 | 5.747156 | 0.093750 |

| TRAIN seed / family | L (s) | R8 T/L | legacy PPO128 T/L | macro v2 PPO512 T/L | macro rule T/L | bundle rule T/L |
|---|---:|---:|---:|---:|---:|---:|
| 8006600 / random | 2165.065154 | 4.560550 | 4.906412 | 4.177611 | 4.567826 | 5.473534 |
| 8006601 / minimum_radius | 2075.529593 | 3.747429 | 3.862984 | 3.978468 | 4.042985 | 6.032582 |

Full paired seed-cluster bootstrap intervals, all-clear Wilson intervals, failure records and timing components are in summary.json. Two clusters make these descriptive only; P95 is an interpolation of two observations, not a population-tail estimate.

One CPU thread and sequential runs; wall/CPU measurements include checkpoint loading and may reflect different cache/import states versus retained rules. Virtual billing and bound definitions are identical.

Reproduce the four RL runs from the archived script and manifests; six rule records are reused byte for byte. No checkpoint was trained or selected here.

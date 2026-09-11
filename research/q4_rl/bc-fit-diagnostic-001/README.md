# Frozen micro BC classification diagnostic

Read-only, single-core analysis completed in 106.9 seconds. All four original synthetic TRAIN batches were included: 64 scenes, seeds 8001000–8001063, 22,376 teacher decisions. Warmstart SHA is bound in `results.json`; its script verifies that SHA and records each raw input hash. No model update or new rollout was performed.

| Original teacher action / role | Records | Greedy top-1 | Cross entropy | Mean teacher probability |
|---|---:|---:|---:|---:|
| measure / cover | 17,609 | 84.66% | 1.898 | 0.2343 |
| measure / localize | 1,343 | 7.97% | 2.948 | 0.1055 |
| certified_clear / safe_clear | 715 | 54.69% | 2.204 | 0.1767 |
| grid_clear / grid | 2,709 | 0.85% | 1.752 | 0.1964 |
| All | 22,376 | 68.95% | 1.953 | 0.2201 |

Exact equality of the actual 50 float32 candidate inputs defines within-decision ambiguity. Cover labels have indistinguishable alternatives in 74.16% of records, explaining a mean cross-entropy floor of 1.346 for this architecture. Localization and grid rows have almost no such ambiguity; accepting any candidate with the teacher's identical input does **not** improve top-1 in any group. Their very low accuracy therefore cannot be dismissed as channel-tie labels. Cover comprises 78.70% of all labels. No cross-state observability or representation-sufficiency claim is made.

Logged online BC loss was 4.397 → 2.317 → 1.991 → 2.084 across four different batches, each with three update epochs. These are not repeated measurements on a fixed set, so they do not establish a convergence plateau. The frozen model's in-sample classification remains weak on localization and grid clearing. More thorough, action/role-monitored BC is a better-controlled next diagnostic than assuming longer PPO will repair the initialization. Additional epochs alone are not guaranteed to work: label imbalance, fine feature distinctions, and missing public phase/history information remain possible explanations.

This is supervised classification on fixed teacher inputs, **not** a complete counterfactual trajectory or an independent performance evaluation. As context only, the original teacher rollouts cleared 64/64 scenes with 3,930 failed-clear attempts, mean T=7877.755 s and mean common lower bound L=1770.227 s, ΣT/ΣL=4.450139. That ratio does not describe the frozen learned policy. Results are in `results.json`; the reusable read-only computation is `classify.py`.

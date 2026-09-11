# Bundle-v3 completed development evaluation

All 6144 runs (12 methods × 512 cases) cleared every source and passed archived audits; all raw physical/feedback/billing histories were independently replayed. The panel contains 32 seeds × eight families × two source modes and is developmental.

R9 current qualification 93922e6f is outside this B panel. R8 is the frozen panel reference, not a claim of the current best rule; subsequent C evaluation includes R9.

| Method | Clear | Mean T (s) | T/L (ratio of sums) | P95 T (s) | Failed clears | Mean CPU (s) |
|---|---:|---:|---:|---:|---:|---:|
| r8 | 512/512 | 7651.59 | 4.27267 | 11938.49 | 29297 | 0.096 |
| macro_v2_ppo512 | 512/512 | 7998.67 | 4.46648 | 12325.29 | 22855 | 1.685 |
| attention_v3_bc512 | 512/512 | 8969.99 | 5.00887 | 14076.63 | 18817 | 5.700 |
| attention_v3_ppo512 | 512/512 | 8717.62 | 4.86794 | 13709.21 | 17195 | 5.343 |
| legacy_ppo128 | 512/512 | 8250.98 | 4.60737 | 12314.24 | 28252 | 0.771 |
| micro_rule512 | 512/512 | 7702.10 | 4.30087 | 11494.99 | 23655 | 2.628 |
| micro_v2_bc512 | 512/512 | 9122.84 | 5.09422 | 14287.12 | 18521 | 5.384 |
| micro_v2_ppo512 | 512/512 | 9294.87 | 5.19028 | 14827.24 | 20699 | 5.314 |
| mlp_v3_bc512 | 512/512 | 8888.42 | 4.96332 | 13743.79 | 19015 | 5.137 |
| mlp_v3_ppo512 | 512/512 | 9218.71 | 5.14776 | 14744.19 | 18785 | 5.128 |
| ppo_initialized512 | 512/512 | 9167.87 | 5.11936 | 13933.84 | 21577 | 5.304 |
| scst_initialized512 | 512/512 | 8749.88 | 4.88596 | 14189.87 | 16742 | 4.895 |

Preregistered eligible prior RL reference: **macro_v2_ppo512**. It is already included in the base memory-v4 specifications, so no additional RL reference arm is required; the separate R9 amendment supplies the thirteenth control. This is reference selection on development data, not a promotion.

- macro_v2_ppo512 versus r8: mean saving -4.54% (paired 32-seed-cluster 95% CI -5.95% to -3.13%; negative means slower).
- attention_v3_ppo512 versus r8: mean saving -13.93% (paired 32-seed-cluster 95% CI -15.51% to -12.37%; negative means slower).
- attention_v3_ppo512 versus macro_v2_ppo512: mean saving -8.99% (paired 32-seed-cluster 95% CI -10.38% to -7.64%; negative means slower).
- scst_initialized512 versus r8: mean saving -14.35% (paired 32-seed-cluster 95% CI -15.85% to -12.88%; negative means slower).
- scst_initialized512 versus macro_v2_ppo512: mean saving -9.39% (paired 32-seed-cluster 95% CI -10.53% to -8.24%; negative means slower).
- mlp_v3_ppo512 versus r8: mean saving -20.48% (paired 32-seed-cluster 95% CI -22.06% to -18.95%; negative means slower).
- mlp_v3_ppo512 versus macro_v2_ppo512: mean saving -15.25% (paired 32-seed-cluster 95% CI -16.73% to -13.81%; negative means slower).
- ppo_initialized512 versus r8: mean saving -19.82% (paired 32-seed-cluster 95% CI -21.33% to -18.31%; negative means slower).
- ppo_initialized512 versus macro_v2_ppo512: mean saving -14.62% (paired 32-seed-cluster 95% CI -16.02% to -13.22%; negative means slower).

Random and stress strata, full time components, inference/fallback costs and paired contrasts against R8 and the selected RL reference are retained in bundle-v3-independent-review-001.json. Full empirical P95 was independently recomputed; original P95 bootstrap intervals are retained without rerunning the same expensive resampling.

All-clear is distinct from zero failed optical clears. Different BC/PPO/SCST training exposure prevents attributing cross-model differences solely to architecture or optimizer. No model is independently qualified by this panel.

Object supervisor reports child_returncode=0 but final sync timeout; directory completeness is established by all 6144 evidence hashes and object readback, not by treating sync timeout as success. No model or simulator was rerun during this audit.

## Additional paired learning-phase contrasts

512/512 is the observed completion count for this fixed panel. Wilson intervals treating runs as independent are descriptive only; 512 correlated transformed scenes are not an independent reliability guarantee. The panel has 32 shared seed clusters.

- mlp_v3_ppo512 versus mlp_v3_bc512: mean saving -3.72% (95% paired-cluster CI -5.19% to -2.28%).
- attention_v3_ppo512 versus attention_v3_bc512: mean saving 2.81% (95% paired-cluster CI 1.68% to 3.93%).
- micro_v2_ppo512 versus micro_v2_bc512: mean saving -1.89% (95% paired-cluster CI -3.55% to -0.23%).
- scst_initialized512 versus ppo_initialized512: mean saving 4.56% (95% paired-cluster CI 3.42% to 5.74%).
- attention_v3_ppo512 versus mlp_v3_ppo512: mean saving 5.44% (95% paired-cluster CI 4.59% to 6.35%).

BC/PPO within architecture tracks the additional training phase. Cross-architecture and SCST/initialized-PPO comparisons also differ in actual training exposure; identical wall allocations do not imply identical data or update counts.

Raw failure-mode comparisons (all 512 cases per method):

| Method | Mean T s | T/L | Fallback cases | Prefix current scans/case | Prefix localization scans/case | Movement s/case | Detection s/case | Optical s/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r8 | 7651.59 | 4.27267 | 0 | n/a | n/a | 5664.43 | 1484.17 | 209.91 |
| macro_v2_ppo512 | 7998.67 | 4.46648 | 8 | n/a | n/a | 5230.17 | 2170.16 | 172.17 |
| micro_v2_ppo512 | 9294.87 | 5.19028 | 359 | 263.46 | 9.87 | 5867.82 | 2729.43 | 159.53 |
| mlp_v3_ppo512 | 9218.71 | 5.14776 | 343 | 225.69 | 9.64 | 5976.03 | 2580.35 | 148.32 |
| attention_v3_ppo512 | 8717.62 | 4.86794 | 287 | 193.69 | 8.74 | 5673.94 | 2418.94 | 139.00 |
| ppo_initialized512 | 9167.87 | 5.11936 | 341 | 213.94 | 6.84 | 5917.65 | 2574.13 | 164.68 |
| scst_initialized512 | 8749.88 | 4.88596 | 315 | 203.40 | 7.88 | 5609.68 | 2501.62 | 136.35 |

Prefix roles apply to explicit micro actions before fallback; R8/macro service macros have no equivalent role labels, so n/a is not zero. Complete measurement phases, switching/removal costs, fallback reasons and source-count exploratory breakdowns remain in the machine-readable audit.

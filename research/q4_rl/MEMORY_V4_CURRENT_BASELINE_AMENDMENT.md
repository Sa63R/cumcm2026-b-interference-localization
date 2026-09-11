# Current state-search reference amendment

Recorded 2026-09-11 19:02 UTC, before any memory-v4 policy rollout or performance result. The original MEMORY_V4_EVALUATION.md panels, eight new endpoints, endpoint selection rule, twelve base methods, accounting, and uncertainty definitions remain fixed.

A read-only check found that the independently qualified general state-search reference has advanced from R8 to R9 probe. Qualification commit: 93922e6f56685ed41fdc7051391bd29a28f6b44c, experiment/q4-r9-joint-visibility. Its research/q4_joint_visibility/qualification.json records phase=confirmation, selected=compact_joint_probe, passed=true. The later e57b016d5a20029abf546d6957ab37796651f3cb adds practice entry controls without changing the selected production sources. This external qualification is a reason to include a current reference, not a memory-v4 selection result.

Add one immutable control named r9_probe, entrypoint strategies.q4_joint_visibility:run_q4_joint_visibility, kwargs {"config":"probe","max_expansions":200}. Retain frozen r8 in all comparisons. Run R9 on both existing fixed512 and fresh128 panels under the same environment, seed requests, CPU constraints, billed costs and common lower bound as every RL arm. Every performance report identifies R9 as current qualified state-search reference and R8 as historical reference; any superiority claim also needs the paired R9 comparison. No existing case or method is removed.

R9 adds two production files to the C source base:

- src/planning/joint_visibility_region.py: c2df945c3a3dc295827fd7afbbb78793b43184cf61e1d1070f47020293953d7a
- src/strategies/q4_joint_visibility.py: eb922aae8052b48c9d54cd4b269a088498a3feeddbae254344b6ed73e31ff0e6

The inherited R8 strategy hash is unchanged: 641e8b0e4e6cce7b6445e88117d08ac23bd073487dfb46b87e903330f678ac69. Before launch, verify common production dependencies against the qualification manifest, preserve the original C release bytes, and freeze a separately named evaluation source package. Do not patch the completed training source tree. Any additional dependency difference requires an explicit reviewed record before evaluation.

The completed bundle-v3 summary selects macro_v2_ppo512 as the provisional prior-best eligible RL control under the original predeclared rule. It is already in the twelve base methods. Final eligibility requires complete raw/source/checkpoint audits, including checking that the reported final-sync timeout did not omit result objects. If this passes, the evaluation has thirteen arms and 8320 requested runs. Freeze this decision and the exact checkpoint SHA before launch; do not select it using memory-v4 results.

All endpoint selection remains developmental. No memory-v4 endpoint is recommended from teacher classification, training progress, the prior RL reference selection, or CPU throughput tests. Independent confirmation is still required for promotion.

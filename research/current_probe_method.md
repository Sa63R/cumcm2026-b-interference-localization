# Primary-channel current-position probe candidate

This branch starts at the protected v1 commit `8c624d0`. It contains one isolated mechanism: optionally enlarge the first active probe's finite position set with the robot's current position. Scheduling, source resolution order, coverage relocation, observation geometry, clearing, action limits, and termination remain inherited from `RelocatingStateSearch`.

## Mechanism

The parent `_next_probe` runs unchanged: the original nine candidates are scored, then the original nine plus fourteen axis-quantile candidates are scored using the same original nine-point winner as the geometry anchor. The resulting **23-point selection** is the comparator. The wrapper then checks whether the primary channel is detected and uncleared; this is probe index zero; there is no prior near-clear or radius-19.9 clear certificate; and the current point is fresh in that channel's six-decimal `observed_positions` set. This set includes all real measurements, including silence, rather than only direction measurements. Visits or measurements on another channel do not prohibit this point.

The extra point is considered only for a finite polygon with at least three vertices and nonzero signed doubled area, with every vertex at distance at most `1000 - 1e-5` m from the current point. The area and 1e-5 m reception margin are conservative engineering restrictions **on this extra point**; neither is claimed necessary mathematically. A line segment can still be handled by the unchanged parent. All original candidate filters still use their original threshold, and this wrapper does not relax reception using positive observations or a sampled subset of sources.

When eligible, the wrapper calls the same `choose_radius_probe` with the same original base set, axis candidates, nominal hypotheses, uncertainty weight and `(score, x, y)` tie rule, adding only the current point. It reuses neither hidden source locations nor actual unreturned feedback. Exact duplicate coordinates add no new candidate. The choice must remain the old selection or the new current point, and the complete finite score cannot worsen beyond numerical tolerance; invalid extended results fall back to the already-computed parent choice. Choosing a different point on an exact score tie is logged separately from strict proxy gain.

This replaces the location of the primary channel's already-planned measurement. It does not insert a cross-channel query and does not interrupt atomic source localization. If the current channel differs, the real protocol still charges switching; a measurement still costs 5 s even at zero movement. It may leave uncertainty requiring an additional subsequent probe, so a lower local score has no whole-episode performance guarantee.

## Audit data

`strategy_parameters.current_probe_log` has one row per enabled `_next_probe` call. It records the public `after_actual_action_count`, next expected one-based physical action ordinal, current position, channel, probe index, gate reason, freshness and maximum vertex distance when available. Eligible rows include a complete `baseline_probe_log` snapshot of the original 23-point decision, its point/score, extended score, exact-duplicate flag, actual selection change and `changed_on_score_tie`.

The existing `probe_search_log` row is updated to the **final selected position and score**, while preserving its original nine-point fields and family geometry. Its total runtime and geometry updates include the original two passes plus the additional third pass exactly once. The wrapper's separate log explicitly stores added computation; full process CPU is still required in experiments.

`_perform` delegates to the inherited physical operation, then attaches the actual accepted action ordinal/position/time in a `finally` block. Matching requires exact action, phase, channel, point and ordinal. Budget/rejection before an accepted action is labeled `not_accepted_or_budget_stopped`; an unexpected next action is labeled `different_next_action`. Neither case is reported as executed gain. A pending `planned` record without an actual action is only a planning record. Logging does not mutate the client state or fabricate an observation. Sixteen-source stopping and explicit exit are unchanged.

## Entry point and reproduction

- Candidate specification: `experiments/state_search_candidate_current_probe_v1.json`.
- Entry point: `strategies.current_probe_state_search:run_current_probe_state_search`.
- The spec's complete `kwargs` equals `state_search_candidate_relocating_cover_v1.json`; its `enabled=True` controls only the new wrapper, with original coverage relocation always enabled.
- For the disabled control, use the same entry point with `enabled=False`. It calls the original `_next_probe` directly and adds no candidate computation.
- Artificial tests: `python -m pytest tests/test_current_probe.py -q`.

The 38 artificial tests pass (0.89 s in the implementation environment). They include real finite-score geometry that chooses the new current point, exact parent-family comparison, no region/observed-key mutation, channel-local freshness, boundary and full-vertex reception checks, inherited fallback, disabled delegation, duplicate non-activation, all-three-pass cost accounting, real protocol charging and physical selected-point binding, action-budget stop, invalid-third-pass cost accounting and JSON serialization, exact-score tie reporting, and atomic same-channel measure then near-clear. These are implementation fixtures, not empirical policy-performance cases.

An additional **strict replay of the already-opened old development case 200114** supplied only the recorded response for each exactly matching action/channel/coordinate. Both original v1 and the disabled wrapper matched all 128 steps including enter/exit and all 126 physical actions byte-for-value in the strategy history. Both T=3361.326128 s, old physical oracle LB=1921.819589 s, T/LB=1.749033128; both complete. The receipt is `research/current_probe_implementation_check.json`, including the input gzip and source hashes. This checks the disabled implementation, not improvement or enabled whole-episode behavior. No new case, SQLite database, native simulator or official test was used in this branch's implementation work.

Root will freeze and coordinate the native old-case smoke, independent gate and fresh paired pilot. The hypothesis comes from a read-only diagnostic of 32 previously opened development histories: only 82/549 old active-probe prefixes had fresh certainly receiving current points, and adding them changed 45 choices with a mean local surrogate difference of 2.538496 s per whole case. This is motivation for a small isolated test, **not** an estimate of actual saved time or an evaluation result. Do not select future evaluation cases from these favorable prefixes or tune around their seeds.

# First-version candidate selection rule

This addendum is fixed before opening seeds 6048–6095. It does not change the original `research/v1_protocol.json`, its hash, its final acceptance thresholds, or the reserved future seeds. There are three separately reported research directions: state search, neural RL and geometric optimization; the original main branch remains the reference.

## Freeze before the selection set is opened

After development, register at most three candidates per direction. Record the actual Git commit, complete source hashes, exact specification hash, checkpoint byte hash where applicable, software platform, fixed tie order and one fallback candidate for each direction. The fallback must be one of that direction's registered candidates. Freeze these identities before generating or evaluating the first extended-validation case. Development results determine this shortlist; extended-validation or final results must not alter it.

The reference is the already declared `rollout_frozen` specification, with identical scene generation and the same Linux platform for every paired method. No official simulator case is part of this paired selection experiment. A model exported only to make its metadata portable is registered using the actual exported bytes before selection; its tensor equivalence is audited separately.

## Deterministic selection

Run all 48 extended-validation cases for every registered candidate and the reference, keeping every failure and its declared 360000-second penalty. Require complete case sets and identical per-case scenario hashes. A usable reference must itself be successful in every case with no failed clearances.

For each direction, first retain candidates for which every record has the original complete `successful` status (all sources cleared, a completion certificate, accepted exit and no errors), total failed-clear count is zero, and the following empirical ratio is at most 1.05:

`percentile(candidate penalized total times, 0.95) / percentile(reference penalized total times, 0.95)`.

This uses the existing shared-report percentile definition. It is not the percentile of the per-case time ratios. Among candidates that pass this screen, choose the smallest mean penalized total virtual time, before rounding displayed results. Exact numerical ties use the registered fixed order. Do not introduce a new confidence-interval filter, retune parameters, train against this set, or add a fourth candidate after viewing it.

If no candidate passes, or the reference makes the screen unusable, choose the previously registered fallback and explicitly mark that direction as **not passing extended screening**. The fallback does not retroactively count as having passed. Retain the failures, adverse tails and all other candidates' selection records.

## Independent final evaluation

Write the selection decision with hashes of all selection records. Keep the chosen source, specification and checkpoint identities unchanged. Then evaluate the selected candidate from each of the three directions and the fixed reference on both original final partitions: 256 random cases and 28 separately reported stress cases. All three directions remain in the final comparison, including one that failed extended screening or the practical performance target. Do not replace a selected candidate after seeing final results.

Use the unchanged practical final criteria and the shared report's separate random/stress checks. The extended set is a selection set: its winner's mean or interval is not an independent performance claim. Forty-eight observed successes also do not establish a universal reliability guarantee. Report the worst paired regression alongside P95 because the latter can conceal rare large regressions.

Final 95% intervals describe each predeclared branch contrast; they are not a simultaneous guarantee for a winner chosen after inspecting all final results. A subsequent implementation change requires a new independently declared evaluation and cannot reuse these final cases as fresh test evidence. The existing future-reserved partition remains reserved for later versions.

The motivation for separating selection and final evaluation is consistent with [Cawley and Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html). This document defines the first-version operational rule; it does not claim the selection procedure is statistically optimal.

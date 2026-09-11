# Any-point discovery: two bounded action-set trials

This changes what a v3 actor may physically do. It does not claim to repair the
observed measurement gap: the fair 48-case audit found RL used fewer coverage
measurements and more source-localization measurements than state search.
The new modes are independent research candidates, with unchanged base source
probes, clear geometry, MLP, flat action distribution and actual-time PPO reward.

## Frozen modes

`--feature-version v3 --probe-candidates anypoint_current` adds one legal
`measure(current_position, undetected_channel)` action per eligible channel.
`anypoint_targets` additionally offers at most four distinct destinations from
the existing known-source probe/clear candidates, ordered by distance from the
current position and then x/y. Sites with no fresh eligible channel are omitted.
The current point is not counted as one of these four target destinations.
No new movement primitive, hidden position, future observation or learned source
count is used. Movement occurs through the existing measure operation.

Both modes retain the full old candidate prefix and its original order. A pair
already offered by an old cover/probe action is not offered twice. A channel is
eligible only if not already detected/cleared, fewer than 16 distinct sources
are known, and the actual seven-site observations have not already proved the
channel absent. This removes only the new, provably unnecessary actions; no
old known-source observations or clear actions are removed.

Each episode permits at most 32 accepted extra measurements, at most 256 actor
decisions by default and the inherited action/time limits. When the extra
budget is spent the old action set remains. At the decision limit the existing
baseline completes discovery/localization/clearance. Its full actual cost is
still charged to the policy's final decision. Interrupted episodes retain the
existing explicit exit and unsuccessful/incomplete reporting rules.

## Evidence and features

An exact `Position -> measured channels` ledger records only accepted physical
measurements, including original coverage, probes and fallback. Float32 feature
rounding and the older six-decimal localization cache never establish equality
for the new ledger. New arbitrary points do not enter the seven-site coverage
certificate. A measurement counts there only if its actual position equals an
original cover point exactly. New discoveries enter the ordinary observed
source regions and become ordinary localization/clear candidates.

The tensor remains v3's 60 candidate features and 12 context features. Old rows
retain all old meanings. New actions have the existing single-channel cover
kind bits (indices 0 and 44), real point coordinates and displacements, the
selected channel, exact immediate cost `(distance/5 + 5 + switch)/1000` at
index 23, and the actual seven-site pending bits. Index 18 (`option/6`) is
`7/6` for current-point discovery and `8/6` for target-point discovery. At a
non-cover point the original point-pending/current-pending cover counts remain
zero: this is not a claim of having completed discovery there. The old cover
scan focus is retained. Additional-budget usage is recorded in metrics; no
new future-value or counterfactual-cost feature is invented.

New rows affect mean/max set pooling and action normalization even when every
old row and copied weight is identical. `action_schema` therefore freezes the
mode, point/measurement bounds, eligibility, ledger and new row conventions.
Old v3/base checkpoints remain loadable as base. Explicit `--initialize-from`
can copy their weights to either new mode with a fresh optimizer/trial, but
records `preserves_initial_probabilities=false`. Resume, inference requests,
and CLI metadata must agree on the full schema. These modes require MLP/flat;
mixing them with attention, grouping, v4 or axis probes is rejected in this
single-factor version. The actor/critic and PPO formulas are unchanged.

## Reproduction and limits

Use the existing CPU Python environment; commands below never contact a server:

```text
python -m pytest tests/test_deep_rl_anypoint_scan.py -q -p no:cacheprovider --basetemp=<new-test-directory>
python research/audit_anypoint_scan.py --output <new-audit-directory>
PYTHONPATH=src:. python -m research_rl.train --device cpu --feature-version v3 --architecture mlp --group-alpha 0 --probe-candidates anypoint_current --initialize-from <v3-base.pt> --hidden 96 --bc-episodes 0 --aux-bc-coef 0 --gae-lambda .95 --entropy-coef .005 --lr .0001 --output <new-trial> --scenario-start <unused-training-start> --scenario-end <inclusive-training-end> --max-attempted-episodes <sampling-cap>
```

The audit uses only fixed training seeds 110402/110403. It exercises both new
action families, teacher compatibility and forced baseline completion, then
checks original accepted-action physics/feedback/microsecond costs independently.
Every reported time includes the old pure physical DP lower bound
`LB=L/5+5N` and `T/LB`; the bound is an oracle relaxation, not an achievable
online optimum. Exercise policies intentionally prefer the new actions to test
them; their costs are not trained-performance results and must not select a
winning mode. The final set 2200001..2200256 and observation database remain
unopened. The separate effectiveness experiment must freeze its own training
budget, initialization, validation selection and final evaluation first.

## Completed mechanism checks

The regression run passed 148 tests, including the initial 14 new any-point
tests, existing joint-scan/axis/training/distribution/attention/portable-checkpoint
tests and interrupted-sampling budget tests. A final targeted check adds explicit
four-target eligibility/cap and tampered-budget metadata rejection coverage.

All 12 mechanism records on the two fixed training cases completed, explicitly
exited, and had zero failed clears. Both extended teacher runs exactly reproduced
the corresponding old teacher's physical actions and time. Both exercise modes
actually used their full 32-extra-measurement allowance; three-decision runs
exercised the original baseline fallback. Raw records, hashes, all time
components and per-case bounds are in `anypoint_scan_evidence/smoke_v1`.

| Training fixture | Pure physical LB (s) | Old teacher T/LB | Current exercise T/LB | Targets exercise T/LB |
| --- | ---: | ---: | ---: | ---: |
| 110402 | 1737.392268 | 2.020249 | 2.023404 | 2.073386 |
| 110403 | 1682.605547 | 2.271234 | 2.074677 | 2.324120 |

These forced-exercise results are mechanism checks and do not select a mode.
One additional fresh random MLP16 trial used one spawned CPU worker, two
training cases, a 64-decision cap and one real PPO update. Both episodes fully
completed with zero failed clears; actual rewards exactly matched full charged
time, including fallback. Its deliberately untrained times were 16342.129449 s
and 18610.088638 s, with bounds 1500.118733 s and 1891.008892 s (T/LB 10.893891
and 9.841354). This only establishes the spawn/training/checkpoint path works;
it is not the production MLP96 policy or a performance result. Filtered evidence
is in `anypoint_scan_evidence/spawn_smoke.json`.

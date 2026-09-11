# Shared service positions: whole inner-station prototype

Status: composable geometry/planning ledger, **not integrated into the controller,
not trained, not promoted**. No simulator, validation database, random scenario
panel or formal test was used. Scripted receipts below test contracts, not policy
performance; there is consequently no whole-task time/lower-bound ratio to report.

## Scope and exact-proof provenance

`src/q4_rl/shared_cover.py` derives at most four prospective service positions
from accepted positive observations: actual near positions, positive-region outer
centers and transverse probes constructed from an actual bearing. Up to two of
these positions can replace one radius-970 inner station. Positions are finite
candidates, not predicted detections, probability certificates or safe-clear masks.
The controller's existing clear safety checks must remain in force.

`src/q4_rl/adaptive_cover.py` is copied **byte for byte, without changes** from the
adaptive-cover branch commit `350e121124bfbfd602a8892e2d3fdc6906e7888f`.
SHA-256: `fa62775bfb998f522174520f14f3d224c286e2e21512f5be82bd0a1b797549f7`.
It searches a dyadic partition and checks every accepted support witness with
exact rational arithmetic on the submitted coordinates. The proof covers every
source position and half-plane orientation; a negative observation is never
treated as an omnidirectional 1000 m exclusion disk. Existing implementation and
provenance replay are retained unchanged.

## Public interface

1. Construct `SharedCoverLedger(compact_22_points)`. Initial coverage proof is
   charged against the episode limit of eight certificate operations.
2. Feed **every accepted physical measure/clear** through
   `observe(index, action, point, channel, result, accepted=True, bearing_deg=...)`.
   Indices start at zero and exclude enter/exit. A rejected response with
   `accepted=False` grants nothing. `direction` requires its actual public bearing.
3. `service_points(current)` returns at most four `ServicePoint` objects entirely
   derived from the ledger's accepted public history. Policy inputs may include
   their public geometry, estimated costs and outstanding channel obligations;
   a policy must not receive the ledger, client, truth or scenario seed.
4. Give `proposals(points, baseline_actions, current, current_channel,
   target_station=None)` the **full explicit remaining public schedule**. Use
   `PlannedAction(kind, position, channel, purpose)` with purpose `service` or
   `coverage`. Every still-pending unknown-channel observation must be present.
   Proposed service positions must already have real planned service visits
   (or be current); no free extra detours are assumed. If any retained action
   still visits the old station, it cannot be cancelled.
5. Choose a returned `SharedPlan`. This changes neither actual observations nor
   pending obligations. Execute actual actions through the existing safe action
   path, retaining the original pending schedule until the scan packet succeeds.
6. After receiving real measurements at the selected service points, call
   `cancel_station(plan)`. Every still-unknown owed channel must have separately
   returned `no_signal` at every selected point. An actual positive/clear can
   remove its channel from the unknown packet. Only then are the entire station's
   still-unknown obligations removed atomically. Missing/changed feedback or
   exhausted proof budget leaves the original obligations intact.
7. Save `ledger.history` and `ledger.artifact()`; audit with
   `replay_shared_cover(history, artifact)`. Replay checks exact proof support,
   action prefixes, channels, atomic station groups, no unaccounted pair events,
   absence claims and actual station visits independently.

The ledger exposes pending obligations as a contract for the executor, not as an
independent task completion mechanism. A full fixed-cover/frozen-resolver fallback
must consume all retained obligations and clear discovered sources. This prototype
does not alter R8, micro-controller, policy features, network or training code.

## Full public cost filter and bounded computation

The supplied schedule is priced with movement `ceil(distance/5 * 1e6)/1e6`, every
five-second detection, every channel switch, and a five-second upper cost for
each retained optical request. Clear requests do not retune the measurement
channel. The comparison retains clear actions and counts all newly inserted
per-channel service scans. A full schedule is required to prevent hiding costs
beyond a short planning horizon. Candidate plans with nonpositive savings are
discarded before exact geometric search.

This is a **schedule proxy**, not an expected return or evidence that the current
policy is faster. Outcomes can change subsequent localization, routing and scan
needs. Choosing a bad baseline route can make a candidate look attractive without
beating a better route; this must be tested in a later frozen paired experiment.

The episode cap is eight certificate operations including initial proof,
prospective queries (even cache hits) and commit-time exact verification. Proposal
enumeration reserves one operation for a possible commit. Default prospective
query cap is 0.5 seconds / 20,000 cells; failure, timeout or inconclusive proof
keeps existing obligations. Initial proof has a cell cap but no strict wall-time
deadline in the unchanged upstream API. Offline replay is a separate audit whose
CPU time must be counted in later evaluations. Subsequent certificate attempts
reuse the base partition; there is no certificate search on every controller step.

## Scripted evidence and limitations

The existing compact-22 geometry admits replacement of `(970, 0)` by the two
public positions `(770, 200)` and `(1170, -200)`. Each point alone fails to obtain
the exact certificate. In the scripted history, channels 1–10 are known through
actual near receipts; channels 11–20 are unknown. The plan grants no credit until
both new positions have actually been measured on each unknown channel. It then
records **10 deleted channel obligations and 1 cancelled station**, with 21 other
stations still pending per unknown channel; this is not a completed episode.

The ordinary frozen point ordering makes the two-point replacement cost-negative,
so it is filtered. A deliberately long-detour order placing the target station
last produces a positive proxy saving and tests the complete proposal/commit/replay
path. This constructed route is explicitly not policy evidence. Adding two scans
per unknown channel in place of one can outweigh the route saving, and one-point
proposals can save one switch while still failing geometric feasibility.

Raw scripted history, complete before/after schedule costs, certificate witnesses,
failed candidate diagnostics and relative source hashes are preserved in
`results/q4_rl/shared-cover-geometry/scripted_evidence.json`.

Run the isolated checks with the current environment's Python and `PYTHONPATH=src`:

```text
python -m pytest tests/test_q4_shared_cover.py -q
```

Next integration should explicitly retain a pending scan packet across decisions,
accept only real wire receipts, and replan when service geometry/outcomes change.
The minimal policy comparison is frozen micro + unchanged action set; micro +
these candidates and deterministic cost selection; then the same candidates with
learning. Report actual skipped station count, extra per-channel detections,
movement/switching, proof CPU cost and full-task T/L on the same new development
panel. Geometry existence alone is insufficient to continue a costly route.

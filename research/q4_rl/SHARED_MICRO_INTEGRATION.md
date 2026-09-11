# Shared micro integration: development smoke protocol

The independent controller is `q4_rl.shared_micro_controller:run_q4_shared_micro`.
`shared_enabled=False` dispatches directly to the unchanged original micro
controller. Enabled mode is rule-only, explicitly rejects a supplied policy and
uses the new `q4-shared-micro-g2-rule-v1` schema (17 global / 54 candidate fields).
Original G1 13/50 checkpoints have unchanged meaning and are not accepted here.

The extension supplies at most four observation-derived service positions times
unknown channels as individual actual-measure candidates. A deterministic rule
only prioritizes a packet after the unchanged shared-cover module has certified
whole-station replacement and found positive full public schedule proxy savings.
It chooses at most one next service point per known source, constructs a nearest
remaining visit schedule containing every retained unknown-channel discovery
obligation and the selected next service requests, and counts moves, detection,
channel switches and clear upper cost. This is not a predicted full-task rollout:
future localization observations and subsequent service requirements remain
unknown. A poor public schedule can still overstate practical savings.

Reviews require at least two eligible public service positions, occur at most once
per public service geometry snapshot and at most eight times per episode. The
whole shared ledger still permits at most eight certificate operations including
initial proof and commit verification. No certificate or review expands sampling.
Uncertified additional candidates are not selected by this first conservative rule.

Execution is one accepted measure/clear per micro decision. At an active packet
position a legal known-source service request can precede unknown-channel scans.
The actual fixed-point ledger is never edited to represent cancelled obligations.
The separate exact residual ledger accepts every real measure/clear receipt in
order. It atomically removes a station only once all currently owed unknown
channels have independently supplied the required actual negative receipts.
Fallback consumes the retained obligations and the original saved grid queues.
The final gate requires exact replay against actual action history before claiming
completion. Later wire-level audits remain independent of this controller.

Packet scans may reuse the original `current` candidate at a service position.
Therefore `shared_service_actual_measurements` counts actual scans performed for
the packet, while `shared_added_candidate_actual_measurements` separately counts
only the new candidate role. Neither equals deleted channel obligations or the
number of cancelled/actually skipped stations.

Before evaluation, 47 scripted safety/original-micro regression tests passed.
They cover exact disabled dispatch parity, no-packet physical parity, single-action
execution, rejected-response exclusion, per-channel atomic cancellation, unchanged
actual fixed ledgers, actual final completion replay and explicit schema rejection.

Frozen panel: development seeds 8106000 and 8106001, family `random`, modes
`mixed` and `all_directional`, all four scenarios retained, methods R8, micro512,
shared512. R8 and the evaluator/protocol/scenario generator are unchanged. Resource
cap: two CPU workers, one numerical thread each, GPU disabled. No official or
practice simulator, verification database, server process or training is used.
This is a small interface/development smoke; two independent seeds are inadequate
for promotion or precision performance claims. Mode variants share their seed
cluster and are not four independent random draws.

```text
python experiments/q4_rl_evaluate.py --output results/q4_rl/shared-micro-smoke --specs research/q4_rl/shared_micro_smoke_specs.json --stage development --start 8106000 --count 2 --families random --source-modes mixed all_directional --workers 2 --reference r8 --bootstrap-samples 2000
```

The standard evaluation freezes all source/config hashes and source archive before
the first arm. It preserves raw wire observations, result metadata, failures,
actual billed costs, common lower bounds, T/L, tails, paired uncertainty and CPU
cost. Shared certificate operation counts, review CPU wall time and final exact
replay time are available inside each raw result's learning metadata. The frozen
evaluator already uses seed-cluster bootstrap; only two clusters exist, so its
numerical interval is a descriptive smoke statistic with very weak generalization.

## Observed outcome: no gain, keep as an unsuccessful development route

All 12 runs completed and passed independent physical and actual-completion audit.
Shared512 and original micro512 have identical accepted action histories on all
four cases. Full wire records differ only by wall-clock response timestamps; the
posthoc audit checks equality after removing that single documented clock field.
No case is omitted and no extension action is credited without a real response.

| Method | Complete | Failed clears | Mean T (s) | P95 T (s) | Sum T / sum L | Mean CPU (s) |
|---|---:|---:|---:|---:|---:|---:|
| R8 | 4/4 | 69 | 7651.808 | 8243.087 | 3.64376 | 0.10156 |
| Micro512 | 4/4 | 69 | 7921.126 | 8382.357 | 3.77201 | 1.21094 |
| Shared512 | 4/4 | 69 | 7921.126 | 8382.357 | 3.77201 | 2.30078 |

Common mean lower bound is 2099.977 seconds. Shared512 saves zero seconds against
micro512 on every observed case (paired bootstrap interval `[0, 0]` only describes
this two-seed panel). Against R8, both are 3.5197% slower on average, with one win
and three losses; maximum observed regression is 787.699 seconds. Shared adds
1.08984 CPU seconds per run on average. None uses fallback on this panel.

Each episode considered its first two eligible natural service positions. Exact
replay of those **same actual observation prefixes**, without alternate-action
feedback, reconstructed the public schedule and enumerated all cheap costs. All
seven two-point/inner-station replacements were cost-negative on every episode.
Each episode also had 14 cost-positive single-point proposals; the top six consumed
the permitted prospective search budget. Across the 24 actual searches, 17 had
counterexamples and seven were inconclusive. No certificate was accepted, no
scan packet executed, and no station was cancelled. These findings explain the
physical identity with original micro512.

This does **not** rule out useful later service geometry: the initial unsuccessful
queries consumed the episode search allowance, so later snapshots were not
examined. The first observed service centers also need not be the best two-point
geometry. A useful next change would need a stronger public necessary-condition
filter or a deliberate budget reservation across observation stages, followed by
fresh paired development evidence. Repeating this exact early-single-point search
with longer training has no demonstrated benefit; this controller is rule-only.

Raw records, frozen source archive and summary are under
`results/q4_rl/shared-micro-smoke/`. `shared_posthoc_audit.json` preserves the full
public-prefix candidate costs, exact final replays, per-record hashes and the
comparison against micro512. Reproduce this additional read-only audit with:

```text
python scripts/q4_shared_smoke_audit.py
```

The audit client rejects every request that differs from the next recorded action
and never predicts a new feedback result. Candidate schedule pricing in this audit
is a geometry/cost diagnostic, not counterfactual policy performance.

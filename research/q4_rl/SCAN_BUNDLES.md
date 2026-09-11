# Fixed-station scan groups: independent action representation experiment

Entry: `q4_rl.bundle_controller:run_q4_bundle`, default 128 upper decisions.
Schema: `q4-scan-bundle-service-g1-v1`, 10 global / 18 candidate features;
`feature_schema()` includes names and `unknown-fixed-station-current-first-v1`.
The CPU actor callback remains `(global_features, candidate_features) -> index`.
No prior macro/micro checkpoint is compatible with this independent schema.

A scan candidate groups the still unknown, unmeasured channels at one certified
station. The current tuned channel executes first if present, then ascending
channel order. Every member submits an actual request and consumes its public
reply before the next member. Discovery certification, including 16 actually
observed distinct sources, cancels the remaining scans immediately. It does not
certify clearing: actual successful clear receipts remain mandatory. Source
service uses the frozen R8 finite resolver, with inherited physical budgets and
complete coverage/resolution fallback. Rejected or interrupted groups retain
only actual receipts; final fallback and exit charges enter the last transition.

The first ten global values are unchanged public macro summaries. Candidate
geometry/count/remaining-cover fields are means over the group's channels.
Index 5 is the full scan-group cost upper bound (microsecond ceiling travel,
five seconds per scan, actual deterministic switching count), zero for service.
Index 16 is remaining group size / 20, zero for service. Index 17 is the service
first-request travel-plus-five estimate, zero for scan groups; it is explicitly
not a prediction of total service cost. Earlier discovery may reduce group cost.
Training consumes actual accumulated charges with gamma=lambda=1, never this
estimate, a hidden source state, seed, case ID, or theoretical bound.

This action set removes actor interleaving inside a group and standalone
known-channel / arbitrary-current-position probing. It is therefore grouping
plus a control restriction, not a claim of algebraically equivalent actions or
learned per-channel joint decisions. Any future change needs a semantic version
and separate action ablation. Frozen original controllers were not modified.

## Two prespecified TRAIN functionality cases

`research/q4_rl/run_bundle_smoke.py` runs seeds 8006600/8006601 (all-directional,
random/minimum-radius respectively; both contain 15 sources) with CPU one.
They are training-partition cases, not independent validation or promotion.
All six runs passed independent physical and public completion audits. Each
bundle transition sum equals complete billed time. No fallback was used.

| TRAIN seed | Frozen R8 T / T:L | Original macro rule T / T:L | Bundle rule T / T:L | Upper decisions macro → bundle |
|---|---:|---:|---:|---:|
| 8006600; L=2165.065154 | 9873.887012 / 4.56055 | 9889.640641 / 4.56783 | 11850.557670 / 5.47353 | 278 → 37 |
| 8006601; L=2075.529593 | 7777.900561 / 3.74743 | 8391.334660 / 4.04298 | 12520.802950 / 6.03258 | 269 → 37 |

| Mean over two cases | Frozen R8 | Original macro | Bundle |
|---|---:|---:|---:|
| Total billed seconds | 8825.894 | 9140.488 | 12185.680 |
| Movement / switching | 6853.894 / 249 | 7213.988 / 241 | 9793.680 / 198 |
| Detection / optical / removal | 1397.5 / 295.5 / 30 | 1360 / 295.5 / 30 | 1375 / 789 / 30 |
| Failed clear attempts | 83.5 | 83.5 | 248 |
| Policy process CPU seconds | 0.1172 | 0.4375 | 0.0938 |
| Policy wall seconds | 0.1183 | 0.4344 | 0.0943 |
| p95 billed seconds, two-point interpolation | 9769.088 | 9814.725 | 12487.291 |

Upper decisions fell 86.5%, but default bundle-rule mean time regressed 38.1%
against R8 and 33.3% against the original macro rule. The main differences are
movement (+2579.693 s vs macro) and optical time (+493.5 s); switching saved
43 seconds. Missing opportunistic known-channel probes and changed service
timing may explain extra optical traversal, but these two cases do not identify
causality. A later controlled ablation should separately restore known-source
probe choices and change service scheduling. No heuristic tuning was attempted
on these outcomes, and no model was trained or recommended here.

All methods clear 2/2; Wilson 95% interval is [0.3424, 1]. With only two TRAIN
clusters, raw paired bootstrap ranges and p95 are descriptive, not convincing
generalization evidence. `summary.json` retains all uncertainty, failures,
components and R8 comparisons. Bounds are computed only after policy exit with
the unchanged common definition. CPU timing excludes the separately reported
physical/completion audit and bound computation; Windows CPU clock is coarse.

Raw histories, transitions, actual group membership, source archive, specs,
requests, summaries and verified hashes are under
`results/q4_rl/bundle-smoke-8006600/`. The source freeze explicitly excludes the
unused network/train modules being developed in parallel. Tests cover schema,
group billing, reply-by-reply updates, early discovery, absence of false clear
credit, four interruption modes, rejected requests, full fallback/exit fees,
failed clears and forged groups: 25 controller tests passed (new plus original).

## Training contract and existing RL supplement

The independent `bundle_network.py` / `bundle_train.py` expose the 10/18 MLP
and transactional CPU BC/PPO driver. They share the original full undiscounted
cost objective, retain every fallback/exit charge, and reject cross-schema
checkpoints. Thirty-six controller/training tests passed on one CPU, covering
actor/critic updates, grouped-return boundaries and exact model/Adam/RNG
recovery after administrative interruption. No scan-bundle model has been
trained or recommended; these checks establish implementation contracts only.

The same two TRAIN requests were also run with both frozen earlier RL controls.
Their original evaluation manifests establish the checkpoint hashes, and their
four new raw records are retained alongside byte-identical copies of the six
original rule records in `results/q4_rl/bundle-smoke-rl-supplement-8006600`.
The complete five-arm table, uncertainty and compute timing are in that
directory's `REPORT.md`. Mean T / ratio of sums is R8 8825.893787 / 4.162574,
legacy PPO128 9320.219460 / 4.395713, macro-v2 PPO512 8651.114084 / 4.080142,
macro rule 9140.487650 / 4.310946 and bundle rule 12185.680310 / 5.747156.
The two RL controls each win one case. These two used TRAIN cases cannot select
a best RL version or establish a generalization gain. The full frozen paired
development comparison remains the basis for choosing the prior RL reference.

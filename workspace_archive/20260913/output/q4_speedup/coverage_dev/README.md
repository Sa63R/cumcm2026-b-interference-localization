# Coverage and route speed development

These are synthetic development results, not official practice results and not an independent final validation. Seeds are 301000000 + scenario_index * 100000 + case_index. The six historical scenario groups use vendor `make_case` and its original error model. Consequently the fraction-0/1 generator and 0.01-degree rounding fixes in the parent `bootstrap.py` do not apply to these development runs. The parent benchmark must perform final validation for any selected candidate.

No vendor file was changed. All planners receive the public observation interface. Every completed coverage run checks that all generated sources were cleared. Every coverage-changing candidate also verifies the continuous certificate on its actual completed scan points when stopping by coverage. Failed or unresolved geometric checks never authorize a moved or removed station.

## Findings

- 2,640 coarse two-ring layouts with 19, 20 or 21 stations were checked. No 19/20-station layout passed; only configurations near the existing 21-station construction passed. An additional 400 tight 20-station configurations also failed to pass. This is a negative search result, not a proof that fewer stations are impossible.
- The original center + 8 inner stations at 998 m + 12 outer stations at 1865 m has a 17,831 m static scan-only route under the existing heuristic. Removing the center or any inner station fails the continuous certificate.
- Single-site replacement at reached clear/probe points almost never succeeds: 1 replacement in 60 cases. Average gain is 0.00275 seconds/source.
- Buying extra discovery scans at clear points to combine their coverage costs 5.92 to 30.71 extra seconds/source. It almost never removes a station. Do not promote this candidate.
- Choosing an initial common rotation from 24 possibilities after the origin scan saves 2.33 seconds/source across 60 cases, but the normal 95% interval is [-3.29, 7.96]. This is not reliable evidence of speedup.
- More route starts (32 instead of 8) lose 2.35 seconds/source in 60 cases. Node relocation plus 2-opt gains 0.334 seconds/source with interval [-0.098, 0.767], also inconclusive.
- A 23-station layout (10 at 975 m, 12 at 1865.5 m) and a 25-station layout (10 at 950 m, 14 at 1848.3 m) both certify. Their static scan-only routes are approximately 17,911 and 17,826 m. Despite increased flexibility, their actual task performance loses 12.59 and 24.52 seconds/source in an 18-case screening because extra scans and altered decisions outweigh the travel changes.
- `FlexibleState` with the original 21 stations changes the next station toward the incoming/outgoing route only when jointly certified. It has 54 accepted shifts in 18 cases, 15 faster/2 slower/1 tied, average gain 1.43 seconds/source (0.33%), interval [-0.48, 3.33]. Runtime is about 1.79 seconds/case. This is the only remaining coverage candidate worth a small separate validation, but no speedup is established.
- Tangential angle changes preserving inner-ring radius were also tested: 21-station version loses 1.36 seconds/source over 18 cases, despite 75 accepted shifts. More freedom alone does not imply faster completion.

## Data files

`rings.json`: coarse certified-layout search results. `tight20.json`: additional tight 20-station search (empty accepted set). `redundant_layouts.json`: certified 22–26-station alternatives and static routes.

`coverage_runs.json` / `coverage_summary.json`: 60 cases × 5 baseline/rotation/replacement methods.
`bonus_runs.json` / `bonus_summary.json`: 60 × 5 paid opportunistic scanning methods.
`route_runs.json` / `route_summary.json`: 60 × 4 route-start settings.
`relocate_runs.json` / `relocate_summary.json`: 60 × 3 route relocation methods.
`flex_runs.json` / `flex_summary.json`: 18 × 6 station flexibility/density methods.
`tangent_runs.json` / `tangent_summary.json`: 18 × 3 tangent-shift methods.

Total: 1,182 full synthetic development runs, all cleared; baseline repetitions are included. These are multiple exploratory comparisons with reused development cases, so reported confidence intervals are descriptive and do not correct for selection.

## Code interfaces

`code/experiments/q4_speedup/coverage_candidate.py`:
- `CandidateState`: initial rotation or certified complete-site replacement.
- `OpportunisticState`: bounded extra full scans at reached clear points; rejected by development evidence.
- `RouteState`: node relocation and 2-opt routing.
- `FlexibleState`: certified next-station movement toward the route.
- `TangentialState`: certified fixed-radius angular station movement.

The flexible/tangential classes override `ordered_actions` only, so their ordering method can be combined with a local action implementation. Preserve the continuous final actual-scan certificate and the source-upper-bound termination condition.

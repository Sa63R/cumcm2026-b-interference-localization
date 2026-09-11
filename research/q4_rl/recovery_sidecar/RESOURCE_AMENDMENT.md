# Evaluation-only recovery resource amendment

Recorded after the externally constrained disk stop and user-approved historical
archive cleanup, before recovery preparation or any additional evaluation rollout.
The old evaluation, model/source freeze and incomplete raw records remain intact.

At 20:05 UTC on 2026-09-11, the shared filesystem reported 21,835,022,336 bytes
free. This leaves only 360,185,856 bytes above the previous 20 GiB guard. The four
GAE training jobs remain stopped because their expected additional journal volume
is too large. This amendment applies solely to completing the fixed C evaluation.

Use a new recovery supervisor run, ten one-thread workers plus parent, CPU budget
50 and the existing adaptive eight-core reserve. No GPU. Set its free-disk guard
to **18 GiB**, and freeze an independent **1,000,000,000-byte cap on newly written
recovery artifacts**. Original immutable record hard links do not duplicate their
payload blocks. Preparation must bind the exact cap; execution cannot enlarge it.
Cap exhaustion, source change or the disk guard produces an administrative stop
with all complete and partial evidence retained, not a partial performance report.
Extra supervisor status/console logs must also be accounted or explicitly bounded.

The lower guard is justified by the evaluation's bounded additional footprint;
it is not a relaxation for the multi-gigabyte training runs. Current disk space
and load must still be rechecked before launching. Preserve the original fixed
panel's absolute deadline 2026-09-11T22:19:58.910479+00:00 for this continuation.

Keep all 512 cases, all 13 arms, their original source/model hashes, reference R9,
historical R8 and prior best macro-v2 RL, full failure retention, actual billed
time/lower-bound definition and the 5000 seed-cluster bootstrap. Retain every
valid old record including failures; only the precisely inventoried missing
matrix is executed. Cross-segment computation and interrupted work remain
explicit. Fresh128 has not started and is not silently mixed into this panel.

Object traffic remains capped by the user at 100G. Avoid bulk training readbacks;
transfer only the small sidecar, recovery manifest and status initially. Ordinary
incremental sync retains `--s3-no-check-bucket`. The artifact cap bounds growth,
not total lifetime network traffic; retries and duplicate object keys must not be
misrepresented as free transfers or a verified historical traffic balance.

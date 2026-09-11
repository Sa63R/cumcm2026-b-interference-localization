# C fixed-panel recovery sidecar — prepared, not launched

The original fixed panel has 512 requests × 13 arms = **6656** pairs. Its stopped progress reports 2581 complete records, its final supervisor records `disk_free_below_minimum`, return code −15 and successful final sync. Fresh evaluation has not started. `LOCAL_METADATA_CHECK.json` binds the actual local manifest/freeze/plan/status bytes. No original C bulk records were downloaded or inventoried in this subtask; **4075 missing is only the progress-derived estimate**, not a frozen recovery inventory yet.

`recover_panel.py` stays outside `src/` and the frozen evaluator's hashed experiment files. It imports and reuses that exact root's `experiments.q4_rl_evaluate.run_one` and `report_rows`. It never changes old source, model, config or output, uses no network/database, and never loads a policy while preparing. Pure seeded scene construction during validation checks case identity; it performs no simulation actions.

## Before deployment

Review this sidecar and the remaining limitations below. Transfer only this small reviewed sidecar through the approved object store, to the unchanged `q4-rl-memory-v4-eval-20260912` task. Resolve the disk guard and verify no old evaluator is running. Do not download/reupload the bulk records just for recovery. Preparation creates hard links on the same filesystem; it fails instead of silently making a bulk copy across volumes.

From that task root, with the existing CPU Python, prepare a **new** output (command shown for review, not executed):

```sh
python research/q4_rl/recovery_sidecar/recover_panel.py prepare \
  --root "$PWD" \
  --original runs/eval-memory-v4-fixed/evaluation \
  --output runs/eval-memory-v4-fixed-recovery-001/evaluation \
  --max-new-artifact-bytes 1000000000 \
  --stopped-status runs/eval-memory-v4-fixed/supervisor.json \
  --manifest-sha256 a350ea4b54c518472a14bbbaa794396687cea064105bc6b2200b62b6f671cf92 \
  --plan-sha256 cc4ec421fc9c9f86796cbfc2d6768bfe0c9680f8e97a10401702a5a722ba61a4 \
  --stopped-status-sha256 c34c8c8b6967bf49ec9c6fe16e05b2f01d8522f80b49d335712c6f35fddcec7f
```

Preparation verifies current and archived source hashes, protocol, exact specs bytes, all ten model hashes, request order, arms, references and bootstrap convention. Every original record is fully decompressed through gzip CRC/EOF and checked against its expected request/spec/case/row identity, stored audit flags, evaluation totals and denominator fields. Decodable mismatches abort; they are not rerun as convenient missing rows. Complete failures remain retained with their unchanged 360000 penalty. Undecodable bytes are retained in `administrative_records/`, individually hashed and listed as administrative unusable attempts; no policy outcome is inferred from them. The old files remain untouched. An unexpected filename also aborts.

The new `recovery.json` freezes all 6656 pair identities, retained file SHA values, failed-retained count, damaged files and exact missing matrix. Review/read back that small manifest and its SHA **before execution**. `prepare` does not generate a partial performance summary. Geometry/physical audits recorded by the original evaluator are preserved, not rerun by the sidecar; this is an identity/integrity recovery check, not a substitute for the subsequent normal raw audit.

Under a new existing CPU supervisor (10 workers + parent, shared CPU budget ≤50, no GPU, **evaluation-only** minimum free space 18 GiB, original absolute deadline `2026-09-11T22:19:58.910479+00:00`, normal periodic/final object sync), invoke only. See `RESOURCE_AMENDMENT.md`; training is not resumed:

```sh
python research/q4_rl/recovery_sidecar/recover_panel.py execute \
  --root "$PWD" \
  --output runs/eval-memory-v4-fixed-recovery-001/evaluation \
  --recovery-sha256 REVIEWED_PREPARE_OUTPUT_SHA \
  --max-new-artifact-bytes 1000000000
```

Execution rechecks source/model and retained bytes before scheduling. It queues at most 20 missing jobs and uses the original `run_one` for each. New gzip files become visible in `records/` only after closing/fsync and atomic hard-link publication. Incomplete writes remain in `inflight/`. It reports only after the complete original matrix validates again, with unchanged R9, R8 and prior macro-RL references and 5000 paired-seed-cluster replicates (seed 4260911). A complete panel with failures returns 1; failures are not filtered out. Fresh remains a separate panel and must not start after another administrative partial/sync failure.

The explicit **1,000,000,000-byte** cap is frozen during prepare and must match exactly at execute. It accounts for unique new logical file bytes: copied freeze metadata, raw records, partial/inflight data, receipts and reports. Old retained/damaged-record hard links are excluded. The writable output quota deducts **8 MiB** for the new parent run's supervisor/job/resource logs and **4096 bytes** for an administrative-stop record. Parent-run sizes outside `evaluation/` are recorded at prepare/execute and checked around writes without traversing evaluation records. If parent logs exceed their reserve, execution stops too. Every payload write checks its byte count before and after; no write may cross its quota. All three reports and completion metadata are staged under `inflight-reports/`; final report names appear only after all fit and the whole matrix is complete. Cap interruption returns **75**, preserves complete records and partials, and writes `administrative_stop.json`. If a late external-log growth triggers the final check after publication, the stop receipt lists the actual published files/marker rather than denying their existence; exit 75 still does not qualify for promotion.

This byte accounting does not hard-control future external supervisor writes, filesystem allocation blocks or directory overhead; the remaining disk guard covers those costs. Report publication uses additional hard links and consumes no extra payload bytes. Artifact bytes are **not** the separate 100 GB cumulative object-transfer budget: syncing the same inode under different object keys can still transfer duplicate bytes, which the root must account for independently.

## Explicit remaining limits

- This is **one reviewed continuation**, not a general multi-interruption journal. A second execution against the same prepared directory is rejected. If another interruption happens, preserve both waves, completed new records and `inflight/`; a further reviewed continuation must merge them and freeze their hashes before new jobs. Do not restart this output or silently fall back to the original 2581-row snapshot.
- No real C record inventory or remote smoke has been performed yet. Preparation must establish actual retained/damaged/missing counts, and supervisor deployment must retain the original resource envelope. The sidecar adds no independent launch automation or disk cleanup.
- Task computation spans both runs, including interrupted work. `compute.json` separates original supervisor elapsed time, retained/new completed-row CPU and recovery-parent overhead. Row CPU excludes original audit/bound work and discarded in-flight work. The supervisor's cgroup CPU includes other tasks and must not be called task CPU. Archive the new final supervisor/resources and verify final object sync before delivering; no fabricated exact total-CPU claim is made.
- The complete result still needs the ordinary full raw/evidence audit before model recommendation. Neither recovery nor the passing offline fixtures demonstrate policy improvement.

Offline checks: `python -m unittest discover -s research/q4_rl/recovery_sidecar -p test_recover_panel.py -v`. The original 12 fixtures cover preserved failures, exact missing matrix, truncated gzip preservation, wrong identity/spec/source/model/freeze, duplicate pairs, failed final sync, changed hard-linked bytes, partial-report suppression, and missing-only scheduling. Eight cap checks cover frozen-parameter mismatch, exact byte boundary, reserved stop metadata, old-link exclusion, partial-gzip retention, parent logs, staged report failure and truthful post-publication interruption. Fixtures use a fake executor/evaluator and no simulator or trained model; current run evidence is in `CAP_TEST_RESULT.json`.

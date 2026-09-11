# Administrative disk stop, 2026-09-12 03:41 Beijing

Both the frozen memory-v4 evaluation and GAE v5 training were stopped by their
predeclared 20 GiB free-disk guards. Remote PID checks confirmed that their workers
exited, followed by both supervisors and the fixed/fresh sequence launcher.
No restart, deletion, new simulator run or model promotion was performed.

The shared filesystem had about 30.49 GB free at the preceding live process check,
then about 5.87 GB after termination. The two current task directories occupied
about 3.38 GB in total. These observations do not identify another process as the
cause; no other user's task or data was inspected or changed.

| Job | Complete PPO batches | Trained episodes | Reserved but unlearned episodes |
|---|---:|---:|---:|
| G1 h128 MC | 13 | 208 | 16 |
| G1 h128 GAE 0.97 | 13 | 208 | 16 |
| G3 h128 MC | 12 | 192 | 16 |
| G3 h128 GAE 0.97 | 13 | 208 | 16 |

The target remains 32 complete batches. These are administrative endpoints with
unequal exposure, not an equal-update comparison or a selected best checkpoint.
`checkpoint.pt` is each job's last numbered complete checkpoint; `latest.pt` is
the separately preserved resumable state. `verify.py` validated all eight hashes,
strict CPU checkpoint schemas, finite weights, episode/update counts, configuration
equality and exact model/Adam equality between each complete/resume pair. Each
latest retains the next 16 reserved scenario and action seeds. The status JSON is
equal to the latest checkpoint state. Original pending raw journals remain on the
server and in object storage; this small archive does not substitute for them.

The evaluation stopped after 2581 of its 6656 fixed-panel arm/case records. No full
paired summary exists and the fresh panel did not start. Incomplete coverage must
not be reported as a new policy result. Its source/model/specification freeze is
unchanged. Continuing requires an explicit missing-record resume with original
records retained, or another preregistered complete comparison, not partial-case
selection.

Both final supervisor receipts report two successful final sync calls and
`final_sync_ok=true`. The terminal receipts were copied to immutable object keys
after the supervisors exited and read back here; this fixes the usual last-status
write not uploading itself. `supervisor-final.json` SHA256 is
5ac80ca27dd2cf1368c85dd111dd234e40c524552744f0f049c019e03e7e0bdd;
`c-evaluation-supervisor-final.json` SHA256 is
c34c8c8b6967bf49ec9c6fe16e05b2f01d8522f80b49d335712c6f35fddcec7f.
Full training raw data was not downloaded again. This verification establishes
checkpoint integrity and final-sync status, not a new complete raw-data audit.

All exchanges used the approved per-task object prefixes. The installed rclone
rejects `--max-transfer 1Mi` before transfer; the successful bounded receipt copies
used `--max-transfer 1M`, `--s3-no-check-bucket`, one transfer/checker, and CPU0-1.
Their reported uploads were 1.568 and 1.585 kBytes. No files were sent over SSH.

Before resuming, restore adequate disk headroom without touching other tasks,
retain the original training source/configuration and pending transactions, and
subtract recorded cumulative wall time from each 3600-second training allowance.
The old absolute 22:00 UTC job deadline is still an additional limit. Neither this
resource interruption nor the current incomplete comparison establishes a plateau
or theoretical optimality.

# Memory-v4 training readback metadata audit

Passed: all 3,508 object names and sizes match the completed 10,598,718,525-byte readback. All 183 EpisodeJournal indexes were independently rehashed; all 2,928 referenced episode SHA values match the readback manifest by exact object name. No raw episode payload was opened or decompressed. Missing objects, mismatched sizes and orphan episodes: zero.

| Job | Committed batches | BC/PPO episodes | Learned TRAIN seed interval | Retained unlearned episodes | Worker CPU s | Update CPU s |
|---|---:|---:|---|---:|---:|---:|
| g1_h64 | 54 | 256/608 | 8012000–8012863 | 0 | 3150.60 | 1717.58 |
| g3_h64 | 48 | 256/512 | 8012000–8012767 | 16 | 3409.46 | 1547.85 |
| g1_h128 | 40 | 256/384 | 8012000–8012639 | 16 | 2180.65 | 2227.58 |
| g3_h128 | 38 | 256/352 | 8012000–8012607 | 16 | 2493.19 | 2091.22 |

All 180 committed batches contain exactly 16 indexed episodes and three complete epochs. The cumulative minibatch-update counts equal Adam's checkpoint step counters. All four endpoint, latest and BC hashes match their frozen copies; ENDPOINTS.json and the launch configuration match committed Git content. Latest model and optimizer tensors exactly equal each final committed endpoint, so the 48 retained administrative episodes did not enter the learned state. No files were removed.

All four job return codes, the pair return code and supervisor child return code are zero. Historical periodic synchronization recorded TimeoutExpired (27 attempts, 15 successes); this history is preserved. The separate terminal final-sync attempt succeeded. The supervisor records GPU disabled and a 50-core requested/quota budget; the four configured jobs request 8 workers + one update CPU each (36 total). These metadata do not independently prove instantaneous process usage.

Committed totals: 27,066 optimizer steps, 1,143,952 training records; worker CPU 11233.89 s, update CPU 7584.23 s. Worker CPU already contains policy CPU 10345.24 s and posthoc-bound CPU 880.89 s. Retained attempts and unmeasured serialization/sync/parent overhead are excluded. No progress.jsonl exists: the four structured job logs were checked against all immutable progress JSON files instead.

This audit is about preservation, accounting and endpoint identity. TRAIN scenes change across batches and totals differ across jobs; no policy improvement or relative performance is inferred. Raw physics/full-clear replay is outside this metadata-only check.

OBJECT_READBACK SHA256: `2c5e8e8adeaa828a2f80b4c802f7221a3e64995fc47bc7f877dfa3341eed55fc`. Evidence: `evidence.json`; reproduction: run `audit.py` with CPU PyTorch. Only this new review directory was written; no server requests, production edits, checkpoint writes, stage or commit.

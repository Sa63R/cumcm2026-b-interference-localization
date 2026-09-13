# Q3/Q4 official practice continuation through 17:30

User first requested more Q3/Q4 official practice, then said 15:30 cutoff, then explicitly corrected: “继续 其实17:30才结束”. Continue existing batch until Beijing 2026-09-13 17:30. User authorizes Windows guest background scripts and simulator login with credentials already in this task history. No Mac mouse/CUA, no formal runs, no algorithm changes. No subagents needed.

## Current state and commands

All host commands run from `/Users/zephyrr/竞赛/26国赛/数模` using `/opt/homebrew/Caskroom/miniforge/base/bin/python3.13`. Host UTM calls require exec_command `sandbox_permissions=require_escalated` because they access the running VM. Read-only file pulls are safe to retry for Windows sharing locks. Never retry an algorithm request blindly.

This directory, called B below:
`output/q3_q4_official_supplement_20260913_1515`

- Guest live batch: `C:\Users\baiwc\Downloads\Q4V6Lite_20260913\practice_supplement_20260913_1515`
- VM UUID: `755E0E00-7B0E-4845-A9C3-7345C9BC2320`
- Guest root: `C:\Users\baiwc\Downloads\Q4V6Lite_20260913`
- Official simulator: `C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe`, version 1.1, guest HTTP localhost:2026.
- Guest Python: `C:\Users\baiwc\Downloads\Q4Practice\python\python.exe`
- Entries: `run_q3_origin.py` (v3_origin20, initial_channels=20), `run_official.py` (Q4 V6 Lite), same frozen code/manifests as all six formal tests. Entry args only `--robot-id`, `--case-label`, `--output`.
- Existing batch alternates Q3 odd / Q4 even. Final deadline is `2026-09-13T09:29:30Z`, where it stops opening new cases. `target=2000` is a high safety cap, NOT a requested 2000 cases. Final success is `status=completed, phase=deadline_reached`; index is the next unused index and is not another completed case.
- Batch originally completed 52 cases (26 each) at 15:29:34. They were exported and all 52 independently audited. User then extended deadline. Same batch resumed from index 53 around 15:32. Do not restart at index 1 or count the checkpoint twice.
- Most recent interactive task: `Codex-Q4V6Lite_20260913-245802c16c`. Full names in `results/created_tasks.txt`.
- Heartbeat automation id `q3-q4-1730`, interval two minutes, attached to this task. Pause it after final verified delivery. No routine progress messages on successful unchanged/non-actionable checks; notify on meaningful failure, completion or necessary user action.

Read live progress:
```
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 output/q3_q4_official_supplement_20260913_1515/monitor.py
```
This pulls status and completed ledger, saves local copies, appends monitor history, and prints counts, cleared counts, pooled virtual sec/source and entry wall time. If progressing normally and before cutoff, finish the heartbeat quietly. If error or stale, inspect details before any restart. `u.ps` is asynchronous.

IMPORTANT: old `utmctl` transport started crashing around 15:16 with ScriptingBridge `-[SBApplication virtualMachines]: unrecognized selector`. UTM and QEMU remained healthy. B/utm_guest.py uses UTM's documented AppleScript open file/pull/write/execute interface and works. B/launch_desktop.py, B/guest_desktop.py, B/monitor.py and B/fetch_checkpoint.py use the new helper. Do NOT import old utm_guest.py from the earlier helper directory. Old CLI scripts can be copied beside B/utm_guest.py to adopt the new transport.

## Known UI and recovery

Main batch `bulk_supplement.ps1` does all guest UI via SendInput in Windows interactive session, with OCR guards and official sidecar checks. It holds a mutex against duplicate test workers. Only start it again if previous worker has exited and no algorithm Python is running. It skips completed ledger indices, supports resuming a completed result that has not been recorded, and refuses unknown/incomplete assignments.

Expected Windows desktop is 1710x946; maximized official window rect (-8,-8,1718,906). Guest snapshot helpers never send input to Mac. Launch helper with B/launch_desktop.py and old `output/q4_v6_lite_official_20260913/desktop_server.ps1`; use B/guest_desktop.py snapshot/maximize/click. Inspect resulting image with view_image. Stop helper with `quit` before batch resumes.

Practice sidebar x75,y137. As of 16:14, official notice grew by44px. Verified CURRENT practice Q3 start x1433,y375, Q4 x1433,y645. Completion modal confirm remains x1018,y531. Completed-page return is now x1466,y374. Main script has pageOffset=44 for list/ready/start/return, while centered completion modal is unchanged. Never press formal start controls. Batch already guards practice identity, readiness and exact source counts.

Known previous batch failure: cached official server-time verification expired after many runs. UI showed XXXX/00:00, not-entered, connection timeout, no valid case. Merely returning to practice list did not fix. After verifying no active algorithm or unknown request, graceful simulator restart and login fixed it. All six formal records were read-only verified “已上传” at UTC06:16–06:17 and no new formal tests have run.

Proven recovery artifacts are in `output/q4_v6_lite_official_20260913/restart_official.ps1` and `output/q3_q4_official_100each_20260913/recovery_112/login_recovery.ps1`. Inspect files before use and copy local transport as needed. Restart script CloseMainWindow, waits 8s, no force kill; checks no Python and expected exe. After restart maximize guest window. Login helper uses a named pipe, update its name to a fresh unique value, 90s connection wait and 15s ReadLineAsync timeout. Credentials are authorized in user conversation; DO NOT write them to a file or echo them.

Send credentials to login server through a guest PowerShell NamedPipeClientStream via `u.ps`, not `u.call(file,push, \\.\pipe\...)` (that did not deliver ReadLine). Pipe payload is one JSON line with account/password, transient in process only. Coordinates on maximized login window: account805,450; password805,525; login850,632. Read fresh login_status.json timestamp and fresh screenshot to verify. Stop only an exactly identified obsolete helper task if necessary. Preserve all failures and partial data; unknown `/enter` must be resolved with original request id.

Batch resumes with:
```
python B/launch_desktop.py B/bulk_supplement.ps1
```
If UI is still ready for a valid unentered practice case and batch has no assignment/result, fresh verification can justify a `RESUME_READY_<prefix>` marker. Do not invent or reuse old evidence. If UI is final-results page for an already-ledgered run, restore practice list before resume.

## Exports and audits

B/practice_supplement_checkpoint_052 contains immutable first 52 new cases; ZIP transferred 24,065,942 bytes with full manifest/hash verification. `audit_practice.py ... --expected 52` passed all 52, 10,629 accepted actions, Q3 303/303 sources, Q4 350/350. New batch contains these cases plus subsequent ones; checkpoints must not be summed together.

The automatic export waiter (`export_after_finish.ps1`) failed at 15:26 due status.json sharing lock and exited. It is NOT running; do not rely on it. After final completion, manually execute archive script once:
- B/archive_practice_guest.py is already pushed to guest root as `archive_supplement_guest.py`.
- It collects the completed ledger, every completed run/result/journal, official jlog/sidecars, per-run screenshots/OCR, assignment/console, config/status, into `practice_supplement_checkpoint_NNN.zip` with SHA-256 export manifest.
- On status=completed it uses the actual completed ledger count regardless of --limit, avoiding PowerShell array-count ambiguity.
- Start via `u.ps` explicit guest Python path plus script path `--limit N`. This sends no simulator commands.
- Wait for guest `practice_supplement_checkpoint_NNN_status.json`, then run B/fetch_checkpoint.py N. It verifies archive bytes/SHA, ZIP CRC, safe paths, every manifest SHA; extracts beside script. About 24MB/52cases, direct AppleScript transport was much faster than old CLI. Larger transfer may take minutes; poll terminal ≤60s and communicate meaningful transfer state. Keep raw evidence.
- Run B/audit_practice.py B/practice_supplement_checkpoint_NNN --expected N. It independently checks all request-response-state triples, IDs, attempts, costs, coverage/channel certificates, frozen source manifests, official counts/timestamps/jlog SHA. Any failure must be inspected, not discarded or silently counted as success. Auditor expects no retry; if legitimate retries occur, carefully extend verification instead of excluding runs.

Previous COMPLETED 200-case baseline is `output/q3_q4_official_100each_20260913/practice_checkpoint_200`; all 200 audited. Use only this previous ledger PLUS final supplement ledger for cumulative stats. Previous Q3 100cases,1307/1307sources,pooled233.243619sec/source; Q4 100cases,1298/1298sources,436.179555. Q3/Q4 are different problems, no direct same-problem algorithm ranking.

B/build_supplement_report.py has been updated for the final 17:30 deadline and the first52 pause/resumption. Its statistics code combines final supplement with previous200, checks distinct case codes, uses sum virtual / sum cleared, and records official first enter/last exit timestamps. The official exit timestamp deadline comparison is now17:30. Do not claim final completion solely from the first52 snapshot. Copy failed-monitor evidence only as transport metadata, not algorithm failures.

Final user deliverables go to `/Users/zephyrr/竞赛/26国赛/gpt_download`, a NEW clearly named folder and ZIP such as `Q3_Q4_截止1730补充演练_20260913`. Keep previous results intact. Include raw logs/screenshots, audits, source manifest, per-case and combined JSON stats, readable result Markdown, SHA file manifest. Verify copy and ZIP. Final answer: actual new/cumulative counts, all-clear/audit result, key virtual sec/source separately from wall, clickable report/ZIP. Do not infer a speedup from changing random-case distribution.

Stop own host keep-awake processes after verified completion only: pid/command in B/host_keepawake.json and B/host_keepawake_until1730.json; inspect actual ps command before kill. Guest batch releases its own keep-awake and mutex on exit. Keep official simulator and all data. Pause heartbeat `q3-q4-1730` after delivery, preserving prompt/fields. No Git push requested.

## Recovery at 16:13–16:17 (completed index242)

Batch stopped returning to the practice list after recording Q4 case N53N-TZKF-KPSY-ZBGF at index242; all121 per question were complete. New official notice text increased banner height by44px. Fresh process check found no Python or batch worker; screenshot confirmed test ended, normal exit and log saved. Clicked verified return x1466,y373, captured practice list, adjusted only orchestration coordinates with pageOffset=44, and resumed from index243. All evidence in recovery_242. Algorithms unchanged. No simulator restart/login needed.

## Recovery at 16:30–16:43 (next index299)

After 298 complete cases (149each), next practice preparation timed out with XXXX/00:00/not entered and empty commands. No assignment or run directory existed; no Python or batch processes. Official client gracefully restarted at08:40:45 UTC; login submitted08:42:11 UTC, fresh screenshot confirmed practice available. Batch resumed from299, no completed algorithm rerun. Evidence in recovery_299. Same pageOffset44 remains valid. One read-only host status query approval timed out and succeeded on allowed retry.

After login, first restart attempt stopped at checking_list because the page switched to Q4 formal history (cause undetermined). Guard ran before any start click or algorithm instruction. Screenshot confirmed three old formal records uploaded and no available formal slots. Switched explicitly to practice sidebar, verified both practice buttons, and relaunched at16:46 from299. No formal test started. Evidence in recovery_299/post_login_page_guard.json and screenshots.

# First complete GAE v5 transactions

Root launched `q4-rl-gae-v5-20260912` at **2026-09-11 19:27:10 UTC**
(2026-09-12 03:27:10 Beijing), using source `df36f0ce`. These four immutable
`checkpoint-000001.pt` files record the first PPO batch of G1/G3 h128 MC/GAE0.97.
They are startup/update checks, not selected endpoints or performance evidence.

Root ran `verify.py` successfully. `VERIFIED.json` covers checkpoint SHA256,
strict schema/objective, 16 episodes/one PPO batch/zero new BC/no pending batch,
next seed8022016, threads/RNG/lambda, initial BC SHA, finite changed parameters,
and Adam steps matching the update count. All14 parameter tensors changed;
G1 recorded147 steps and G3 recorded138. That does not establish improvement.

Within each schema, first-batch recorded case ratios, full-clear/failed-clear
counts and mean actual time match. Both arms used the same initial policy before
updating. This does not verify every trajectory byte or compare efficacy with
R9/R8; those comparisons require the planned paired evaluation.

Replay needs repository code and the two SHA-bound BC256 files under
`models/initialization`, which are not duplicated here. The verifier writes
`VERIFIED.json` exclusively: use a separate restoration copy with a fresh result
path. Archiving rechecked four checkpoint and two snapshot hashes; no redownload,
training or large tests were performed.

The19:29:42UTC `supervisor.json` snapshot shows42 affinity CPUs, noGPU, a20GiB
guard and two sync attempts with zero completed successes then; it does not prove
complete raw synchronization. `training-pair.json` records learner PIDs/arguments.
PID/thread-affinity fields in VERIFIED are root's launch observations, not results
of the checkpoint verifier. Root handles later progress and the100GB transfer cap.

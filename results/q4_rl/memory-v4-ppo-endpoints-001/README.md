# V4 final complete PPO endpoints

All four training jobs and their wrapper returned zero. Endpoints were selected solely by each final `status.json` batch count, without reading performance to choose a checkpoint. Each checkpoint and final `latest.pt` passed its original strict CPU loader, exact schema/width/configuration checks, and complete model-tensor equality. Each selected checkpoint has no pending batch and contains 256 completed BC episodes plus the PPO updates below.

| Job | Complete checkpoint | Trained episodes | PPO batches | Final wall seconds | Pending episodes |
|---|---|---:|---:|---:|---:|
| G1 h64 | checkpoint-000054.pt | 864 | 38 | 3597.2185 | 0 |
| G3 h64 | checkpoint-000048.pt | 768 | 32 | 3599.6617 | 16 |
| G1 h128 | checkpoint-000040.pt | 640 | 24 | 3599.5308 | 16 |
| G3 h128 | checkpoint-000038.pt | 608 | 22 | 3599.5915 | 16 |

G1 h64 stopped at its wall/deadline check. The other three retain a reserved PPO batch in final `latest.pt` with stop reason `administrative_stop_reserved_batch_retained`. Their final next-seed/attempt cursors advance by the 16 reserved samples, while completed episode/update counters and all model tensors remain equal to the selected complete checkpoint. Those samples are not counted as trained. Both checkpoint and final latest, final statuses, reservations, reasons, and hashes are preserved. No raw episodes were deleted or modified.

`ENDPOINTS.json` summarizes the frozen identities; each job's `MODEL_VERIFIED.json` includes source/configuration checks and the final training diagnostic. Those last-batch T/L values describe different synthetic training scenes and are not a paired evaluation or evidence for model promotion.

The four complete checkpoints were uploaded through the approved object task and downloaded to new `models/eval-memory-v4/*_ppo512.pt` aliases on the verified problem-4 server. Server loading used CPU affinity 1, PyTorch intra/inter-op threads 1, and hidden CUDA. The six previous frozen model hashes were rechecked, and both rule entrypoints import successfully: all 12 base arms are ready. No evaluation or training was started by this task.

`prestage_ppo.json` is the server receipt returned through object storage; `READBACK_VERIFIED.json` verifies it against the local manifest, endpoint hashes, previous six-model receipt, and all 12 declared base specs. This directory contains only models and metadata. Full raw training evidence remains in its original object/local locations.

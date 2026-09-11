# Offline XZ journal prototype

Only `journal.py` and synthetic serialization tests are new. No production
`src/` or `scripts/` file, historical episode or model was changed. No simulator,
real training data, remote process or object-store transfer is used here. The
separate four-episode storage probe motivates XZ3; this prototype makes no new
compression-ratio or policy-performance claim.

## Protocol and compatibility

`EpisodeJournal(path, *, episode_byte_cap, index_byte_cap, stop_check,
prior_entries=())` retains `.append(row)` and the original row object contract.
It is intentionally **not** an automatic substitute for the old arbitrary
`writer` callback or a resume loader. It only creates a fresh attempt; an
existing index or exclusive `.xz-owner` marker refuses reuse.

New episode files use `.json.xz`, XZ preset 3 / CRC64. Their decoded bytes are
exactly `json.dumps([row], ensure_ascii=False, allow_nan=False).encode('utf-8')`,
including original order, separators and float spelling. Rows are not edited.
Every successful append atomically publishes a small `.json.gz` gzip6 index:

```json
{"format":"q4-training-episode-index-v2","episodes":[
  {"path":"attempt-episode-0000.json.xz","seed":1,"codec":"xz",
   "compressed_sha256":"...","compressed_bytes":123,
   "decoded_sha256":"...","decoded_bytes":456}
]}
```

`read_batch` accepts historical gzip-list batches, v1 indexes, and mixed gzip/XZ
v2 indexes. V2 verifies both byte counts/hashes and codec; v1 verifies its stored
compressed SHA. A fresh v2 index can import `prior_entries` from immutable files
in the same directory, verifying and enriching metadata without rewriting old
files. The unchanged production v1 reader explicitly rejects the v2 format.
For an index or historical monolithic file, the caller can pass the trusted
manifest's `expected_sha256`; without an external expected hash, that outer
file has codec integrity checks but no claim of external provenance verification.

Readers reject parent/absolute/drive paths, separators, duplicate references,
symlinks, wrong seed/type, wrong codec, bad hashes/sizes, truncated streams and
trailing/concatenated streams. Output is exposed only after verification. Reads
use 64 KiB chunks, a compressed-input cap, decoded-output cap, and 64 MiB XZ
decoder memory limit. The default decoded cap is 128 MiB **over all episode JSON
bytes returned by a batch**, so larger historical batches require an explicitly
chosen bound. Decoded JSON objects have additional Python memory overhead.
For example, 16 real episodes of 20–41 MiB each exceed the default batch limit;
the defaults are not certified for that workload. Future audit must explicitly
budget the full read or iterate and verify one episode at a time, never truncate
records to make the cap pass. The prototype's write caps are per output file,
not a shared hard limit for four jobs.

## Publication and failure semantics

Episode compression writes to an exclusive `.partial` with a compressed-byte
cap, checking administrative stop between chunks and before flush. Publication
uses an atomic hard link that cannot replace an existing episode, then removes
only this successfully published temporary link. A filesystem without hard-link
support fails closed. The next gzip index is also bounded and fully written/
fsynced before atomic publication. Existing index replacement requires its
previous acknowledged SHA to match; the in-memory cursor advances only after
publication. Directory fsync is used on POSIX. One owner per attempt remains
required; this is not a distributed multi-writer protocol.

Compression, cap, cancellation or publication failure makes this journal fail
closed. It retains `.partial`, complete orphan episodes, and any ready but
unpublished index. A small exclusive administrative JSON receipt records stage,
exception **type only**, acknowledged/observed index hashes and artifact hashes.
It does not copy exception messages, absolute paths or row contents. A hard kill
or full disk can prevent even this receipt; exclusive partial/orphan names still
remain. A failure after index replacement but before acknowledgment is detectable
from the observed index hash, and must be reconciled before any future recovery.

**A capped partial does not preserve the complete original worker row.** It is
administrative evidence only, explicitly ineligible for training updates; this
prototype cannot establish a complete overflow archive. The original row is
serialized with one `json.dumps` allocation before chunked compression, so the
output cap is not an upper bound on serializer peak memory. No failed row may be
silently omitted while the rest of a production batch is updated.

## Required production design before integration

Four jobs need one task-root, process-safe byte-reservation ledger, not four
independent checks of `disk_usage`. Before reserving/dispatching each full batch,
atomically reserve worst-case storage for every episode plus a simultaneous
index/checkpoint publication, while maintaining the shared 20 GiB floor and an
explicit separate checkpoint/log/emergency reserve. The ledger must include
already live reservations, retained failed attempts and other task reservations;
free space can still shrink through external jobs, so runtime disk checks remain.

Do not size reservations from the observed 89.9% compression reduction. A valid
worst-case bound needs bounded requests/candidates/history and a bounded JSON
size, codec overhead, checkpoint size and retained retry count. If that bound
cannot be established, production needs a separately verified **complete**
overflow destination/transfer protocol before accepting a completed worker row.
Network availability and the common 100 G traffic cap cannot be assumed. If no
space reservation is available, stop before dispatching the next batch, retain
pending/optimizer/RNG, and save a checkpoint from the reserved area.

Reservations settle only after durable episode/index publication and provenance
recording; partial bytes remain charged until an explicit audited cleanup. A
recovery owner must reconcile surviving tokens/files after a crash. Reserving
bytes does not itself preallocate disk, and external writers can still exhaust
it. Shared locking, reservation math, complete overflow handling, checkpoints
and production transaction/resume semantics are **not implemented here**.

## Offline tests

```text
python -B -m pytest research/q4_rl/xz_journal_sidecar/test_journal.py -q
```

Fixtures cover multichunk byte-exact roundtrip, input immutability, old gzip/v1 and
mixed v2 reading, v1-reader rejection, hash/size/path/codec/identity rejection,
caps and interruption, compressor errors, no-clobber publication races, atomic
index publication and preservation of old evidence. See `OFFLINE_VERIFIED.json`
for actual results and file hashes. Windows without symlink-creation permission
skips only the real filesystem symlink case; no Linux deployment is implied.

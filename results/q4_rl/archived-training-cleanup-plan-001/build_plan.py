"""Build an approval-only allowlist from completed local readback metadata.

No remote calls, transfers, deletion, or bulk payload hashing. The resulting
allowlist is evidence for approval, not an executable cleanup command.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[3]
SERVER_BASE = "/home/dataset-assist-0/usr/lh/ysh/bwc/shumo"
OBJECT_BASE = "jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo"


def sha(data):
    return hashlib.sha256(data).hexdigest()


SPECS = [
    dict(task="q4-rl-memory-v4-20260912", run="train-memory-v4",
         local="q4-rl-negative-memory/results/q4_rl/server-memory-v4-training-001",
         manifest_sha="2c5e8e8adeaa828a2f80b4c802f7221a3e64995fc47bc7f877dfa3341eed55fc",
         regex=r"(?:g1_h64|g3_h64|g1_h128|g3_h128)/training/batch-\d{6}-attempt-\d{6}-episode-\d{4}\.json\.gz",
         count=2928, size=10445695269),
    dict(task="q4-rl-bundle-v3-20260912", run="train-bundle-v3",
         local="q4-rl-micro-attention/results/q4_rl/server-bundle-v3-training-001",
         manifest_sha="b8d9158c818a1fc4889beb532139b7e17107486812e1943fdad6f01ded3ccba0",
         regex=r"(?:(?:attention_scratch|mlp_scratch|ppo_initialized)/training/batch-\d{6}-attempt-\d{6}-episode-\d{4}|scst_initialized/training/raw/batch-\d{6}-attempt-\d{6}-leg-\d{4})\.json\.gz",
         count=2256, size=6027389302),
]

tasks = []
for spec in SPECS:
    base = WORKSPACE / spec["local"]
    raw_manifest = (base / "OBJECT_READBACK.json").read_bytes()
    assert sha(raw_manifest) == spec["manifest_sha"]
    manifest = json.loads(raw_manifest)
    assert manifest["task"] == spec["task"]
    assert manifest["prefix"] == "runs/" + spec["run"]
    objects = {x["name"]: x for x in manifest["objects"]}
    selected = sorted((x for x in objects.values() if re.fullmatch(spec["regex"], x["name"])), key=lambda x: x["name"])
    assert len(selected) == spec["count"]
    assert sum(x["bytes"] for x in selected) == spec["size"]
    names = {x["name"] for x in selected}
    references, indexes = set(), []
    for name, entry in objects.items():
        ordinary = re.search(r"/batch-\d{6}-attempt-\d{6}\.json\.gz$", name)
        scst = re.fullmatch(r"scst_initialized/training/raw/batch-\d{6}-attempt-\d{6}\.jsonl", name)
        if not (ordinary or scst):
            continue
        data = (base / name).read_bytes()
        assert sha(data) == entry["sha256"]
        rows = (json.loads(gzip.decompress(data))["episodes"] if ordinary else
                [json.loads(line) for line in data.splitlines() if line.strip()])
        for row in rows:
            relative = name.rsplit("/", 1)[0] + "/" + row["path" if ordinary else "file"]
            assert relative in names
            assert relative not in references
            assert objects[relative]["sha256"] == row["sha256"]
            if scst:
                assert objects[relative]["bytes"] == row["bytes"]
            references.add(relative)
        indexes.append({"name": name, "sha256": sha(data), "references": len(rows)})
    assert references == names
    for entry in selected:
        local = base / entry["name"]
        assert local.is_file() and not local.is_symlink()
        assert local.stat().st_size == entry["bytes"]
    supervisor = json.loads((base / "supervisor.json").read_bytes())
    pair = json.loads((base / "training-pair.json").read_bytes())
    assert supervisor["child_returncode"] == 0 and pair["returncode"] == 0
    canonical = "".join(f"{x['sha256']}  {x['bytes']}  {x['name']}\n" for x in selected).encode()
    group_counts = {}
    for x in selected:
        group = x["name"].rsplit("/", 1)[0]
        g = group_counts.setdefault(group, {"files": 0, "bytes": 0})
        g["files"] += 1
        g["bytes"] += x["bytes"]
    tasks.append({
        "task": spec["task"], "run": spec["run"],
        "remote_task_root": SERVER_BASE + "/" + spec["task"],
        "remote_run_root": SERVER_BASE + "/" + spec["task"] + "/runs/" + spec["run"],
        "object_archive_prefix": OBJECT_BASE + "/" + spec["task"] + "/runs/" + spec["run"],
        "local_readback_directory_workspace_relative": spec["local"],
        "OBJECT_READBACK_sha256": spec["manifest_sha"],
        "file_count": len(selected), "logical_payload_bytes": sum(x["bytes"] for x in selected),
        "canonical_allowlist_sha256": sha(canonical),
        "canonical_allowlist_format": "UTF-8, sorted name; each line SHA256 + two spaces + decimal bytes + two spaces + relative name + LF",
        "groups": group_counts, "local_all_files_present_and_sizes_match": True,
        "all_payload_sha_values_match_small_journal_indexes_and_OBJECT_READBACK": True,
        "journal_indexes_rehashed": indexes,
        "historical_terminal": {"pair_returncode": pair["returncode"],
                                "child_returncode": supervisor["child_returncode"],
                                "finished_utc": supervisor["finished_utc"],
                                "final_sync_attempts": supervisor["final_sync_attempts"]},
        "files": selected,
    })

plan = {
    "schema": "q4-archived-training-cleanup-approval-plan-v1",
    "status": "PROPOSAL ONLY: explicit user approval and fresh remote read-only preflight required; nothing deleted",
    "scope": "Only enumerated historical raw training payload files in two completed tasks. Do not delete directories or expand wildcards.",
    "requested_action_after_approval": "Unlink only exact allowlisted ordinary files from the old server run roots, after preflight; retain both archive copies.",
    "total_files": sum(t["file_count"] for t in tasks),
    "total_logical_payload_bytes": sum(t["logical_payload_bytes"] for t in tasks),
    "total_logical_payload_GiB": sum(t["logical_payload_bytes"] for t in tasks)/2**30,
    "exclude_entire_tasks": ["q4-rl-memory-v4-eval-20260912", "q4-rl-gae-v5-20260912"],
    "retain": ["all .pt files, including random/warmstart/latest/checkpoint and final selected models",
               "all source/code/dependencies and source archives", "all journal indexes (.json.gz batch index and SCST .jsonl)",
               "all progress/status/log/config/manifests and evaluation outputs", "all local and object-store backup copies"],
    "fresh_preflight_required_before_action": [
        "Resolve each supplied remote root again: canonical path under the approved shumo root, distinct from current eval/GAE roots; no symlink traversal.",
        "Read process/PID and open-file metadata to verify no old trainer, supervisor, writer, or current job depends on any selected payload; abort on any reference or uncertainty.",
        "Enumerate exact allowlist paths only, verify regular-file status, canonical containment, count and sizes against manifest. Any missing/mismatched/newly modified file requires review, never wildcard expansion.",
        "Reconfirm the two backup manifests and object-prefix availability with metadata only. Existing SHA evidence binds archived bytes; it is not a new content-hash measurement of current remote files.",
        "Record an immutable dry-run result and obtain explicit approval for these exact files. After approved unlink, report actual filesystem blocks/df recovered and recheck disk growth before considering any restart."
    ],
    "limits": [
        "Numbers are summed logical file sizes, not guaranteed reclaimed filesystem blocks; current remote existence, open handles, hard links, compression and concurrent disk consumption were not checked in this local-only task.",
        "The total is 15.342 GiB. Starting from approximately 5.9 GB/GiB leaves only a small margin above the 20 GiB guard; this proposal does not establish enough space to resume 32-episode-batch training.",
        "Administrative and failed training payloads remain preserved in both backups, even when their server copies are proposed for cleanup. No evidence or unsuccessful route is discarded.",
        "No new large hashes, remote operations or object transfers were performed while preparing this proposal."
    ],
    "tasks": tasks, "builder_sha256": sha(Path(__file__).read_bytes()),
}
(OUT/"approval-manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
lines = ["# Approval proposal: archived Q4 training payload cleanup", "",
         "Proposal only. No deletion, upload, download or remote access was performed. Explicit user approval and the manifest's fresh remote preflight remain required.", "",
         "The exact 5,184 files in `approval-manifest.json` total **16,473,084,571 bytes (15.342 GiB)**. "
         "Delete only those ordinary files if later approved; do not delete entire directories or expand filesystem globs.", "",
         "| Completed task | Payload group | Files | Logical bytes |", "|---|---|---:|---:|"]
for t in tasks:
    for group, g in t["groups"].items():
        lines.append(f"| {t['task']} | `{group}` | {g['files']} | {g['bytes']:,} |")
lines += ["", "Remote task roots, supplied and previously checked by the main task:", ""]
lines += [f"- `{t['remote_task_root']}`" for t in tasks]
lines += ["", "Each allowlist path is relative to its task's `remote_run_root`; `object_archive_prefix` plus the same relative path locates the object backup. "
          "Local backup folders, complete readback hashes, per-file SHA/size, and exact allowlist hashes are embedded. "
          "This check rehashed only 324 small indexes (183 memory + 72 bundle episode + 69 SCST), joined every one of the 5,184 expected payload hashes to the complete readback records, and checked local existence and sizes. "
          "No archived payload was bulk rehashed.", "",
          "Keep **all models/checkpoints, code, indexes, logs, configurations, source archives and evaluation outputs**. "
          "Exclude the entire current `q4-rl-memory-v4-eval-20260912` and `q4-rl-gae-v5-20260912` tasks, and all other users/tasks. "
          "Keep both local and object-store archives, including failed and administratively interrupted attempts.", "",
          "Before any deletion: freshly resolve approved roots, verify no symlinks or live PID/open-file consumers, and compare exact relative filenames and sizes against the allowlist. "
          "Any discrepancy must stop the proposed cleanup. These are historical archive hashes, not fresh remote-content hashes. "
          "Approval must cover the exact manifest, not a blanket directory removal.", "",
          "**Space limit:** 15.342 GiB is logical payload size; actual reclaimed blocks may differ. From roughly 5.9 GB/GiB free this would leave only a narrow margin over the 20 GiB guard, "
          "and concurrent shared-disk growth remains uncontrolled. This proposal does **not** establish enough space to resume 32-episode-batch training. "
          "Recheck actual free space and growth after any approved cleanup before considering a restart.", "",
          f"Manifest SHA256: `{sha((OUT/'approval-manifest.json').read_bytes())}`."]
(OUT/"README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print(json.dumps({"files":plan["total_files"], "bytes":plan["total_logical_payload_bytes"],
                  "GiB":plan["total_logical_payload_GiB"],
                  "manifest_sha256":sha((OUT/"approval-manifest.json").read_bytes())},indent=2))

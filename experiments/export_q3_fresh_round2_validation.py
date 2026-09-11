"""Prepare a disjoint second-round validation set after policy freezing.

No database access occurs on import. This module does not fit a prior, select
thresholds, or run a strategy. It calls the unchanged bounded-label witness
reconstruction used in round one, with a new deterministic selection salt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.export_q3_fresh_validation import export


SELECTION_SALT = "fresh-q3-v2:"
DEFAULT_EXCLUDE_REPORT = ROOT / "results" / "q3_fresh" / "validation" / "results.json"


def _exclusions(report_path: Path) -> tuple[list[str], str]:
    raw = report_path.read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    # Evaluated round-one results wrap the original export under manifest;
    # accept a direct export too, so an unwrapped manifest remains usable.
    metadata = report.get("manifest", {}).get("metadata")
    if metadata is None:
        metadata = report.get("metadata", {})
    evidence = metadata.get("evidence_sha256") if isinstance(metadata, dict) else None
    if (not isinstance(evidence, list) or len(evidence) != 12 or
            any(not isinstance(item, str) or len(item) != 64 or
                any(c not in "0123456789abcdef" for c in item) for item in evidence) or
            len(set(evidence)) != 12):
        raise ValueError("Exclusion report must contain 12 unique lowercase evidence SHA256 values in manifest.metadata.evidence_sha256 or metadata.evidence_sha256")
    return evidence, hashlib.sha256(raw).hexdigest()


def export_round2(database: Path, count: int = 12,
                  exclude_report: Path = DEFAULT_EXCLUDE_REPORT) -> dict:
    """Select complete original groups by SHA256('fresh-q3-v2:' + case_code).

    The 12 first-round evidence hashes are removed BEFORE ranking. Each
    selected original case produces exactly two radius variants, at 0.25 and
    0.75 of its history-compatible interval. If any selected original cannot
    be reconstructed, fail explicitly rather than silently replacing it with
    a later-ranked case or exporting fewer groups.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("Requested validation group count must be a positive integer")
    excluded, report_hash = _exclusions(Path(exclude_report))
    result = export(Path(database), count, selection_salt=SELECTION_SALT,
                    excluded_evidence=excluded, case_prefix="validation2")
    metadata = result["metadata"]
    if metadata["selected_episodes"] != count:
        raise ValueError(f"Only {metadata['selected_episodes']} eligible new groups; {count} requested")
    if metadata["reconstructed_episodes"] != count or metadata["failures"]:
        raise ValueError(f"Only {metadata['reconstructed_episodes']} of {count} selected groups reconstructed; no validation export accepted")
    new_hashes = metadata["evidence_sha256"]
    if len(set(new_hashes)) != count or set(new_hashes) & set(excluded):
        raise ValueError("New validation groups are duplicated or overlap first-round evidence")
    if len(result["cases"]) != 2 * count:
        raise ValueError("Every original validation group must have two radius variants")
    metadata.update(
        validation_round=2,
        selection_rule="Sort by ascending SHA256(('fresh-q3-v2:' + case_code).encode('utf-8')); exclude round-one evidence before ranking; take first count original groups",
        excluded_report_sha256=report_hash,
        excluded_original_group_count=len(excluded),
        new_evidence_sha256=list(new_hashes),
        original_group_count=count,
        reconstructed_case_count=2 * count,
        usage="Validation only after policy freezing; never fit priors, error models, or thresholds to these reconstructions",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--exclude-report", type=Path, default=DEFAULT_EXCLUDE_REPORT)
    args = parser.parse_args(argv)
    result = export_round2(args.database, args.count, args.exclude_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result["metadata"][key] for key in (
        "validation_round", "original_group_count", "reconstructed_case_count",
        "excluded_original_group_count", "new_evidence_sha256", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Register GUI-verified official practice totals without changing solver code."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["problem", "case_code", "variant", "source_total", "cleared_count",
              "clearance_ratio", "virtual_time_s", "average_clear_time_s",
              "program_runtime_s", "runtime_source", "search_completed",
              "summary_path", "summary_sha256"]


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def register(args):
    raw = args.summary.read_bytes()
    summary = json.loads(raw.decode("utf-8-sig"))
    if (summary.get("data_origin") != "simulator_http_session"
            or summary.get("declared_mode") != "practice"):
        raise ValueError("Only practice HTTP-session summaries are accepted; local research and formal results are excluded")
    state = summary.get("state", {})
    if state.get("session") != "exited" or summary.get("pending_request") is not None:
        raise ValueError("Practice session must have an accepted exit and no pending action")
    if summary.get("problem") not in (3, 4):
        raise ValueError("Summary problem must be 3 or 4")
    if not args.case_code.strip() or not all(c.isprintable() for c in args.case_code):
        raise ValueError("Case code must be a nonempty printable string")
    if summary.get("case_code") and summary["case_code"] != args.case_code:
        raise ValueError("Case code does not match the session summary")
    total = args.source_total
    if isinstance(total, bool) or not isinstance(total, int) or not 10 <= total <= 16:
        raise ValueError("Official GUI source total must be an integer from 10 to 16")
    cleared = state.get("cleared_count")
    if isinstance(cleared, bool) or not isinstance(cleared, int) or not 0 <= cleared <= total:
        raise ValueError("Cleared count must be an integer from 0 to the official source total")
    virtual = nonnegative(state.get("virtual_time_s"), "Virtual time")
    runtime = args.runtime if args.runtime is not None else summary.get("program_wall_time_s")
    if runtime is not None:
        runtime = nonnegative(runtime, "Program runtime")
    runtime_source = ("official_gui_user_transcribed" if args.runtime is not None else
                      "local_program_wall_time" if runtime is not None else "not_recorded")
    summary_digest = hashlib.sha256(raw).hexdigest()
    directory = args.output_dir
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(directory.glob("practice-*.json"))]
    if any(r["case_code"] == args.case_code for r in records):
        raise ValueError("This practice case code is already registered")
    if any(r["summary_sha256"] == summary_digest for r in records):
        raise ValueError("This session summary is already registered")
    key = hashlib.sha256(args.case_code.encode("utf-8")).hexdigest()[:20]
    archived = Path("evidence") / f"{summary_digest}.json"
    record = {
        "data_origin": "registered_official_practice",
        "problem": summary["problem"], "case_code": args.case_code,
        "variant": summary.get("variant", "unknown"),
        "source_total": total, "source_total_source": "official_gui_user_transcribed",
        "cleared_count": cleared, "clearance_ratio": cleared / total,
        "virtual_time_s": virtual,
        "average_clear_time_s": virtual / cleared if cleared else None,
        "program_runtime_s": runtime, "runtime_source": runtime_source,
        "search_completed": summary.get("completed") is True,
        "summary_path": archived.as_posix(), "summary_sha256": summary_digest,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "gui_verified_by_program": False,
        "note": "GUI total and optional runtime are user-transcribed; HTTP evidence alone cannot authenticate GUI mode or official origin",
    }
    (directory / "evidence").mkdir(parents=True, exist_ok=True)
    evidence = directory / archived
    if evidence.exists():
        if evidence.read_bytes() != raw:
            raise ValueError("Archived summary differs; preserve evidence and inspect the directory")
    else:
        with evidence.open("xb") as stream:
            stream.write(raw)
    destination = directory / f"practice-{key}.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    records.append(record)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for item in sorted(records, key=lambda r: (r["problem"], r["case_code"])):
                row = dict(item)
                if row["average_clear_time_s"] is None:
                    row["average_clear_time_s"] = "undefined"
                writer.writerow(row)
        temporary.replace(directory / "总表.csv")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"record": str(destination), "table": str(directory / "总表.csv"),
            "clearance_ratio": record["clearance_ratio"],
            "average_clear_time_s": record["average_clear_time_s"],
            "program_runtime_s": runtime, "runtime_source": runtime_source}


def main(argv=None):
    parser = argparse.ArgumentParser(description="登记真实官方演练并自动计算清除比例与平均定位清除时间")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-total", type=int, required=True, help="演练结束后官方 GUI 显示的真实源总数")
    parser.add_argument("--case-code", required=True)
    parser.add_argument("--runtime", type=float, help="可选：官方 GUI 程序运行秒数；否则保留本地墙钟时间并标注来源")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "practice_registered")
    args = parser.parse_args(argv)
    try:
        result = register(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

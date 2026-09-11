"""Register official practice totals from simulator result files or the GUI."""

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
              "summary_path", "summary_sha256", "source_total_source",
              "official_result_path", "official_result_sha256"]


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def read_official_result(path, summary, *, raw_journal=None):
    raw = path.read_bytes()
    result = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError("Official result JSON must be an object")
    problem = result.get("problem_no")
    if isinstance(problem, bool) or not isinstance(problem, int) or problem != summary["problem"]:
        raise ValueError("Official result problem_no does not match the session")
    values = [result.get(key) for key in ("jammer_count", "omnidirectional_jammer_count",
                                         "directional_jammer_count")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Official result source counts must be nonnegative integers")
    total, omni, directional = values
    if not 10 <= total <= 16 or omni + directional != total:
        raise ValueError("Official result source counts must sum to a total from 10 to 16")
    if problem == 3 and (omni != total or directional != 0):
        raise ValueError("Problem 3 requires only omnidirectional sources")
    # Q4 practice can contain only directional sources; a nonzero omni count
    # is not required (observed in the simulator's completed practice results).
    if problem == 4 and directional == 0:
        raise ValueError("Problem 4 requires at least one directional source")
    code = result.get("case_code")
    if not isinstance(code, str) or not code.strip() or not code.isprintable():
        raise ValueError("Official result case code must be nonempty and printable")
    from practice_control.evidence_time import match_session_time
    time_evidence = match_session_time(summary, result, raw_journal=raw_journal)
    return result, raw, time_evidence


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
    case_code = getattr(args, "case_code", None)
    total = getattr(args, "source_total", None)
    official_path = getattr(args, "official_result_json", None)
    official_result, official_raw, time_evidence = None, None, None
    raw_journal = None
    source_total_source = "official_gui_user_transcribed"
    if official_path is not None:
        journal_path = args.summary.parent / "requests.jsonl"
        if journal_path.is_file():
            raw_journal = journal_path.read_bytes()
        official_result, official_raw, time_evidence = read_official_result(
            official_path, summary, raw_journal=raw_journal)
        if case_code is not None and case_code != official_result["case_code"]:
            raise ValueError("Explicit case code does not match the official result file")
        if total is not None and total != official_result["jammer_count"]:
            raise ValueError("Explicit source total does not match the official result file")
        case_code, total = official_result["case_code"], official_result["jammer_count"]
        source_total_source = "official_simulator_result_file"
    if not isinstance(case_code, str) or not case_code.strip() or not case_code.isprintable():
        raise ValueError("Case code must be a nonempty printable string")
    if summary.get("case_code") and summary["case_code"] != case_code:
        raise ValueError("Case code does not match the session summary")
    if isinstance(total, bool) or not isinstance(total, int) or not 10 <= total <= 16:
        raise ValueError("Official source total must be an integer from 10 to 16")
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
    if any(r["case_code"] == case_code for r in records):
        raise ValueError("This practice case code is already registered")
    if any(r["summary_sha256"] == summary_digest for r in records):
        raise ValueError("This session summary is already registered")
    official_digest = hashlib.sha256(official_raw).hexdigest() if official_raw is not None else None
    if official_digest and any(r.get("official_result_sha256") == official_digest for r in records):
        raise ValueError("This official result file is already registered")
    key = hashlib.sha256(case_code.encode("utf-8")).hexdigest()[:20]
    archived = Path("evidence") / f"{summary_digest}.json"
    record = {
        "data_origin": "registered_official_practice",
        "problem": summary["problem"], "case_code": case_code,
        "variant": summary.get("variant", "unknown"),
        "source_total": total, "source_total_source": source_total_source,
        "cleared_count": cleared, "clearance_ratio": cleared / total,
        "virtual_time_s": virtual,
        "average_clear_time_s": virtual / cleared if cleared else None,
        "program_runtime_s": runtime, "runtime_source": runtime_source,
        "search_completed": summary.get("completed") is True,
        "summary_path": archived.as_posix(), "summary_sha256": summary_digest,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "gui_verified_by_program": False,
        "note": ("Source total and case code read from the archived simulator result file; timestamps and type counts were checked against the session. The result window is not program runtime."
                 if official_result is not None else
                 "GUI total and optional runtime are user-transcribed; HTTP evidence alone cannot authenticate GUI mode or official origin"),
    }
    if official_result is not None:
        result_archive = Path("evidence") / f"{official_digest}.result.json"
        record.update(official_result_path=result_archive.as_posix(), official_result_sha256=official_digest,
                      official_result_original_name=official_path.name,
                      official_result_window_started_at_utc=official_result["window_started_at_utc"],
                      official_result_ended_at_utc=official_result["ended_at_utc"],
                      omnidirectional_source_total=official_result["omnidirectional_jammer_count"],
                      directional_source_total=official_result["directional_jammer_count"],
                      official_result_session_time_matched=True,
                      session_time_evidence=time_evidence)
        if time_evidence["basis"] == "simulator_http_enter":
            journal_archive = Path("evidence") / f"{hashlib.sha256(raw_journal).hexdigest()}.requests.jsonl"
            record["session_time_journal_path"] = journal_archive.as_posix()
    (directory / "evidence").mkdir(parents=True, exist_ok=True)
    evidence = directory / archived
    if evidence.exists():
        if evidence.read_bytes() != raw:
            raise ValueError("Archived summary differs; preserve evidence and inspect the directory")
    else:
        with evidence.open("xb") as stream:
            stream.write(raw)
    if official_result is not None:
        result_evidence = directory / result_archive
        if result_evidence.exists():
            if result_evidence.read_bytes() != official_raw:
                raise ValueError("Archived official result differs; inspect the evidence directory")
        else:
            with result_evidence.open("xb") as stream:
                stream.write(official_raw)
        if time_evidence["basis"] == "simulator_http_enter":
            journal_evidence = directory / journal_archive
            if journal_evidence.exists():
                if journal_evidence.read_bytes() != raw_journal:
                    raise ValueError("Archived request journal differs; preserve evidence")
            else:
                with journal_evidence.open("xb") as stream:
                    stream.write(raw_journal)
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
            "case_code": case_code, "source_total": total, "source_total_source": source_total_source,
            "clearance_ratio": record["clearance_ratio"],
            "average_clear_time_s": record["average_clear_time_s"],
            "program_runtime_s": runtime, "runtime_source": runtime_source}


def main(argv=None):
    parser = argparse.ArgumentParser(description="登记真实官方演练并自动计算清除比例与平均定位清除时间")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-total", type=int, help="官方 GUI 真实源总数；提供官方 result JSON 时可省略")
    parser.add_argument("--case-code", help="真实案例编码；提供官方 result JSON 时可省略")
    parser.add_argument("--official-result-json", type=Path, help="模拟器结束后自动保存的原始 .result.json")
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

"""Portable evidence ledger and table export; never manufacture formal results."""

import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile


EXTRA_FIELDS = ["summary_sha256", "requests_sha256", "result_source"]
EXPECTED_SLOTS = {(str(p), str(s)) for p in (3, 4) for s in (1, 2, 3)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ledger(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    required = {"problem", "slot", "case_code", "cleared_count", "virtual_time_s",
                "average_clear_time_s", "program_runtime_s", "runtime_source",
                "summary_path", "official_log", "sha256", "uploaded_confirmed"}
    if not fields or not required.issubset(fields):
        raise ValueError("Ledger is missing required columns")
    if len(rows) != 6 or {(r["problem"], r["slot"]) for r in rows} != EXPECTED_SLOTS:
        raise ValueError("Ledger must contain the six unique Q3/Q4 slots")
    return fields + [f for f in EXTRA_FIELDS if f not in fields], rows


def write_ledger(path, fields, rows):
    # A unique same-directory temporary file also makes interrupted previous
    # writes harmless. The authoritative ledger changes in one replace.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evidence_path(ledger, value):
    path = Path(value)
    return path if path.is_absolute() else ledger.parent / path


def finite_nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid {name}: must be finite and nonnegative")
    return number


def validate_count(value):
    number = finite_nonnegative(value, "cleared count")
    if not number.is_integer() or number > 16:
        raise ValueError("Cleared count must be an integer from 0 to 16")
    return int(number)


def validate_report(report, problem, case_code):
    if report.get("data_origin") != "simulator_http_session" or report.get("declared_mode") != "formal":
        raise ValueError("Formal table requires a formal HTTP-session summary, never synthetic results")
    if report.get("problem") != int(problem):
        raise ValueError("Problem number does not match summary")
    if report.get("case_code") and report["case_code"] != case_code:
        raise ValueError("Case code does not match summary")


def register(args):
    """Archive supplied evidence byte-for-byte; GUI assertions remain explicit."""
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    validate_report(report, args.problem, args.case_code)
    if not args.case_code.strip() or any(ord(c) < 32 for c in args.case_code):
        raise ValueError("Case code must be a nonempty printable string")
    gui_count = getattr(args, "cleared_count", None)
    gui_time = getattr(args, "virtual_time", None)
    if (gui_count is None) != (gui_time is None):
        raise ValueError("Provide --cleared-count and --virtual-time together")
    if report.get("pending_request") is not None and gui_count is None:
        raise ValueError("Unresolved action: transcribe --cleared-count and --virtual-time from the official GUI")
    state = report.get("state", {})
    cleared = validate_count(state.get("cleared_count") if gui_count is None else gui_count)
    virtual = finite_nonnegative(state.get("virtual_time_s") if gui_time is None else gui_time, "virtual time")
    runtime = finite_nonnegative(args.runtime, "official GUI runtime")
    if not args.official_log.is_file() or not 0 < args.official_log.stat().st_size <= 2_000_000:
        raise ValueError("Original encrypted log must exist, be nonempty, and be <= 2 MB")
    request_log = args.summary.parent / "requests.jsonl"
    if not request_log.is_file() or not request_log.stat().st_size:
        raise ValueError("The summary's requests.jsonl evidence is missing or empty")
    fields, rows = read_ledger(args.ledger)
    row = next(r for r in rows if r["problem"] == str(args.problem) and r["slot"] == str(args.slot))
    if row["case_code"]:
        raise ValueError("Slot already registered; inspect existing evidence before editing the ledger")
    if any(r["case_code"] == args.case_code for r in rows):
        raise ValueError("Case code already registered")
    log_digest = digest(args.official_log)
    if any(r["sha256"] == log_digest for r in rows):
        raise ValueError("This log is already associated with a different formal slot")
    target = args.ledger.parent / "正式日志" / f"问题{args.problem}" / args.official_log.name
    archive = args.ledger.parent / "正式记录" / f"问题{args.problem}" / f"slot-{args.slot}"
    archive.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in ((args.official_log, target),
                                (args.summary, archive / "summary.json"),
                                (request_log, archive / "requests.jsonl")):
        # An interrupted registration can resume only when existing evidence is
        # byte-identical. Original official filenames are never changed.
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f"Evidence filename collision: {destination}")
        else:
            with destination.open("xb") as stream:
                stream.write(source.read_bytes())
    row.update({
        "case_code": args.case_code, "cleared_count": str(cleared),
        "virtual_time_s": str(virtual),
        "average_clear_time_s": str(virtual / cleared) if cleared else "undefined",
        "program_runtime_s": str(runtime), "runtime_source": "official_gui_user_transcribed",
        "summary_path": (archive / "summary.json").relative_to(args.ledger.parent).as_posix(),
        "official_log": target.relative_to(args.ledger.parent).as_posix(),
        "sha256": log_digest, "summary_sha256": digest(archive / "summary.json"),
        "requests_sha256": digest(archive / "requests.jsonl"),
        "result_source": "official_gui_user_transcribed" if gui_count is not None else "accepted_http_state",
        "uploaded_confirmed": "yes" if args.uploaded else "pending",
    })
    write_ledger(args.ledger, fields, rows)
    export_tables(args)
    print(f"Registered Q{args.problem} slot {args.slot}; original log preserved; SHA-256 {log_digest}")
    return 0


def audit_findings(ledger):
    _, rows = read_ledger(ledger)
    missing, seen_codes, seen_hashes = [], set(), set()
    for row in rows:
        label = f"Q{row['problem']}-{row['slot']}"
        if not row["case_code"]:
            missing.append(f"{label}: official case not registered")
            continue
        if row["case_code"] in seen_codes or row.get("sha256") in seen_hashes:
            missing.append(f"{label}: duplicate case code or log")
        seen_codes.add(row["case_code"])
        seen_hashes.add(row.get("sha256"))
        log = evidence_path(ledger, row["official_log"])
        if not log.is_file():
            missing.append(f"{label}: official log missing")
        elif digest(log) != row.get("sha256"):
            missing.append(f"{label}: official log changed")
        elif not 0 < log.stat().st_size <= 2_000_000:
            missing.append(f"{label}: official log size invalid")
        summary = evidence_path(ledger, row["summary_path"])
        try:
            if not summary.is_file() or digest(summary) != row.get("summary_sha256"):
                raise ValueError("summary missing or changed")
            requests = summary.parent / "requests.jsonl"
            if not requests.is_file() or digest(requests) != row.get("requests_sha256"):
                raise ValueError("requests journal missing or changed")
            report = json.loads(summary.read_text(encoding="utf-8"))
            validate_report(report, row["problem"], row["case_code"])
            count = validate_count(row["cleared_count"])
            virtual = finite_nonnegative(row["virtual_time_s"], "virtual time")
            finite_nonnegative(row["program_runtime_s"], "GUI runtime")
            if count:
                average = finite_nonnegative(row["average_clear_time_s"], "average clear time")
                if not math.isclose(average, virtual / count, rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("average does not equal virtual time / cleared count")
            elif row["average_clear_time_s"] != "undefined":
                raise ValueError("zero-clear average must be undefined")
            if row.get("runtime_source") != "official_gui_user_transcribed":
                raise ValueError("official runtime has not been transcribed")
            if row.get("result_source") == "accepted_http_state":
                state = report.get("state", {})
                if (report.get("pending_request") is not None
                        or count != state.get("cleared_count")
                        or virtual != state.get("virtual_time_s")):
                    raise ValueError("table differs from accepted HTTP state")
            elif row.get("result_source") != "official_gui_user_transcribed":
                raise ValueError("result source is missing")
        except (OSError, ValueError, TypeError) as exc:
            missing.append(f"{label}: {exc}")
        if row.get("uploaded_confirmed") != "yes":
            missing.append(f"{label}: upload not confirmed in official GUI")
    return rows, missing


def audit(args):
    _, missing = audit_findings(args.ledger)
    print(json.dumps({"ready_for_submission": not missing,
                      "scope": "Evidence consistency only; official log decryption and GUI verification are manual",
                      "missing": missing}, ensure_ascii=False, indent=2))
    return 1 if missing else 0


def confirm_upload(args):
    fields, rows = read_ledger(args.ledger)
    row = next(r for r in rows if r["problem"] == str(args.problem) and r["slot"] == str(args.slot))
    if not row["case_code"] or row["case_code"] != args.case_code:
        raise ValueError("Provide the exact registered case code for this slot")
    log = evidence_path(args.ledger, row["official_log"])
    if not log.is_file() or digest(log) != row["sha256"]:
        raise ValueError("Official log missing or changed")
    row["uploaded_confirmed"] = "yes"
    write_ledger(args.ledger, fields, rows)
    export_tables(args)
    print(f"Recorded user confirmation of official GUI upload: {args.case_code}")
    return 0


def export_tables(args):
    rows, missing = audit_findings(args.ledger)
    directory = args.ledger.parent
    lines = ["# 正式测试结果表（由登记表自动生成）", "",
             "数据仅来自已登记的正式会话及人工核对的官方界面；空缺不会使用本地研究数据替代。",
             "审计仅检查材料一致性，无法解密官方日志或代替官方界面核验。", "",
             f"当前材料审计：{'通过' if not missing else '未完成，共 ' + str(len(missing)) + ' 项待处理'}。", ""]
    exported = []
    for problem in (3, 4):
        lines += [f"## 问题 {problem}", "",
                  "| 次序 | 测试案例编码 | 清除干扰源个数 | 平均定位清除时间（秒） | 程序运行时间（秒） | 上传确认 |",
                  "|---|---|---:|---:|---:|---|"]
        for row in sorted((r for r in rows if r["problem"] == str(problem)), key=lambda r: int(r["slot"])):
            registered = bool(row["case_code"])
            code = row["case_code"].replace("|", "\\|") if registered else "待正式测试"
            count = row["cleared_count"] if registered else "—"
            avg = ("未定义" if row["average_clear_time_s"] == "undefined" else row["average_clear_time_s"]) if registered else "—"
            runtime = row["program_runtime_s"] if registered else "—"
            status = "已人工确认" if row["uploaded_confirmed"] == "yes" else "待确认"
            lines.append(f"| {row['slot']} | {code} | {count} | {avg} | {runtime} | {status} |")
            exported.append({k: row[k] for k in ("problem", "slot", "case_code", "cleared_count", "average_clear_time_s", "program_runtime_s")})
        lines.append("")
    lines += ["平均定位清除时间 = 总虚拟耗时 / 清除个数；清除 0 个时未定义。", "",
              "正式测试不公开真实总数，表中不生成无法核实的清除比例。", ""]
    (directory / "正式结果表.md").write_text("\n".join(lines), encoding="utf-8")
    with (directory / "正式结果表.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(f"Exported {directory / '正式结果表.md'} and CSV; missing results remain blank")
    return 0

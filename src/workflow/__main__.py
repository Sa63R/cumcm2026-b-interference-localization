"""Run the solver in an already-open simulator session; register GUI evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from simulator_client import SimulatorClient
from simulator_client.errors import SimulatorError
from .evidence import audit, confirm_upload, export_tables, register


def revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def working_tree_dirty():
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args):
    from strategies import run_search

    if not args.robot_id:
        raise ValueError("Provide --robot-id or CUMCM_ROBOT_ID; do not provide your password")
    if args.max_actions < 2:
        raise ValueError("--max-actions must be at least 2")
    variant = args.variant or ("triangular" if args.problem == 4 else "efficient")
    active_policy = getattr(args, "active_policy", "center")
    if variant == "triangular" and args.problem != 4:
        raise ValueError("triangular variant is only available for problem 4")
    if variant == "efficient" and (args.problem != 3 or active_policy != "center"):
        raise ValueError("efficient requires problem 3 and center policy")
    rollout_config_path = getattr(args, "rollout_config", None)
    rollout_options = {}
    if variant == "rollout":
        if args.problem != 3 or active_policy != "center":
            raise ValueError("rollout requires problem 3 and center policy")
        from dataclasses import asdict
        from strategies.rollout import RolloutConfig
        values = (json.loads(rollout_config_path.read_text(encoding="utf-8-sig"))
                  if rollout_config_path else None)
        rollout_options["rollout_config"] = asdict(RolloutConfig.parse(values))
    elif rollout_config_path is not None:
        raise ValueError("--rollout-config requires --variant rollout")
    destination = args.output or Path("results/sessions") / args.mode / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "data_origin": "simulator_http_session",
        "declared_mode": args.mode,
        "problem": args.problem,
        "case_code": args.case_code,
        "code_revision": revision(),
        "working_tree_dirty": working_tree_dirty(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "active_policy": active_policy,
        "gui_mode_verified_by_api": False,
        "official_result_verified": False,
    }
    error = None
    start = time.monotonic()
    try:
        with SimulatorClient(args.robot_id, base_url=args.base_url,
                             log_path=destination / "requests.jsonl") as client:
            try:
                result = run_search(client, problem=args.problem, variant=variant,
                                    max_actions=args.max_actions, active_policy=active_policy,
                                    **rollout_options)
                report["search"] = result.as_dict()
                # The strategy returns a report even on deadlines, unresolved
                # targets and protocol errors. A returned object is not success.
                if result.error or result.exit_error:
                    error = result.error or result.exit_error
                elif not result.completion_certified_under_model:
                    error = f"SearchIncomplete: {result.completion_reason}"
            except KeyboardInterrupt:
                error = "KeyboardInterrupt: interrupted by user"
            except (SimulatorError, ValueError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if client.state.session == "active" and client.pending_request is None:
                    try:
                        client.exit()
                    except (SimulatorError, OSError, ValueError) as exc:
                        error = error or f"{type(exc).__name__}: {exc}"
                report["state"] = client.state.snapshot()
                report["pending_request"] = client.pending_request
    except (OSError, SimulatorError, ValueError) as exc:
        error = error or f"{type(exc).__name__}: {exc}"
    report["program_wall_time_s"] = time.monotonic() - start
    if report.get("pending_request") is not None:
        error = error or "OutcomeUnknown: inspect the simulator GUI and requests.jsonl"
    if report.get("state", {}).get("session") != "exited":
        error = error or "SessionIncomplete: no accepted exit response"
    report["completed"] = error is None
    report["error"] = error
    report["runtime_note"] = "Local elapsed time; use official GUI runtime for final Table 1 when available"
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    search = report.get("search", {})
    print(json.dumps({"summary": str(destination / "summary.json"), "error": error,
                      "completed": report["completed"],
                      "completion_reason": search.get("completion_reason"),
                      "cleared_count": report.get("state", {}).get("cleared_count"),
                      "virtual_time_s": report.get("state", {}).get("virtual_time_s")},
                     ensure_ascii=False, indent=2))
    return 1 if error else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="B题策略执行与正式材料登记（模拟器界面需先就绪）")
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run", help="执行一次已在界面启动的演练或正式测试")
    execute.add_argument("--problem", type=int, choices=(3, 4), required=True)
    execute.add_argument("--mode", choices=("practice", "formal"), required=True)
    execute.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    execute.add_argument("--case-code", default="")
    execute.add_argument("--base-url", default="http://127.0.0.1:2026")
    execute.add_argument("--variant", choices=("baseline", "adaptive", "deferred", "triangular", "efficient", "rollout"),
                         help="默认问题3为efficient、问题4为triangular；rollout为问题3实验性前瞻策略")
    execute.add_argument("--rollout-config", type=Path,
                         help="rollout参数JSON对象；仅适用于问题3 --variant rollout")
    execute.add_argument("--active-policy", choices=("center", "minimax"), default="center",
                         help="问题3主动选点策略；问题4始终使用center几何启发式")
    execute.add_argument("--max-actions", type=int, default=20000)
    execute.add_argument("--output", type=Path)
    execute.set_defaults(func=run)
    record = sub.add_parser("register", help="登记真实正式结果、原始加密日志及界面信息")
    record.add_argument("--problem", type=int, choices=(3, 4), required=True)
    record.add_argument("--slot", type=int, choices=(1, 2, 3), required=True)
    record.add_argument("--case-code", required=True)
    record.add_argument("--summary", type=Path, required=True)
    record.add_argument("--official-log", type=Path, required=True)
    record.add_argument("--runtime", type=float, required=True, help="官方界面程序运行秒数")
    record.add_argument("--cleared-count", type=int, help="可选：从官方界面核对清除数，与 --virtual-time 同时使用")
    record.add_argument("--virtual-time", type=float, help="可选：从官方界面核对总虚拟秒数；有未知请求时必填")
    record.add_argument("--uploaded", action="store_true", help="仅在界面确认上传成功后使用")
    record.add_argument("--ledger", type=Path, default=Path("支撑材料/正式测试登记.csv"))
    record.set_defaults(func=register)
    check = sub.add_parser("audit", help="检查六份正式记录、文件摘要和上传确认")
    check.add_argument("--ledger", type=Path, default=Path("支撑材料/正式测试登记.csv"))
    check.set_defaults(func=audit)
    exported = sub.add_parser("export-tables", help="生成两问正式结果表；未登记结果保留空缺")
    exported.add_argument("--ledger", type=Path, default=Path("支撑材料/正式测试登记.csv"))
    exported.set_defaults(func=export_tables)
    uploaded = sub.add_parser("confirm-upload", help="官方界面显示上传成功后记录人工确认")
    uploaded.add_argument("--problem", type=int, choices=(3, 4), required=True)
    uploaded.add_argument("--slot", type=int, choices=(1, 2, 3), required=True)
    uploaded.add_argument("--case-code", required=True)
    uploaded.add_argument("--ledger", type=Path, default=Path("支撑材料/正式测试登记.csv"))
    uploaded.set_defaults(func=confirm_upload)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Connection check and a fixed-action practice smoke test."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit

from .client import SimulatorClient
from .errors import SimulatorError
from .rules import DEFAULT_BASE_URL


def smoke_practice(client: SimulatorClient) -> dict:
    """Use the four actions from attachment 2's timing example, then exit.

    The user must select practice mode in the official GUI first. The documented
    HTTP API has no endpoint to query or switch the GUI's test mode.
    """
    responses = []
    error = None
    try:
        responses.append({"path": "/enter", "response": client.enter()})
        for path, position, channel in (
            ("/measure", (300, 400), 1),
            ("/measure", (300, 400), 2),
            ("/clear", (300, 0), 3),
            ("/measure", (300, 0), 2),
        ):
            operation = client.measure if path == "/measure" else client.clear
            responses.append({"path": path, "response": operation(position, channel)})
    except (SimulatorError, OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                responses.append({"path": "/exit", "response": client.exit()})
            except (SimulatorError, OSError, ValueError) as exc:
                error = error or f"{type(exc).__name__}: {exc}"
    return {
        "run_kind": "practice_interface_smoke",
        "mode_selection": "Practice must be selected manually in the simulator GUI",
        "completed": error is None and client.state.session == "exited",
        "error": error,
        "responses": responses,
        "state": client.state.snapshot(),
        "pending_request": client.pending_request,
        "journal_path": str(client.log_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CUMCM 2026 B 模拟器接口工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="仅检查 TCP 端口，不发送模拟器动作")
    check.add_argument("--base-url", default=DEFAULT_BASE_URL)
    check.add_argument("--timeout", type=float, default=2.0)
    smoke = subparsers.add_parser(
        "smoke-practice", help="在已启动的演练中执行附件固定动作，验证四个接口",
        description="先在官方界面选择问题 3 演练测试，等待接口就绪后运行；HTTP 接口不能识别测试类型。",
    )
    smoke.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    smoke.add_argument("--base-url", default=DEFAULT_BASE_URL)
    smoke.add_argument("--log", type=Path, help="新建 JSONL 日志路径，不能覆盖已有日志")
    args = parser.parse_args(argv)

    if args.command == "check":
        parts = urlsplit(args.base_url)
        try:
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError("需要合法的 http(s) 地址")
            with socket.create_connection(
                (parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)),
                timeout=args.timeout,
            ):
                pass
        except (OSError, ValueError) as error:
            print(f"端口不可连接：{error}。请检查模拟器是否已启动演练并结束倒计时。", file=sys.stderr)
            return 1
        print("端口可连接。请在模拟器界面确认当前为演练模式；此检查未发送任何动作。")
        return 0

    if not args.robot_id:
        parser.error("请通过 --robot-id 或 CUMCM_ROBOT_ID 提供当前登录参赛队号")
    try:
        with SimulatorClient(args.robot_id, base_url=args.base_url, log_path=args.log) as client:
            report = smoke_practice(client)
            summary_path = client.log_path.with_suffix(".summary.json")
            with summary_path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print(json.dumps({
                "completed": report["completed"],
                "virtual_time_s": client.state.virtual_time_s,
                "journal": str(client.log_path), "summary": str(summary_path),
                "error": report["error"],
            }, ensure_ascii=False, indent=2))
            return 0 if report["completed"] else 1
    except (OSError, ValueError, SimulatorError) as error:
        print(f"接口检查失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

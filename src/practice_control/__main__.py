"""Practice-only command line; deliberately no mode or generic invoke option."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from .bridge import BridgeError, PracticeBridge
from .runner import PracticeOwnershipError, controller_lock, run_once, validate_run, write_json

ROOT = Path(__file__).resolve().parents[2]
# Registration is an existing repository experiment, outside the src packages.
# Resolve it from this checkout even when the command is launched elsewhere.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description="自动调用原模拟器的第三问、第四问演练")
    parser.add_argument("--debug-port", type=int, default=19226)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="仅读取当前测试的公开状态")
    sub.add_parser("show", help="显示模拟器窗口，供手动登录")
    run = sub.add_parser("run", help="串行启动演练、运行策略、登记结果")
    run.add_argument("--problem", type=int, choices=(3, 4), required=True)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    run.add_argument("--variant", choices=("baseline", "adaptive", "deferred", "efficient", "rollout", "triangular"))
    run.add_argument("--max-actions", type=int, default=20000)
    run.add_argument("--simulator-dir", type=Path, required=True)
    run.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        with PracticeBridge(args.debug_port) as bridge:
            if args.command == "show":
                bridge.show_window()
                return 0
            if args.command == "status":
                print(json.dumps(bridge.current_test(), ensure_ascii=False, indent=2))
                return 0
            variant = validate_run(args.problem, args.variant, args.repeat, args.max_actions, args.robot_id)
            sim_dir = args.simulator_dir.resolve(strict=True)
            if not (sim_dir / "jammers-simulator-full.exe").is_file():
                raise ValueError("simulator-dir must contain the original simulator executable")
            output = args.output or ROOT / "results" / "practice_batches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            with controller_lock(sim_dir / ".practice-control" / "controller.lock"):
                output.mkdir(parents=True, exist_ok=False)
                write_json(output / "batch.json", {"mode": "practice", "problem": args.problem,
                           "variant": variant, "requested_runs": args.repeat})
                records = []
                for index in range(args.repeat):
                    record = run_once(bridge, problem=args.problem, robot_id=args.robot_id,
                                      variant=variant, max_actions=args.max_actions,
                                      output=output / f"run-{index+1:03d}", simulator_dir=sim_dir)
                    records.append(record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    if not record["completed"]:
                        break
                write_json(output / "results.json", records)
                return 0 if len(records) == args.repeat and all(r["completed"] for r in records) else 1
    except (BridgeError, PracticeOwnershipError, OSError, ValueError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

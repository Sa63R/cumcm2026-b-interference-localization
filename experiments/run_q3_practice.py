"""Run one Q3 research spec in an explicitly confirmed official practice window.

The documented HTTP API does not identify practice/formal mode. Current GUI
evidence is required before --gui-practice-confirmed is supplied. Case code is optional: the user authorized running without manual case binding. This command
cannot open a test window and has no formal-test mode. --dry-run sends nothing.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.research_v1_eval import identity, read_json, write_json
from simulator_client import SimulatorClient
from simulator_client.errors import DeadlineExceeded


class DeadlineClient:
    """Stop new physical actions before the hard cutoff, leaving exit time."""
    def __init__(self, client, deadline):
        self.client, self.deadline = client, deadline

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _check(self):
        if time.time() >= self.deadline:
            raise DeadlineExceeded("Practice action deadline reached")

    def enter(self):
        self._check()
        return self.client.enter()

    def measure(self, position, channel):
        self._check()
        return self.client.measure(position, channel)

    def clear(self, position, channel):
        self._check()
        return self.client.clear(position, channel)


def prepare(spec_path):
    spec = read_json(spec_path)
    if not isinstance(spec.get("name"), str) or not isinstance(spec.get("kwargs", {}), dict):
        raise ValueError("spec requires name and an object of kwargs")
    kwargs = spec.get("kwargs", {})
    if "problem" in kwargs or "max_actions" in kwargs:
        raise ValueError("problem is fixed at Q3; max_actions is a runner option")
    module, name = spec["entrypoint"].split(":", 1)
    callback = getattr(importlib.import_module(module), name)
    if not callable(callback):
        raise ValueError("entrypoint must be callable")
    if kwargs.get("checkpoint"):
        # Validate dependencies, weights and feature semantics before /enter.
        from research_rl.network import load_policy
        load_policy(kwargs["checkpoint"], device=kwargs.get("device", "cpu"),
                    deterministic=kwargs.get("deterministic", True))
    protocol = read_json(ROOT / "research/v1_protocol.json")
    return spec, callback, identity(spec, protocol)


def action_window(protocol, explicit_deadline=None):
    """Keep the frozen default; an explicit new practice window is separately logged."""
    protocol_deadline = datetime.fromisoformat(protocol["hard_deadline"])
    default_deadline = protocol_deadline - timedelta(seconds=30)
    deadline = default_deadline
    source = "protocol_minus_30_seconds"
    if explicit_deadline is not None:
        try:
            deadline = datetime.fromisoformat(explicit_deadline)
        except (TypeError, ValueError) as exc:
            raise ValueError("action-deadline must be an ISO8601 time with an explicit timezone") from exc
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise ValueError("action-deadline requires an explicit timezone")
        if deadline.timestamp() <= time.time():
            raise ValueError("action-deadline must be in the future")
        source = "explicit_cli_override"
    return deadline.timestamp(), {
        "protocol_hard_deadline": protocol["hard_deadline"],
        "default_action_deadline": default_deadline.isoformat(),
        "action_deadline": deadline.isoformat(),
        "action_deadline_source": source,
        "action_deadline_override": explicit_deadline,
    }


def run(args, client_factory=SimulatorClient):
    if args.max_actions < 2:
        raise ValueError("max-actions must be >=2")
    protocol = read_json(ROOT / "research/v1_protocol.json")
    action_deadline, deadline_record = action_window(protocol, getattr(args, "action_deadline", None))
    spec, callback, provenance = prepare(args.spec)
    if args.dry_run:
        return dict(dry_run=True, simulator_requests_sent=False, spec=spec, identity=provenance,
                    **deadline_record)
    if not args.gui_practice_confirmed or not args.robot_id:
        raise ValueError("Need current Q3 practice GUI confirmation and robot id before any request")
    if time.time() >= action_deadline:
        message = ("First-version execution deadline reached" if deadline_record["action_deadline_override"] is None
                   else "Explicit practice action deadline reached during preflight")
        raise ValueError(message)
    destination = args.output or ROOT / "results/research_v1/practice" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("Q3_SOURCE_COMMIT", "unavailable")
    report = dict(data_origin="official_simulator_http_practice", problem=3,
                  declared_mode="practice", gui_mode_verified_by_api=False,
                  gui_confirmation_supplied=True, case_code=args.case_code,
                  official_result_verified=False, spec=spec, identity=provenance,
                  code_revision=commit, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  started_at=datetime.now(timezone.utc).isoformat(), completed=False, errors=[],
                  **deadline_record)
    write_json(destination / "preflight.json", report)
    started = time.monotonic()
    try:
        with client_factory(args.robot_id, base_url=args.base_url,
                            log_path=destination / "requests.jsonl") as client:
            try:
                result = callback(DeadlineClient(client, action_deadline), problem=3,
                                  max_actions=args.max_actions, **spec.get("kwargs", {}))
                report["search"] = result.as_dict()
                report["errors"].extend(str(e) for e in (result.error, result.exit_error) if e)
                if not result.completion_certified_under_model:
                    report["errors"].append(f"Uncertified completion: {result.completion_reason}")
            except (Exception, KeyboardInterrupt) as exc:
                report["errors"].append(f"{type(exc).__name__}: {exc}")
                report["exception_traceback"] = traceback.format_exc()
            finally:
                if client.state.session == "active" and client.pending_request is None:
                    try:
                        client.exit()
                    except Exception as exc:
                        report["errors"].append(f"Exit: {type(exc).__name__}: {exc}")
                report["state"] = client.state.snapshot()
                report["pending_request"] = client.pending_request
                if client.pending_request is not None:
                    report["errors"].append("Outcome unknown: preserve pending request; inspect GUI and journal")
                if client.state.session != "exited":
                    report["errors"].append("No accepted exit")
    except Exception as exc:
        report["errors"].append(f"Client: {type(exc).__name__}: {exc}")
    report["program_wall_time_s"] = time.monotonic()-started
    report["completed"] = bool(report.get("search")) and not report["errors"]
    report["runtime_scope"] = "Process wall time including HTTP/SSH latency; official GUI timing not yet verified"
    write_json(destination / "summary.json", report)
    report["summary_path"] = str(destination / "summary.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--case-code", default="")
    parser.add_argument("--gui-practice-confirmed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--action-deadline", help="Explicit new practice action cutoff: future ISO8601 time with timezone; default is frozen protocol deadline minus 30 seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({k: result[k] for k in ("summary_path", "completed", "errors", "state") if k in result},
                     ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

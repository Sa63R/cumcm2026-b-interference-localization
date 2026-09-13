"""One Q4 PRACTICE session for archived speedup candidates; UI check required."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
SPEEDUP = HERE.parent / "q4_speedup"
sys.path.insert(0, str(SPEEDUP))
from runtime import SimulatorClient, configure_geometry, run_policy
from online import State
from strategy import build_candidate
from compute_fast import ExactComputeCache
from coverage_compute_fast import CoverageComputeCache

METHODS = ("v4_fast", "prior_fixed", "cells_flex", "combined")
VERSION = "q4-speedup-practice-20260912-v2"


def prepare_speedup_policy(method, seed=90210000):
    """Build all strategy modules before entering the cache context/network."""
    if method not in METHODS:
        raise ValueError(method)
    geometry = configure_geometry()
    state = State() if method == "v4_fast" else build_candidate(method)
    planner = getattr(state, "planner", None)
    if planner is not None:
        planner.seed = seed
    modules = ["compute_fast.py", "coverage_compute_fast.py", "strategy.py"]
    if method == "prior_fixed":
        modules += ["unrestricted_planner.py", "unrestricted_posterior.py"]
    elif method in ("cells_flex", "combined"):
        modules += ["local_candidate.py", "coverage_candidate.py"]
    if method == "combined":
        modules += ["domain_candidate.py"]
    metadata = dict(
        method=method,
        version=VERSION,
        geometry=geometry,
        cache_enabled=True,
        coverage_cache_enabled=True,
        compute_note=(
            "Exact V4 equivalence was validated for v4_fast only. In prior_fixed "
            "the faster computation can change wall-clock timeout branches; no "
            "uncached action-equivalence claim is made."
        ),
        configuration=asdict(state.config),
        planner_seed=seed if planner else None,
        planning_configuration=asdict(planner.config) if planner else None,
        source_sha256={
            name: hashlib.sha256((SPEEDUP / name).read_bytes()).hexdigest()
            for name in modules
        },
        entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    return state, metadata


def run_speedup_session(client, state, report, progress=None):
    """Run a candidate while preserving unresolved requests exactly as received."""
    cache = ExactComputeCache()
    coverage_cache = CoverageComputeCache()
    policy_cpu = policy_wall = None
    try:
        report["enter_response"] = client.enter()
        report["status"] = "entered"
        policy_cpu = time.process_time()
        policy_wall = time.perf_counter()
        try:
            with cache, coverage_cache:
                report["policy"] = run_policy(
                    client, state, getattr(state, "planner", None), progress
                )
        finally:
            report["policy_cpu_seconds"] = time.process_time() - policy_cpu
            report["policy_wall_seconds"] = time.perf_counter() - policy_wall
            report["compute_cache"] = cache.stats()
            report["coverage_compute_cache"] = coverage_cache.stats()
        report["exit_response"] = client.exit()
        report["status"] = "policy_completed_and_exited"
    except BaseException:
        report["status"] = "interrupted_or_failed"
        report["exception"] = traceback.format_exc()
        if client.state.session == "active" and client.pending_request is None:
            try:
                report["error_exit_response"] = client.exit()
            except Exception:
                report["error_exit_exception"] = traceback.format_exc()
        raise
    finally:
        report["client_state"] = client.state.snapshot()
        report["pending_request"] = client.pending_request
        count = client.state.cleared_count
        report["seconds_per_accepted_clear"] = (
            client.state.virtual_time_s / count if count else None
        )
        report.setdefault("compute_cache", cache.stats())
        report.setdefault("coverage_compute_cache", coverage_cache.stats())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="v4_fast")
    parser.add_argument("--robot-id", required=True, help="Simulator team ID; never a password")
    parser.add_argument(
        "--practice-confirmed",
        action="store_true",
        help="Operator has just visually verified Q4 PRACTICE in the simulator UI",
    )
    parser.add_argument("--case-label", default="not_recorded")
    parser.add_argument("--seed", type=int, default=90210000)
    parser.add_argument("--output", type=Path, default=Path("practice_results"))
    args = parser.parse_args(argv)
    if not args.practice_confirmed:
        parser.error(
            "No connection made. First visually verify Q4 PRACTICE, then pass "
            "--practice-confirmed."
        )
    run_dir = args.output / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + args.method
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        method=args.method,
        version=VERSION,
        case_label=args.case_label,
        status="not_prepared",
        official_ui_feedback="not_recorded",
        mode="operator_confirmed_q4_practice",
        mode_note="API cannot query mode; flag is an operator assertion, not server verification.",
        python=sys.version,
        platform=platform.platform(),
        requested_planner_seed=args.seed,
    )
    preparation = time.perf_counter()
    started_cpu = started_wall = None
    try:
        state, metadata = prepare_speedup_policy(args.method, args.seed)
        report.update(metadata)
        report["preparation_wall_seconds"] = time.perf_counter() - preparation
        report["status"] = "not_entered"
        started_cpu = time.process_time()
        started_wall = time.perf_counter()

        def progress(state, client):
            print(
                f"method={args.method} actions={state.actions} "
                f"cleared={client.state.cleared_count} "
                f"virtual_s={client.state.virtual_time_s:.2f} "
                f"remaining_real_s={client.remaining_real_time_s:.1f}",
                flush=True,
            )

        with SimulatorClient(args.robot_id, log_path=run_dir / "requests.jsonl") as client:
            run_speedup_session(client, state, report, progress)
    except BaseException:
        if "exception" not in report:
            report["status"] = "preparation_or_client_setup_failed"
            report["exception"] = traceback.format_exc()
        raise
    finally:
        report["cpu_seconds"] = (
            time.process_time() - started_cpu if started_cpu is not None else None
        )
        report["wall_seconds"] = (
            time.perf_counter() - started_wall if started_wall is not None else None
        )
        report.setdefault("preparation_wall_seconds", time.perf_counter() - preparation)
        report["timing_note"] = (
            "CPU and wall cover client creation through close; policy times include "
            "cache setup/cleanup and final coverage check. Preparation is separate."
        )
        (run_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Result saved: {run_dir.resolve()}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()

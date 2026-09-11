"""Resumable collection of real Q3/Q4 practice trajectories into one SQLite file.

The lifecycle stays in runner.run_once: only public robot observations reach
the solver, and completed practice results are registered after accepted exit.
The collector never retries an uncertain lifecycle mutation or replaces a case.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from .bridge import BridgeError, PracticeBridge, _require_idle
from .runner import controller_lock, run_once, validate_run

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _event(path, payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


class CollectionStopped(Exception):
    """A requested pause was reached before the next start request."""


class SpacedPracticeBridge:
    """Keep starts at least five seconds apart; preserve native readiness waits."""

    def __init__(self, bridge, *, should_stop=lambda: False, interval=5.0):
        if not math.isfinite(interval) or interval < 5.0:
            raise ValueError("Practice start interval must be at least five seconds")
        self.bridge = bridge
        self.should_stop = should_stop
        self.interval = interval
        # Wait once after acquiring ownership too: a resumed collector cannot
        # know the prior process's monotonic clock or its most recent start.
        self.last_start = time.monotonic()

    def current_test(self):
        return self.bridge.current_test()

    def clear_finished_test(self, case):
        return self.bridge.clear_finished_test(case)

    def start_practice(self, problem):
        while True:
            if self.should_stop():
                raise CollectionStopped("Pause requested before the next practice start")
            state = self.current_test()
            _require_idle(state)
            remaining = state.get("countdown_remaining_ms", 0) or 0
            if type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining < 0:
                raise BridgeError("Invalid simulator readiness countdown; collection stopped")
            gap = 0 if self.last_start is None else self.last_start + self.interval - time.monotonic()
            delay = max(gap, remaining / 1000.0)
            if delay <= 0:
                break
            # Short sleeps allow stop-file checks without interrupting a case.
            time.sleep(min(delay, 0.5))
        self.last_start = time.monotonic()
        # Exactly one call. Any rejection or uncertain result stops collection.
        return self.bridge.start_practice(problem)


def _counts(summary):
    by_problem = summary.get("complete_by_problem", {})
    return {problem: int(by_problem.get(str(problem), by_problem.get(problem, 0)))
            for problem in (3, 4)}


def _next_problem(counts, targets):
    candidates = [p for p in (3, 4) if counts[p] < targets[p]]
    return min(candidates, key=lambda p: (counts[p] / targets[p], p)) if candidates else None


def collect(*, output, simulator_dir, robot_id, q3=500, q4=500, debug_port=19226,
            max_actions=20000, max_hours=None, stop_file=None, resume=False,
            bridge_factory=PracticeBridge, episode_runner=run_once,
            importer=None, statistics=None):
    """Collect until the per-problem targets are met or a safe stop is reached.

    Targets count unique fully cleared scenes. Safely ended incomplete searches
    remain in the dataset, labelled by its quality fields, outside the targets.
    """
    for target in (q3, q4):
        if type(target) is not int or not 0 <= target <= 1_000_000:
            raise ValueError("Each target must be an integer from 0 to 1000000")
    if q3 + q4 == 0:
        raise ValueError("Request at least one practice episode")
    if max_hours is not None and (type(max_hours) not in (int, float)
                                  or not math.isfinite(max_hours) or max_hours <= 0):
        raise ValueError("max_hours must be finite and greater than zero")
    for problem in (3, 4):
        validate_run(problem, None, 1, max_actions, robot_id)
    output = Path(output).resolve()
    simulator_dir = Path(simulator_dir).resolve(strict=True)
    if not (simulator_dir / "jammers-simulator-full.exe").is_file():
        raise ValueError("simulator-dir must contain the original simulator executable")
    raw = output.parent / f"{output.stem}-raw"
    episodes = raw / "episodes"
    progress_path = raw / "progress.json"
    stop_file = Path(stop_file).resolve() if stop_file is not None else raw / "STOP"
    targets = {3: q3, 4: q4}
    if importer is None or statistics is None:
        from .dataset import import_episode, stats
        importer = importer or import_episode
        statistics = statistics or stats
    from .refinement import REFINEMENT_CONFIG, run_collection_search
    started = time.monotonic()
    deadline = None if max_hours is None else started + max_hours * 3600
    stop_reason = None

    def should_stop():
        nonlocal stop_reason
        if stop_file.exists():
            stop_reason = "stop_file"
        elif deadline is not None and time.monotonic() >= deadline:
            stop_reason = "max_hours"
        return stop_reason is not None

    with controller_lock(simulator_dir / ".practice-control" / "controller.lock"):
        if not resume and (output.exists() or raw.exists()):
            raise ValueError("Output already exists; use --resume to import saved episodes and continue")
        output.parent.mkdir(parents=True, exist_ok=True)
        episodes.mkdir(parents=True, exist_ok=True)
        prior = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        errors = list(prior.get("errors", []))
        collection_started_at = prior.get("collection_started_at", _utc_now())
        recovered = 0
        incomplete_streak = 0
        report = None
        initial_counts = None
        current_episode = None

        def publish(status, *, last_episode=None):
            nonlocal report, initial_counts
            dataset_stats = statistics(output)
            counts = _counts(dataset_stats)
            if initial_counts is None:
                initial_counts = counts.copy()
            elapsed = time.monotonic() - started
            new_count = sum(counts[p] - initial_counts[p] for p in (3, 4))
            rate = new_count / elapsed if new_count > 0 and elapsed > 0 else None
            remaining = sum(max(0, targets[p] - counts[p]) for p in (3, 4))
            report = {
                "schema_version": 1, "mode": "practice", "status": status,
                "pid": os.getpid(), "output": str(output), "raw_directory": str(raw),
                "stop_file": str(stop_file), "updated_at": _utc_now(),
                "collection_started_at": collection_started_at,
                "targets": {str(p): targets[p] for p in (3, 4)},
                "counts": {str(p): counts[p] for p in (3, 4)},
                "dataset": dataset_stats, "invocation_elapsed_s": elapsed,
                "new_episodes_this_invocation": new_count, "recovered_episodes": recovered,
                "consecutive_incomplete_this_invocation": incomplete_streak,
                "episodes_per_hour": rate * 3600 if rate else None,
                "estimated_remaining_s": remaining / rate if rate else None,
                "errors": errors, "stop_reason": stop_reason,
                "current_episode": str(current_episode) if current_episode else None,
                "last_episode": last_episode,
            }
            _atomic_json(progress_path, report)
            _event(raw / "events.jsonl", report)
            return counts

        try:
            # Recover DB imports after a crash between saved registration and
            # its atomic progress update. Importer deduplicates by practice case.
            for episode in sorted(episodes.iterdir()):
                if episode.is_dir() and (episode / "registration.json").is_file():
                    recovered += int(bool(importer(output, episode).get("inserted")))
            counts = publish("starting")
            if any(counts[p] > targets[p] for p in (3, 4)):
                raise ValueError("Targets cannot be smaller than the already collected complete episode counts")
            if _next_problem(counts, targets) is None:
                publish("complete")
                return report
            if should_stop():
                publish("paused")
                return report
            with bridge_factory(debug_port) as native:
                # Never clear/adopt/abort an existing practice or formal case.
                _require_idle(native.current_test())
                bridge = SpacedPracticeBridge(native, should_stop=should_stop)
                while (problem := _next_problem(counts, targets)) is not None:
                    if should_stop():
                        publish("paused")
                        return report
                    current_episode = episodes / (
                        f"p{problem}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
                    )
                    publish("running")
                    variant = "efficient" if problem == 3 else "triangular"
                    record = episode_runner(
                        bridge, problem=problem, robot_id=robot_id,
                        variant=variant, solver=run_collection_search,
                        method_label=variant + "+bounded_refinement",
                        method_metadata={"dataset_refinement": dict(REFINEMENT_CONFIG)},
                        max_actions=max_actions, output=current_episode, simulator_dir=simulator_dir,
                    )
                    _atomic_json(current_episode / "collector-result.json", record)
                    imported = importer(output, current_episode)
                    incomplete_streak = 0 if imported.get("complete") is True else incomplete_streak + 1
                    counts = publish("running", last_episode={**record, "dataset_import": imported})
                    # SearchIncomplete is useful labelled training data. Other
                    # solver failures must be inspected instead of repeating.
                    summary = json.loads((current_episode / "summary.json").read_text(encoding="utf-8"))
                    error = summary.get("error")
                    if error and not (isinstance(error, str) and error.startswith("SearchIncomplete:")):
                        raise BridgeError(f"Solver failure preserved in {current_episode.name}: {error}")
                    if incomplete_streak >= 3:
                        raise BridgeError("Three consecutive incomplete episodes preserved; inspect the strategy before resuming")
                    current_episode = None
                publish("complete")
                return report
        except CollectionStopped:
            publish("paused")
            return report
        except (Exception, KeyboardInterrupt) as exc:
            errors.append({"at": _utc_now(), "type": type(exc).__name__, "error": str(exc),
                           "episode": str(current_episode) if current_episode else None})
            try:
                publish("failed")
            except Exception:
                # Even if importing/statistics fails, preserve the failure in
                # a separate atomically written record; no next case starts.
                _atomic_json(raw / "collector-failure.json", {"errors": errors, "at": _utc_now()})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect only Q3/Q4 simulator practice into one SQLite dataset")
    parser.add_argument("--output", type=Path, required=True, help="Single output SQLite dataset file")
    parser.add_argument("--q3", type=int, default=500)
    parser.add_argument("--q4", type=int, default=500)
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--debug-port", type=int, default=19226)
    parser.add_argument("--max-actions", type=int, default=20000)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true", help="Read existing progress without connecting to the simulator")
    args = parser.parse_args(argv)
    if args.status:
        progress = args.output.parent / f"{args.output.stem}-raw" / "progress.json"
        print(progress.read_text(encoding="utf-8"), end="")
        return 0
    if args.simulator_dir is None:
        parser.error("--simulator-dir is required for collection")
    try:
        report = collect(**{key: value for key, value in vars(args).items() if key != "status"})
        return 0 if report["status"] == "complete" else 3
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Practice collection stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

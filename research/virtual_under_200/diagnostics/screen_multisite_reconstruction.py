"""Read old public prefixes; screen fixed geometric completion candidates.

No simulated feedback, policy, SQLite, training, or network is executed.
The current clear point is only a PROPOSED unknown-channel scan: coverage
feasibility depends on actually completing that additional scan in a policy.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(archive, source):
    sys.path.insert(0, str(source / "src"))
    from planning.disk_cover import disk_cover_radius
    from planning.routing import exact_open_route
    from simulator_client.state import Position

    ledger_path = archive / "research/round2/diagnoses/BASELINE_OBSERVATION_COST.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    inputs = {str(ledger_path): sha(ledger_path)}
    code = {name: sha(source / name) for name in (
        "src/planning/disk_cover.py", "src/planning/routing.py", "src/simulator_client/state.py")}

    def length(points, current):
        value = 0.0
        for point in exact_open_route([Position(*p) for p in points], start=Position(*current)):
            target = (point.x, point.y)
            value += math.dist(current, target)
            current = target
        return value

    deletion = {"eligible_after_clear_boundaries": 0, "joint_trials": 0, "feasible_joint_trials": 0,
                "simple_delete_feasible_combinations": 0, "positive_boundaries": 0, "events": []}
    transform = {"eligible_after_clear_boundaries": 0, "joint_trials": 0, "feasible_joint_trials": 0,
                 "positive_boundaries": 0, "best_candidate": None, "events": []}
    for info in ledger["cases"]:
        path = Path(info["input_path"])
        if not path.is_absolute():
            path = archive / path
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info["input_sha256"]
        inputs[str(path)] = info["input_sha256"]
        summary = json.loads(gzip.decompress(raw))["summary"]
        actions = summary["action_history"]
        known, cleared, at = set(), set(), {0: (set(), set())}
        for ordinal, action in enumerate(actions, 1):
            if action["action"] == "measure" and action["result"] in ("direction", "near"):
                known.add(action["channel"])
            if action["action"] == "clear" and action["result"] == "success":
                cleared.add(action["channel"])
            at[ordinal] = (known.copy(), cleared.copy())
        previous = 0
        for log in summary["strategy_parameters"]["relocation_log"]:
            count = log["after_actual_action_count"]
            known, cleared = at[count]
            segment = actions[previous:count]
            previous = count
            successful_clears = [a for a in segment if a["action"] == "clear" and a["result"] == "success"]
            if not count or len(known | cleared) >= 16 or not successful_clears:
                continue
            current = actions[count - 1]["position"]
            if successful_clears[-1]["position"] != current:
                continue
            remaining = log.get("remaining_after", log["remaining_before"])
            past = log["executed_discovery_stations"]
            if len(remaining) < 2:
                continue
            old_length = length(remaining, current)
            unknown_count = 20 - len(known | cleared)
            context = {"seed": info["seed"], "after_action_count": count,
                       "remaining_count": len(remaining), "unknown_count": unknown_count}

            if len(remaining) >= 3:
                deletion["eligible_after_clear_boundaries"] += 1
                best = None
                for dropped, station in enumerate(remaining):
                    kept = [p for index, p in enumerate(remaining) if index != dropped]
                    if disk_cover_radius(past + [current] + kept) <= 1000 - 1e-5:
                        deletion["simple_delete_feasible_combinations"] += 1
                    target_angle = math.atan2(station[1], station[0])
                    angles = [((math.atan2(p[1], p[0]) - target_angle + math.pi) % math.tau - math.pi, index)
                              for index, p in enumerate(kept)]
                    left = [item for item in angles if item[0] < 0]
                    right = [item for item in angles if item[0] > 0]
                    if not left or not right:
                        continue
                    selected = [max(left)[1], min(right)[1]]
                    for fraction in (.2, .4, .6):
                        for mode in ("chord", "angle"):
                            alternate = list(kept)
                            for index in selected:
                                x, y = kept[index]
                                if mode == "chord":
                                    alternate[index] = [x + fraction * (station[0] - x), y + fraction * (station[1] - y)]
                                else:
                                    angle = math.atan2(y, x)
                                    delta = (target_angle - angle + math.pi) % math.tau - math.pi
                                    angle += fraction * delta
                                    radius = math.hypot(x, y)
                                    alternate[index] = [radius * math.cos(angle), radius * math.sin(angle)]
                            deletion["joint_trials"] += 1
                            radius = disk_cover_radius(past + [current] + alternate)
                            if radius > 1000 - 1e-5:
                                continue
                            deletion["feasible_joint_trials"] += 1
                            gain = (old_length - length(alternate, current)) / 5
                            # Current scan replaces one future scan: frozen
                            # unknown-channel scan count cancels in this proxy.
                            if gain > 1 and (best is None or gain > best["cover_only_proxy_gain_s"]):
                                best = {**context, "mode": mode, "fraction": fraction,
                                        "coverage_radius_m": radius, "cover_only_proxy_gain_s": gain,
                                        "dropped_index": dropped, "moved_indices": selected}
                if best:
                    deletion["events"].append(best)

            transform["eligible_after_clear_boundaries"] += 1
            average_x = sum(p[0] for p in remaining) / len(remaining)
            average_y = sum(p[1] for p in remaining) / len(remaining)
            candidates = []
            for fraction in (.025, .05, .1, .2, .4):
                candidates.append(("contract", fraction, [[p[0] + fraction * (current[0] - p[0]),
                    p[1] + fraction * (current[1] - p[1])] for p in remaining]))
                candidates.append(("translate", fraction, [[p[0] + fraction * (current[0] - average_x),
                    p[1] + fraction * (current[1] - average_y)] for p in remaining]))
            for degrees in (-10., -5., -2., 2., 5., 10.):
                co, si = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
                candidates.append(("rotate", degrees, [[co * p[0] - si * p[1], si * p[0] + co * p[1]] for p in remaining]))
            best = None
            for mode, parameter, alternate in candidates:
                transform["joint_trials"] += 1
                radius = disk_cover_radius(past + [current] + alternate)
                if radius > 1000 - 1e-5:
                    continue
                transform["feasible_joint_trials"] += 1
                gain = (old_length - length(alternate, current)) / 5 - 6 * unknown_count
                item = {**context, "mode": mode, "parameter": parameter, "coverage_radius_m": radius,
                        "cover_only_proxy_gain_after_extra_scan_s": gain}
                previous_best = transform["best_candidate"]
                if previous_best is None or gain > previous_best["cover_only_proxy_gain_after_extra_scan_s"]:
                    transform["best_candidate"] = item
                if gain > 1 and (best is None or gain > best["cover_only_proxy_gain_after_extra_scan_s"]):
                    best = item
            if best:
                transform["events"].append(best)
    for group in (deletion, transform):
        group["positive_boundaries"] = len(group["events"])
        group["positive_cases"] = len({row["seed"] for row in group["events"]})
    return {"scope": "Read-only posthoc geometry on 32 already-opened development traces; no policy performance result",
            "case_count": len(ledger["cases"]), "input_sha256": inputs, "source_sha256": code,
            "script_sha256": sha(Path(__file__)), "delete_one_move_two": deletion,
            "transform_all_without_deletion": transform,
            "limitations": ["Only declared finite transforms screened, not arbitrary nonrigid multi-site reconstruction.",
                "Cover-only exact open-route proxy omits joint routing with remaining known sources and future observations.",
                "Current point is not yet an observed unknown-channel station; certificate is conditional on actual extra scans.",
                "No positive proxy trigger does not prove global optimization impossible or expected time unimprovable.",
                "Six seconds per unknown-channel query is a declared conservative fixed scan proxy; no new physical bill exists."],
            "old_reference": {"mean_T_s": ledger["aggregate"].get("mean_virtual_time_s"),
                "documented_mean_T_s": 3164.97420946875, "mean_old_LB_s": 1788.6548308942008,
                "mean_T_over_LB": 1.7825433187810271}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to overwrite prior evidence"
    result = run(args.archive.resolve(), args.source.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: result[name] for name in ("delete_one_move_two", "transform_all_without_deletion")}, ensure_ascii=False))

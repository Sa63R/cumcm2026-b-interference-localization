"""Read only four preselected adverse final cases; never run a policy/simulator."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
METHODS = {"state": "state-future-cover", "rl": "rl-gae095-u512", "geo": "geo-future-cover"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    record = json.loads(gzip.decompress(path.read_bytes()))
    row, summary = record["row"], record["summary"]
    assert row["successful"] and row["failed_clear_count"] == 0
    actions = summary["action_history"]
    phases, totals = defaultdict(Counter), Counter()
    discovered, cleared, stops, moves, costs = {}, [], [], [], []
    seen_stops, seen_measures = set(), set()
    repeats, active_results, active_channels, no_signal = 0, Counter(), Counter(), []
    point, tuned, stamp = [0., 0.], 1, 0
    residuals = []
    for index, action in enumerate(actions, 1):
        destination, channel = action["position"], action["channel"]
        delta = Counter(movement_s=round(math.hypot(destination[0]-point[0], destination[1]-point[1])/5*1e6))
        if action["action"] == "measure":
            delta.update(switching_s=int(channel != tuned)*1000000, detection_s=5000000)
            tuned = channel
            key = (channel, *destination)
            repeats += key in seen_measures
            seen_measures.add(key)
            if action["result"] in {"direction", "near"}:
                discovered.setdefault(channel, [channel, index, action["virtual_time_s"]])
            if action["phase"] != "coverage":
                active_results[action["result"]] += 1
                active_channels[channel] += 1
                if action["result"] == "no_signal":
                    no_signal.append([index, channel, destination])
        else:
            assert action["action"] == "clear" and action["result"] == "success"
            delta.update(optical_s=3000000, removal_s=2000000)
            cleared.append([channel, index, action["virtual_time_s"]])
        now = round(action["virtual_time_s"]*1e6)
        residuals.append(now-stamp-sum(delta.values()))
        totals.update(delta)
        phases[action["phase"]].update(delta)
        phases[action["phase"]]["actions"] += 1
        costs.append(delta)
        if action["phase"] == "coverage" and tuple(destination) not in seen_stops:
            seen_stops.add(tuple(destination))
            stops.append([index, destination, action["virtual_time_s"]])
        moves.append(dict(step=index, from_point=point, to_point=destination, channel=channel,
                          phase=action["phase"], result=action["result"],
                          movement_s=delta["movement_s"]/1e6))
        point, stamp = destination, now
    assert max(abs(x) for x in residuals) <= 2
    assert len(discovered) == len(cleared) == row["source_total"]
    for key in COMPONENTS:
        assert abs(totals[key]/1e6-row[key]) <= 1e-5
    tail = Counter()
    for cost in costs[cleared[-1][1]:]:
        tail.update(cost)
    metrics = summary.get("learning", {})
    return dict(input_sha256=sha(path), case_sha256=row["case_sha256"],
                row={k: row[k] for k in ("seed", "case_id", "strategy", "source_total", "virtual_time_s",
                     "measurement_count", "successful", "failed_clear_count", *COMPONENTS)},
                component_reconstruction_max_step_residual_us=max(abs(x) for x in residuals),
                phases={k: {q: (v if q == "actions" else v/1e6) for q, v in c.items()}
                        for k, c in phases.items()},
                first_discoveries=list(discovered.values()), clearances=cleared,
                first_coverage_site_visits=stops, largest_movement_actions=sorted(
                    moves, key=lambda m: -m["movement_s"])[:6],
                all_sources_first_detected_s=max(v[2] for v in discovered.values()),
                last_clearance_s=cleared[-1][2],
                after_last_clearance=dict(total_s=sum(tail.values())/1e6,
                    actions=len(actions)-cleared[-1][1], **{k:tail[k]/1e6 for k in COMPONENTS}),
                noncoverage_measure_results=dict(active_results),
                noncoverage_measure_by_channel=dict(active_channels),
                exact_position_channel_measure_repeats=repeats, noncoverage_no_signal=no_signal,
                longest_detection_to_clear_waits=sorted(
                    [[c, t-discovered[c][2]] for c, _, t in cleared], key=lambda r: -r[1])[:3],
                rl_metrics={k: metrics[k] for k in ("decisions", "action_counts", "fallback_counts",
                            "fallback_actions", "fallback_virtual_time_s") if k in metrics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("../q3-v1-artifacts/final-evaluations-complete-0930/v1-selection/evaluations"))
    parser.add_argument("--reports", type=Path, default=Path("../q3-state-search/research/final_results_v1"))
    parser.add_argument("--output", type=Path, default=Path("research/final_adverse_cases_v1.json"))
    args = parser.parse_args()
    reports = {s: json.loads((args.reports/f"report-{s}/comparison.json").read_text(encoding="utf-8"))
               for s in ("random", "stress")}
    cases = []
    for split, method, expected in (("random", "state", 800015), ("random", "rl", 800081),
                                    ("random", "geo", 800102), ("stress", "rl", 810003)):
        comparison = reports[split]["comparisons"][method]
        worst = min(comparison["paired_cases"], key=lambda r: r["saved_s"])
        assert worst == comparison["paired_cases"][0] and worst["seed"] == expected
        paths = {key: args.records/f"final_{split}"/name/f"case-{expected}.json.gz"
                 for key, name in (("baseline", "baseline"), ("candidate", METHODS[method]))}
        parts = {key: summarize(path) for key, path in paths.items()}
        baseline, candidate = [parts[k]["row"] for k in ("baseline", "candidate")]
        assert parts["baseline"]["case_sha256"] == parts["candidate"]["case_sha256"]
        delta = {k: candidate[k]-baseline[k] for k in COMPONENTS}
        assert abs(sum(delta.values())+worst["saved_s"]) < 1e-5
        cases.append(dict(split=split, method=method, seed=expected, paired_case=worst,
                     record_paths={k: p.as_posix() for k, p in paths.items()},
                     cost_increase_s=delta, **parts))
    result = dict(schema=1, scope="Post-freeze descriptive diagnosis, no new rollout/counterfactual/parameter choice",
        script_sha256=sha(Path(__file__)), step_convention="1-based accepted measure/clear; excludes enter/exit",
        sequence_columns=dict(first_discoveries=["channel", "step", "virtual_time_s"],
                              clearances=["channel", "step", "virtual_time_s"],
                              first_coverage_site_visits=["step", "position", "virtual_time_s"]),
        report_sha256={s: sha(args.reports/f"report-{s}/comparison.json") for s in reports},
        random_summary={m: {k: reports["random"]["comparisons"][m][k] for k in
                       ("pairs", "wins", "losses", "mean_reduction_fraction", "mean_seconds_saved")}
                       for m in METHODS}, cases=cases)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([dict(seed=c["seed"], method=c["method"], increase_s=c["cost_increase_s"])
                      for c in cases], indent=2))


if __name__ == "__main__":
    main()

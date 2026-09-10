"""Read-only decomposition of already completed validation trajectories.

No simulator, checkpoint evaluation, optimization, or hidden source coordinates
are used. Input is archived public action histories and aggregate evaluator rows.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np

from localization.omni import OmniCandidateRegion
from research_rl.controller import DeepRLSearch, polygon_shape
from simulator_client.state import Position


def replay_probe_geometry(actions):
    """Rebuild only public observations; never run a policy or simulator.

    The source-candidate routine is shared by all archived RL variants, and its
    action semantics have not changed. No cover actions are generated here.
    """
    state = SimpleNamespace(sources={}, position=Position(0, 0), current_channel=1)
    controller = DeepRLSearch(SimpleNamespace(state=state), lambda f, c, t: t)
    probes = []
    for index, item in enumerate(actions):
        channel, action = item["channel"], item["action"]
        position = Position(*item["position"])
        before = None
        if item["phase"] == "rl_active_localization":
            region = controller.regions[channel]
            choices = controller._source_candidates(channel)
            matching = [c for c in choices if c.point.distance_to(position) <= 1e-6]
            if not matching:
                raise ValueError("archived RL probe does not match reconstructed legal candidates")
            choice = matching[0]
            shape = polygon_shape(region.vertices)
            disk = region.enclosing_disk()
            before = dict(index=index, channel=channel, option=choice.option,
                          point=item["position"], result=item["result"],
                          center=list(disk.center), radius_before=disk.radius,
                          major_before=shape["major"], minor_before=shape["minor"],
                          area_before=region.area,
                          move_s=state.position.distance_to(position) / 5,
                          fresh_probes_before=controller.probe_counts.get(channel, 0),
                          center_distance_m=Position.coerce(disk.center).distance_to(position))
        if action == "measure":
            region = controller.regions.setdefault(channel, OmniCandidateRegion())
            controller.observed_positions.setdefault(channel, set()).add((round(position.x, 6), round(position.y, 6)))
            if item["result"] == "direction":
                controller.detected.add(channel)
                controller.first_bearings.setdefault(channel, item["bearing_deg"])
                region.observe(position, item["bearing_deg"])
            elif item["result"] == "near":
                controller.detected.add(channel)
                controller.near_points[channel] = position
            else:
                region.observe_no_signal(position)
            state.current_channel = channel
            if before is not None:
                controller.probe_counts[channel] = controller.probe_counts.get(channel, 0) + 1
                before.update(area_after=region.area,
                              radius_after=region.enclosing_disk().radius if region.vertices else 0)
                probes.append(before)
        elif item["result"] == "success":
            controller.cleared.add(channel)
        state.position = position
    return probes


def load_case(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def costs(actions):
    result = Counter()
    position, channel = (0.0, 0.0), 1
    for action in actions:
        position2 = action["position"]
        result["movement_s"] += round(math.dist(position, position2) / 5 * 1e6) / 1e6
        if action["action"] == "measure":
            result["switching_s"] += int(channel != action["channel"])
            result["detection_s"] += 5
            channel = action["channel"]
        else:
            result["optical_s"] += 3
            result["removal_s"] += 2 * (action["result"] == "success")
        position = position2
    return dict(result)


def adjacent_completed_source_swaps(actions):
    """Retrospective route opportunity, NOT an executable online oracle policy.

    Swap adjacent, disjoint, completed source blocks. All positions, results and
    within-channel measurement/clear orders remain fixed. Feedback in the local
    simulator is position/channel dependent, not cross-channel ordering dependent.
    The later measurement coordinates may not have been predictable at the swap
    point, so the result cannot be advertised as an online-achievable saving.
    """
    blocks = []
    for index, action in enumerate(actions):
        if action["phase"] == "coverage":
            continue
        if blocks and blocks[-1]["end"] == index and blocks[-1]["channel"] == action["channel"]:
            blocks[-1]["end"] += 1
        else:
            blocks.append(dict(start=index, end=index + 1, channel=action["channel"]))
    original = costs(actions)
    candidates = []
    for a, b in zip(blocks, blocks[1:]):
        if a["end"] != b["start"] or a["channel"] == b["channel"]:
            continue
        if any(actions[z["end"] - 1]["action"] != "clear" or
               actions[z["end"] - 1]["result"] != "success" for z in (a, b)):
            continue
        swapped = actions[:a["start"]] + actions[b["start"]:b["end"]] + actions[a["start"]:a["end"]] + actions[b["end"]:]
        alternate = costs(swapped)
        gain = sum(original.values()) - sum(alternate.values())
        if gain > 1e-6:
            candidates.append(dict(channels=[a["channel"], b["channel"]],
                                   action_ranges=[[a["start"], a["end"]], [b["start"], b["end"]]],
                                   saving_s=gain, movement_s=original["movement_s"] - alternate["movement_s"],
                                   switching_s=original["switching_s"] - alternate["switching_s"]))
    return sorted(candidates, key=lambda c: c["saving_s"], reverse=True)


def write_evidence(output, destination):
    """Compact auditable evidence and route plots; large full replay stays local."""
    destination.mkdir(parents=True, exist_ok=True)
    methods = ("rollout", "joint-ppo-001-u344", "joint-ppo-001-u522", "paired-v2-001-u994")
    focus = "joint-ppo-001-u344"
    selected_seeds = [r["seed"] for r in output["paired_comparisons"][f"{focus}_minus_rollout"]["worst_seeds"][:2]]
    swaps = {}
    for name, cases in output["cases"].items():
        swaps[name] = {seed: adjacent_completed_source_swaps(c["actions"]) for seed, c in cases.items()}
    compact = {key: value for key, value in output.items() if key != "cases"}
    compact["retrospective_adjacent_swap"] = {name: dict(
        cases_with_opportunity=sum(bool(c) for c in cases.values()),
        mean_best_single_swap_s=statistics.mean(c[0]["saving_s"] if c else 0 for c in cases.values()))
        for name, cases in swaps.items()}
    compact["caveat"] = "Adjacent swaps hold later observed coordinates fixed; this is retrospective route diagnosis, not an online policy or achievable bound."
    compact["source_candidate_replay"] = "All recorded nonfallback RL probe positions matched candidates rebuilt from only earlier public observations."
    (destination / "summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    examples = {name: {str(seed): dict(output["cases"][name][str(seed)],
                                      retrospective_swaps=swaps[name][str(seed)])
                       for seed in selected_seeds + [6003]} for name in methods}
    (destination / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for row, seed in enumerate(selected_seeds):
        for column, name in enumerate(("rollout", focus)):
            case = output["cases"][name][str(seed)]
            axis = axes[row, column]
            positions = np.array([[0, 0]] + [a["position"] for a in case["actions"]])
            axis.plot(positions[:, 0], positions[:, 1], color="#5174b5", linewidth=1.3, alpha=0.8)
            for move in case["moves"]:
                if move["movement_s"] > 40:
                    axis.annotate("", xy=move["end"], xytext=move["start"],
                                  arrowprops=dict(arrowstyle="->", lw=0.7, color="#5174b5", alpha=0.65))
            clear = np.array([a["position"] for a in case["actions"] if a["action"] == "clear"])
            axis.scatter(clear[:, 0], clear[:, 1], s=28, color="#2b9b68", label="Successful clear")
            for order, site in enumerate(case["site_visits"]):
                x, y = site["point"]
                axis.scatter([x], [y], s=60, color="#df812b", marker="s", zorder=5)
                axis.annotate(str(order), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
            axis.set_title(f"Seed {seed} | {name}\nTotal {case['row']['virtual_time_s']:.1f}s, move {case['row']['movement_s']:.1f}s")
            axis.set_aspect("equal")
            axis.set_xlim(-1900, 1900); axis.set_ylim(-1900, 1900)
            axis.grid(alpha=0.2); axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)")
    fig.suptitle("Archived public actions only: orange squares show coverage visit order; green dots show clear positions")
    fig.tight_layout()
    fig.savefig(destination / "route_comparison.png", dpi=160)
    fig.savefig(destination / "route_comparison.svg")
    plt.close(fig)


def analyze_case(data):
    row, summary = data["row"], data["summary"]
    actions = summary["action_history"]
    counters = Counter()
    phase_costs = defaultdict(Counter)
    discovered, cleared, measured = set(), set(), set()
    first_seen, last_seen = {}, {}
    clear_delay = []
    last_clear, last_discovery = 0.0, 0.0
    last_position, last_channel, previous_time = (0.0, 0.0), 1, 0.0
    moves = []
    site_visits = []
    for index, item in enumerate(actions):
        action, phase, channel = item["action"], item["phase"], item["channel"]
        position = tuple(item["position"])
        movement = round(math.dist(last_position, position) / 5 * 1e6) / 1e6
        phase_costs[phase]["movement_s"] += movement
        phase_costs[phase]["total_s"] += item["virtual_time_s"] - previous_time
        phase_costs[phase]["actions"] += 1
        if movement > 1e-6:
            moves.append(dict(index=index, action=action, phase=phase, channel=channel,
                              start=list(last_position), end=list(position), movement_s=movement,
                              virtual_time_s=item["virtual_time_s"]))
        if action == "measure":
            counters["coverage_measurements" if phase == "coverage" else "active_measurements"] += 1
            counters[f"measure_result_{item['result']}"] += 1
            counters["already_detected_measurements"] += channel in discovered
            counters["already_cleared_measurements"] += channel in cleared
            if phase == "coverage":
                counters["known_source_coverage_measurements"] += channel in discovered
                counters["known_source_coverage_no_signal"] += channel in discovered and item["result"] == "no_signal"
                if not site_visits or site_visits[-1]["point"] != list(position):
                    site_visits.append(dict(index=index, point=list(position), count=0,
                                            arrival_s=item["virtual_time_s"], channels=[]))
                site_visits[-1]["count"] += 1
                site_visits[-1]["channels"].append(channel)
            else:
                counters["active_no_signal"] += item["result"] == "no_signal"
            key = (position, channel)
            counters["exact_repeated_measurements"] += key in measured
            measured.add(key)
            if item["result"] in ("direction", "near") and channel not in discovered:
                first_seen[channel] = item["virtual_time_s"]
                discovered.add(channel)
                last_discovery = item["virtual_time_s"]
            last_seen[channel] = item["virtual_time_s"]
            last_channel = channel
        elif item["result"] == "success":
            cleared.add(channel)
            last_clear = item["virtual_time_s"]
            clear_delay.append(last_clear - first_seen[channel])
        previous_time = item["virtual_time_s"]
        last_position = position
    reconstructed = costs(actions)
    for key in ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s"):
        if abs(reconstructed.get(key, 0) - row[key]) > 1e-4:
            raise ValueError(f"cost reconstruction mismatch: {row['strategy']} {row['seed']} {key}")
    tail = [item for item in actions if item["virtual_time_s"] > last_clear + 1e-7]
    initial_origin_measures = 0
    for item in actions:
        if item["action"] != "measure" or tuple(item["position"]) != (0.0, 0.0):
            break
        initial_origin_measures += 1
    learning = summary.get("learning", {})
    counters.update(dict(initial_origin_measures=initial_origin_measures,
                         tail_after_last_clear_s=row["virtual_time_s"] - last_clear,
                         tail_measurements=len(tail),
                         tail_after_last_discovery_s=row["virtual_time_s"] - last_discovery,
                         mean_discovery_to_clear_s=statistics.mean(clear_delay),
                         coverage_site_runs=len(site_visits),
                         fallback_virtual_time_s=learning.get("fallback_virtual_time_s", 0),
                         interrupted_scans=learning.get("interrupted_scans", 0),
                         scan_site_changes=learning.get("scan_site_changes", 0)))
    geometry = replay_probe_geometry(actions) if learning and not learning.get("fallback_actions") else []
    return dict(row=row, metrics=dict(counters), phase_costs=dict(phase_costs),
                reconstructed_costs=reconstructed, moves=moves, site_visits=site_visits,
                learning=learning, actions=actions, probe_geometry=geometry)


def paired(first, second):
    seeds = sorted(set(first) & set(second))
    if not seeds:
        return None
    for seed in seeds:
        if first[seed]["row"]["case_sha256"] != second[seed]["row"]["case_sha256"]:
            raise ValueError("paired cases differ")
    delta = np.array([first[s]["row"]["virtual_time_s"] - second[s]["row"]["virtual_time_s"] for s in seeds])
    rng = np.random.default_rng(913)
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    keys = ("virtual_time_s", "movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
    return dict(n=len(seeds), interpretation="first minus second; positive means first is slower",
                mean_delta={k: statistics.mean(first[s]["row"][k] - second[s]["row"][k] for s in seeds) for k in keys},
                time_delta_ci95=[float(v) for v in np.quantile(draws, [0.025, 0.975])],
                faster=int((delta < -1e-6).sum()), slower=int((delta > 1e-6).sum()),
                worst_seeds=[dict(seed=seeds[i], delta_s=float(delta[i])) for i in np.argsort(delta)[-5:][::-1]],
                best_seeds=[dict(seed=seeds[i], delta_s=float(delta[i])) for i in np.argsort(delta)[:5]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestones", nargs="+", default=["milestone-0200", "milestone-0230"])
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args(argv)
    directories = sorted((args.artifacts / "baseline-validation").glob("validation-*"))
    for milestone in args.milestones:
        directories += sorted((args.artifacts / milestone).glob("*/results/research_v1/validation-*"))
    results, provenance = {}, {}
    for directory in directories:
        if not (directory / "rows.json").is_file():
            continue
        name = directory.name.removeprefix("validation-")
        rows_hash = hashlib.sha256((directory / "rows.json").read_bytes()).hexdigest()
        archive_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(directory.glob("case-*.json.gz"))}
        relative_directory = directory.relative_to(args.artifacts).as_posix()
        provenance[relative_directory] = dict(rows_sha256=rows_hash,
            archives_manifest_sha256=hashlib.sha256(json.dumps(archive_hashes, sort_keys=True).encode()).hexdigest())
        if name in results:
            previous = next(value for path, value in provenance.items()
                            if path.endswith('/' + directory.name))
            if previous != provenance[relative_directory]:
                raise ValueError(f"duplicate archived strategy differs: {name}")
            continue
        cases = {}
        for path in sorted(directory.glob("case-*.json.gz")):
            data = load_case(path)
            # This tool is for these already exposed validation cases only.
            if not 6000 <= data["row"]["seed"] <= 6047:
                raise ValueError("refusing nondeclared validation artifact")
            cases[data["row"]["seed"]] = analyze_case(data)
        if len(cases) != 48:
            raise ValueError(f"expected 48 archived cases: {directory}")
        results[name] = cases
    aggregate = {}
    for name, cases in results.items():
        keys = set().union(*(c["metrics"] for c in cases.values()))
        row_keys = ("virtual_time_s", "movement_s", "switching_s", "detection_s", "optical_s", "removal_s", "source_total")
        aggregate[name] = dict(n=len(cases), mean_cost={k: statistics.mean(c["row"][k] for c in cases.values()) for k in row_keys},
            mean_metrics={k: statistics.mean(c["metrics"].get(k, 0) for c in cases.values()) for k in sorted(keys)},
            failed_count=sum(not c["row"]["successful"] for c in cases.values()))
        probes = [probe for case in cases.values() for probe in case["probe_geometry"]]
        if probes:
            aggregate[name]["probe_options"] = dict(Counter(p["option"] for p in probes))
            aggregate[name]["probe_count"] = len(probes)
            aggregate[name]["center_probe_fraction"] = sum(p["center_distance_m"] < 1e-5 for p in probes) / len(probes)
        aggregate[name]["full_origin_scan_cases"] = sum(c["metrics"]["initial_origin_measures"] == 20 for c in cases.values())
        aggregate[name]["scan_interruption_cases"] = sum(c["metrics"]["interrupted_scans"] > 0 for c in cases.values())
    comparisons = {f"{name}_minus_rollout": paired(cases, results["rollout"]) for name, cases in results.items() if name != "rollout"}
    if "joint-ppo-001-u178" in results:
        comparisons["joint_ppo_minus_joint_bc"] = paired(results["joint-ppo-001-u178"], results["joint-bc-001"])
    output = dict(kind="read_only_existing_validation_diagnostic", new_simulator_runs=0,
                  hidden_coordinates_used=False, provenance=provenance, aggregate=aggregate,
                  analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  paired_comparisons=comparisons, cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.evidence_dir:
        # Normalize integer dictionary keys in memory to match a reloaded JSON.
        write_evidence(json.loads(json.dumps(output)), args.evidence_dir)
    print(json.dumps(dict(aggregate=aggregate, paired_comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()

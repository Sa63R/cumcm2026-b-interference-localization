"""Read-only replay of fixed, already completed public observation prefixes.

No scenario, simulator, controller, evaluation truth, SQLite, or network API is
used. This is mechanism screening, never a counterfactual full-policy run.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE_SHA256 = "b82db545aa71dafbc7878488c0f669a3f7897cc862e2b28d719612dbefe6c0e4"
INPUT_SHAS = (
    "dea8111757799bbe9fb6d2f97538ea52f26c48a21724aaa1ac1cff993c6c3145",
    "d2c9c818f40b38d487afee510a773f0c3d1e9363bf4dae2d2ecf3b631006639c",
    "debeecbc7805e16e40ac0b3e20a0233af03471c1254ac8035a10718440f33835",
    "417b2595741d2221d6fd97ab914b61dd99fc5baee003c0c472847ad0aaad4c43",
    "b3cb4a9b5f2c82a6bcd5f1e12e010dd9bf596654fdd10cdc44ade3cd7a21055b",
    "572ebe3d8043933b37732e8cfedb5e9607f33f66746e4e491e0c8acfe623dcb3",
    "54b8a31a20003e549b070906516f5228c80c1d2979fe64ec93929fecf085d2ca",
    "57f820a5e68beaca067351c772ef515bad5e71ded8d2a5c139c45f0bedb7b810",
    "34d0cb40efc0672f5553d71303c8a84b421b14e9f6b0c5759795b0233c557640",
    "8f279e84d085ae1f6e5578caf8d57edda00096ff1c26536a5c5710df6b1ea680",
    "b9b69a8b878c5d1dbd43dd49a248f04fc2c16fcd84cd4197168dc4dc7f931693",
    "ba5ad75ab50524b1b6575b93e0fabf815640b6c4f93003a70296e49eccc2eca2",
    "90bb11415dc535e63d5c7418c303a263b906f16ddacefaf69c5a8d597f3621b3",
    "2ee80fdb07330a70fc5ceb2037912e16c1013a91d23b35edcc476f0fd0f89e68",
    "1874783c6d6de2e6cecf362243805d52be26e2091e64fe70d2eab67252a56982",
    "b0430a799c12f2cec7c9a69300ccf2086600f6bbfaa1ffacf61e06f327b1ae79",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen(report):
    from localization.omni import OmniCandidateRegion
    from planning.enroute_scan import choose_enroute_scan, _guaranteed, FRACTIONS
    from planning.relative_silence import relative_silence_certificate
    from planning.silence_certificate import certify_silence
    from simulator_client.state import Position

    history = report["action_history"]
    params = report["strategy_parameters"]
    boundaries = []
    for scan in params["derived_scan_audit"]:
        start = scan["after_actual_action_count_before"]
        station = Position(*scan["position"])
        regions, observed, negatives = {}, {}, {}
        detected, cleared, near = set(), set(), set()
        current, current_channel = Position(0, 0), None
        inferred = {}
        for item in params["inferred_no_signal_constraints"]:
            count = item["after_actual_action_count"]
            # Do not install the upcoming scan's own inferred observations.
            # A different-point event at this count can only be a prior scan.
            if count < start or (count == start and Position(*item["position"]) != station):
                inferred.setdefault(count, []).append(item)
        for count in range(start + 1):
            for item in inferred.get(count, []):
                regions.setdefault(item["channel"], OmniCandidateRegion()).observe_no_signal(item["position"])
            if count == start:
                break
            action = history[count]
            channel = action["channel"]
            current = Position(*action["position"])
            if action["action"] == "measure":
                current_channel = channel
                observed.setdefault(channel, set()).add((round(current.x, 6), round(current.y, 6)))
                region = regions.setdefault(channel, OmniCandidateRegion())
                if action["result"] == "direction":
                    region.observe(current, action["bearing_deg"])
                    detected.add(channel)
                elif action["result"] == "near":
                    near.add(channel)
                    detected.add(channel)
                elif action["result"] == "no_signal":
                    region.observe_no_signal(current)
                    negatives.setdefault(channel, []).append(current)
            elif action["action"] == "clear" and action["result"] == "success":
                cleared.add(channel)
        if sorted(detected | cleared) != scan["known_channels_before"]:
            raise ValueError("known-channel prefix differs from retained scan audit")
        eligible, filtered, mixed_only_opportunities = [], [], []
        for channel in sorted(detected - cleared - near):
            region = regions[channel]
            reason = None
            if not region.vertices:
                reason = "empty_region"
            elif region.enclosing_disk().radius <= 19.9:
                reason = "certified"
            elif certify_silence(region, station):
                reason = "old_silence"
            elif any(relative_silence_certificate(region, station, p)
                     for p in negatives.get(channel, [])):
                reason = "relative_actual_negative"
            if reason:
                filtered.append({"channel": channel, "reason": reason})
                continue
            eligible.append(channel)
            if (current != station and not _guaranteed(region, station)
                    and (round(station.x, 6), round(station.y, 6)) not in observed.get(channel, set())):
                fractions = []
                for fraction in FRACTIONS:
                    point = Position(current.x + fraction * (station.x - current.x),
                                     current.y + fraction * (station.y - current.y))
                    if (_guaranteed(region, point)
                            and (round(point.x, 6), round(point.y, 6)) not in observed.get(channel, set())):
                        fractions.append(fraction)
                if fractions:
                    mixed_only_opportunities.append({"channel": channel, "fractions": fractions})
        models = {}
        for model in ("guaranteed", "mixed"):
            _, _, log = choose_enroute_scan(regions, eligible, current, station,
                                            observed, current_channel, reception_model=model)
            models[model] = log
        boundaries.append({"prefix_action_count": start, "current": [current.x, current.y],
                           "station": list(scan["position"]), "current_channel": current_channel,
                           "known_channels": sorted(detected | cleared),
                           "cleared_channels": sorted(cleared), "near_channels": sorted(near),
                           "eligible_known_channels": eligible, "filtered_channels": filtered,
                           "q_safe_b_not_safe": mixed_only_opportunities, "models": models})
    return boundaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPOSITORY.parent)
    parser.add_argument("--output", type=Path, required=True, help="new directory; overwrite refused")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = REPOSITORY / "src/planning/enroute_scan.py"
    if sha(source) != SOURCE_SHA256:
        raise ValueError("frozen enroute model source changed")
    sys.path.insert(0, str(REPOSITORY / "src"))
    locked = {str(source): SOURCE_SHA256, str(Path(__file__).resolve()): sha(Path(__file__))}
    # Bind every imported local geometry/state dependency without importing a
    # strategy controller. These source bytes are evidence, not model inputs.
    for directory in ("localization", "geometry", "simulator_client"):
        for path in sorted((REPOSITORY / "src" / directory).rglob("*.py")):
            locked[str(path)] = sha(path)
    for name in ("shared_observation.py", "relative_silence.py", "silence_certificate.py"):
        path = REPOSITORY / "src/planning" / name
        locked[str(path)] = sha(path)
    cases, inputs = [], []
    for offset, expected in enumerate(INPUT_SHAS):
        seed = 926091210101 + offset
        relative = f"q3-round3/results/round3/directed_localization/pilot/records/baseline-{seed}.json.gz"
        path = args.workspace / relative
        if sha(path) != expected:
            raise ValueError(f"input changed: {relative}")
        locked[str(path)] = expected
        inputs.append({"case": seed, "path_relative_to_workspace": relative, "sha256": expected})
        report = json.loads(gzip.decompress(path.read_bytes()))["summary"]
        if report["error"] or not report["all_cleared"]:
            raise ValueError("expected a completed retained baseline record")
        cases.append({"case": seed, "boundaries": screen(report)})
    summary = {}
    for model in ("guaranteed", "mixed"):
        totals, skips, gains = Counter(), Counter(), []
        selected_cases = 0
        for case in cases:
            selected_case = False
            for boundary in case["boundaries"]:
                log = boundary["models"][model]
                totals.update({"scan_boundaries": 1,
                    "eligible_known_channels": len(boundary["eligible_known_channels"]),
                    "channels_scored": log["channels_scored"],
                    "points_scored": log["candidate_points_scored"],
                    "geometry_updates": log["geometry_updates"],
                    "selected_boundaries": int(log["selected"] is not None)})
                selected_case |= log["selected"] is not None
                skips.update(log["skip_reasons"])
                gains.extend(point["gross_proxy_gain_s"] for row in log["channel_results"]
                             for point in row["candidates"] if "gross_proxy_gain_s" in point)
            selected_cases += int(selected_case)
        summary[model] = {**totals, "selected_cases": selected_cases,
                          "skip_reasons": dict(skips), "maximum_gross_proxy_gain_s": max(gains, default=None)}
    opportunity_cases = [case["case"] for case in cases
                         if any(b["q_safe_b_not_safe"] for b in case["boundaries"])]
    opportunities = [item for case in cases for boundary in case["boundaries"]
                     for item in boundary["q_safe_b_not_safe"]]
    payload = {"scope": "old public-prefix mechanism screen; no new complete policy executions",
               "not_performance_evidence": True, "fixed_gross_threshold_s": 7.0,
               "source_sha256": SOURCE_SHA256, "script_sha256": sha(Path(__file__)),
               "summary": summary, "mixed_only_geometric_opportunities": {
                   "cases": opportunity_cases, "channel_boundary_count": len(opportunities),
                   "point_count": sum(len(item["fractions"]) for item in opportunities)},
               "cases": cases}
    for path, expected in locked.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"input/source changed during screening: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    result = args.output / "enroute_prefix_screen.json"
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    source_inputs = [{"path_relative_to_repository": str(Path(path).relative_to(REPOSITORY)).replace('\\', '/'),
                      "sha256": digest} for path, digest in locked.items()
                     if Path(path).is_relative_to(REPOSITORY)]
    manifest = {"inputs": inputs, "source_inputs": source_inputs,
                "input_source_hashes_verified_before_and_after": True,
                "result_sha256": sha(result)}
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary,
                      "result_sha256": sha(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

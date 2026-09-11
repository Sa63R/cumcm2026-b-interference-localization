"""Independent prefix audit for the two derived-silence scan reductions.

Only observation histories and public logs are read.  The candidate's certificate
routine is never imported.  Exact rational arithmetic verifies the relative
inequality on binary64 input vertices; the outer-region construction retains the
legacy floating-geometry contract.  Real-history authenticity/physical costs are
the caller's separate frozen-record/physical audit responsibility.

`observation_audit(record, cover_cache, legacy)` is the integration entry point.
The original auditor is run again with *only* the new relative inferences removed:
it must still certify real clears and terminal per-channel absence.  No inferred
point becomes an actual negative measurement or an absence/coverage disk.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path


VERSION = "q3-derived-silence-prefix-audit-v1"
RELATIVE = "relative_actual_negative"
MARGIN_M = 1e-5
CHANNELS = frozenset(range(1, 21))


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or invalid {label}")
    return value


def _point(value, label="position"):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid {label}")
    return tuple(_number(v, label) for v in value)


def _channel(value):
    if type(value) is not int or value not in CHANNELS:
        raise ValueError("Invalid channel")
    return value


def _count(value, maximum, label="action prefix"):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid {label}")
    return value


def _channels(value, label):
    if not isinstance(value, list) or any(type(c) is not int or c not in CHANNELS for c in value):
        raise ValueError(f"Invalid {label}")
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate channel in {label}")
    return value


def _same_time(value, expected, label):
    if abs(_number(value, label) - expected) > 2e-6:
        raise ValueError(f"{label} differs from actual prefix")


def exact_relative_certificate(vertices, query, negative, margin_m=MARGIN_M):
    """Prove min G > margin*|n-q| without any square root or float products.

    G(v)=(n-q).(v-(n+q)/2) is half the squared-distance difference.
    Fractions represent input numeric coordinates exactly.  This is a sufficient
    certificate for the supplied conservative polygon, not a proof of its own
    numerical construction or a posterior/hidden-source test.
    """
    query, negative = _point(query), _point(negative, "actual negative")
    margin_m = _number(margin_m, "margin")
    if margin_m < MARGIN_M or not vertices:
        raise ValueError("Relative certificate requires a nonempty region and fixed safety margin")
    q, n = tuple(map(Fraction, query)), tuple(map(Fraction, negative))
    delta = tuple(n[i] - q[i] for i in range(2))
    norm2 = sum(d*d for d in delta)
    if not norm2:
        raise ValueError("Relative certificate has identical query and negative")
    midpoint = tuple((n[i] + q[i])/2 for i in range(2))
    values = [sum(delta[i]*(Fraction(v[i])-midpoint[i]) for i in range(2))
              for v in (_point(v, "region vertex") for v in vertices)]
    lower = min(values)
    if lower <= 0 or lower*lower <= Fraction(margin_m)**2 * norm2:
        raise ValueError("Relative certificate lacks a strict positive distance margin")
    return lower, norm2


def _check_relative(event, region, actual, count):
    witness = event.get("witness_action_ordinal")
    if type(witness) is not int or not 1 <= witness <= count:
        raise ValueError("Relative witness must be an earlier actual action ordinal")
    action = actual[witness-1]
    if (action["action"] != "/measure" or action["channel"] != event["channel"]
            or action["response"].get("measure_result") != "no_signal"
            or event.get("witness_kind") != "prior_actual_measure_no_signal"):
        raise ValueError("Relative witness must be a prior actual same-channel no_signal")
    negative = (action["position"]["x"], action["position"]["y"])
    if _point(event.get("actual_negative_position"), "witness position") != negative:
        raise ValueError("Relative witness position differs from actual measurement")
    _same_time(event.get("witness_virtual_time_s"), action["response"]["virtual_time_s"],
               "Witness virtual time")
    if (type(event.get("outer_region_vertex_count")) is not int
            or event["outer_region_vertex_count"] != len(region.vertices)):
        raise ValueError("Relative vertex count differs from prefix region")
    margin = _number(event.get("margin_m"), "relative margin")
    lower, norm2 = exact_relative_certificate(region.vertices, event["position"], negative, margin)
    # Independently check the directions of the candidate's reported enclosures.
    claimed = Fraction(_number(event.get("affine_lower_bound_m2"), "affine lower bound"))
    required = Fraction(_number(event.get("required_affine_margin_m2"), "required margin"))
    upper = Fraction(_number(event.get("query_negative_distance_upper_m"), "distance upper bound"))
    signed = Fraction(_number(event.get("signed_bisector_distance_lower_m"), "signed lower bound"))
    if (not 0 < claimed <= lower or upper <= 0 or upper*upper < norm2
            or required < Fraction(margin)*upper or claimed <= required
            or signed <= 0 or signed*upper > claimed):
        raise ValueError("Reported relative certificate enclosure is not conservative")
    return {"channel": event["channel"], "after_actual_action_count": count,
            "witness_action_ordinal": witness, "exact_binary64_vertex_certificate": True,
            "vertex_count": len(region.vertices), "coverage_credits": 0}


def _physical_prefixes(record):
    summary = record.get("summary") or {}
    actual = [a for a in record["history"] if a["action"] in ("/measure", "/clear")]
    reported = summary.get("action_history", [])
    if len(actual) != len(reported):
        raise ValueError("Physical versus strategy action-history length mismatch")
    enters = [a for a in record["history"] if a["action"] == "/enter"]
    initial_time = 0.
    if enters:
        initial_time = _number(enters[0]["response"]["virtual_time_s"], "entry time")
    detected, cleared = set(), set()
    prefixes = [{"known": frozenset(), "detected": frozenset(), "cleared": frozenset(),
                 "channel": 1, "time": initial_time, "position": (0., 0.)}]
    for action, report in zip(actual, reported):
        channel = _channel(action["channel"])
        kind, response = action["action"][1:], action["response"]
        q = _point((action["position"]["x"], action["position"]["y"]))
        if (response.get("accepted") is not True or action.get("physical_measurement", True) is not True
                or report.get("physical_measurement", True) is not True):
            raise ValueError("Physical history contains an unaccepted or inferred physical action")
        outcome = response["measure_result" if kind == "measure" else "clear_result"]
        t = _number(response["virtual_time_s"], "actual virtual time")
        if (report["action"] != kind or _channel(report["channel"]) != channel
                or _point(report["position"]) != q or report["result"] != outcome):
            raise ValueError("Physical versus strategy action-history content mismatch")
        _same_time(report["virtual_time_s"], t, "Reported virtual time")
        if t < prefixes[-1]["time"]:
            raise ValueError("Actual virtual times are not monotone")
        if kind == "measure":
            if outcome not in ("direction", "near", "no_signal"):
                raise ValueError("Unknown physical measurement outcome")
            if outcome == "direction" and report.get("bearing_deg") != response.get("svd_deg"):
                raise ValueError("Strategy bearing differs from actual response")
            if outcome in ("direction", "near"):
                if channel in cleared:
                    raise ValueError("Physical positive signal after a successful clear")
                detected.add(channel)
        else:
            if outcome not in ("success", "no_target_in_range"):
                raise ValueError("Unknown physical clear outcome")
            if outcome == "success":
                if channel in cleared:
                    raise ValueError("Repeated successful clear of the same channel")
                cleared.add(channel)
        if len(detected | cleared) > 16:
            raise ValueError("Actual public evidence exceeds the Q3 count bound")
        # Optical clearing does not retune the receiver; only a real measure
        # changes the channel used first by the next scan.
        tuned = channel if kind == "measure" else prefixes[-1]["channel"]
        prefixes.append({"known": frozenset(detected | cleared), "detected": frozenset(detected),
                         "cleared": frozenset(cleared), "channel": tuned, "time": t, "position": q})
    return actual, reported, prefixes


def _scan_schedule(summary, actual, reported, prefixes, events):
    """Check scan accounting using actual prefixes, never final success counts."""
    parameters = summary.get("strategy_parameters", {})
    scans = parameters.get("derived_scan_audit", [])
    config = parameters.get("derived_silence_config", {})
    active = config.get("enabled", False) and (config.get("relative_silence", True)
                                               or config.get("stop_scan_at_16", True))
    if scans and not active:
        raise ValueError("Derived scan log present while both mechanisms are disabled")
    if any(e.get("method") == RELATIVE for e in events) and not scans:
        raise ValueError("Relative inference is missing its scan audit")
    available = defaultdict(list)
    for index, event in enumerate(events):
        available[(event["after_actual_action_count"], event["channel"], tuple(event["position"]),
                   "relative" if event.get("method") == RELATIVE else "old")].append(index)
    consumed, clear_requests, checked = set(), defaultdict(set), []
    last_end, credit_points = 0, []
    totals = {"relative_silence_skips": 0, "count_cap_skips": 0, "completed_geometric_scans": 0,
              "count_certified_scans": 0, "no_physical_action_scans": 0}
    for scan in scans:
        start = _count(scan.get("after_actual_action_count_before"), len(actual), "scan start")
        end = _count(scan.get("after_actual_action_count_after"), len(actual), "scan end")
        if start < last_end or end < start:
            raise ValueError("Derived scan ranges overlap or run backwards")
        last_end = end
        q = _point(scan.get("position"))
        before, after = prefixes[start], prefixes[end]
        for label, prefix in (("known_channels_before", before), ("known_channels_after", after)):
            if _channels(scan.get(label), label) != sorted(prefix["known"]):
                raise ValueError("Scan known-channel list differs from actual prefix")
        if type(scan.get("completed")) is not bool:
            raise ValueError("Scan completion must be boolean")
        measured = _channels(scan.get("actual_measured_channels"), "actual measured channels")
        categories = {"actual": measured}
        for key, label in (("relative", "relative_silence_skipped"), ("old", "old_silence_skipped"),
                           ("clear", "clear_certified_skipped")):
            categories[key] = _channels(scan.get(label), label)
        counts = scan.get("count_skipped")
        if not isinstance(counts, list):
            raise ValueError("Missing scan count-skipped log")
        categories["count"] = _channels([e["channel"] for e in counts], "count skipped")
        if categories["count"] and not config.get("stop_scan_at_16", True):
            raise ValueError("Count skip present while mechanism disabled")
        if categories["relative"] and not config.get("relative_silence", True):
            raise ValueError("Relative skip present while mechanism disabled")
        channel_kind = {}
        for kind, values in categories.items():
            for channel in values:
                if channel in channel_kind:
                    raise ValueError("Scan channel belongs to multiple action/skip categories")
                channel_kind[channel] = kind
        order = sorted(CHANNELS - before["cleared"])
        if before["channel"] in order:
            order.remove(before["channel"])
            order.insert(0, before["channel"])
        processed = order if scan["completed"] else order[:len(channel_kind)]
        if set(processed) != set(channel_kind):
            raise ValueError("Scan categories are not a completed/prefix channel traversal")
        for kind, values in categories.items():
            if values != [c for c in processed if channel_kind[c] == kind]:
                raise ValueError("Scan category order differs from actual channel traversal")
        cursor, count_map = start, {e["channel"]: e for e in counts}
        for channel in processed:
            kind, prefix = channel_kind[channel], prefixes[cursor]
            if kind == "actual":
                if cursor >= end:
                    raise ValueError("Scan claims a nonexistent physical measurement")
                action = actual[cursor]
                if (action["action"] != "/measure" or action["channel"] != channel
                        or (action["position"]["x"], action["position"]["y"]) != q
                        or reported[cursor].get("phase") != "coverage"):
                    raise ValueError("Scan physical query differs from actual history")
                cursor += 1
            elif kind == "count":
                event = count_map[channel]
                if (_count(event.get("after_actual_action_count"), len(actual), "count-skip prefix") != cursor
                        or len(prefix["known"]) != 16
                        or channel in prefix["known"]
                        or _channels(event.get("known_channels"), "count witness") != sorted(prefix["known"])):
                    raise ValueError("Count skip lacks 16 distinct actual known channels at that prefix")
            else:
                if channel not in prefix["detected"] - prefix["cleared"]:
                    raise ValueError("Known-channel skip refers to unknown/cleared channel")
                if kind == "clear":
                    clear_requests[cursor].add(channel)
                else:
                    matching = available[(cursor, channel, q, kind)]
                    if not matching:
                        raise ValueError("Scan inferred skip lacks its separate certificate event")
                    consumed.add(matching.pop(0))
        if cursor != end or _count(scan.get("physical_action_count"), len(actual), "physical action count") != end-start or measured != [
                a["channel"] for a in actual[start:end]]:
            raise ValueError("Scan physical action count/list differs from actual history")
        unknown = CHANNELS - after["known"]
        real_negatives = {a["channel"] for a in actual[start:end]
                          if a["response"].get("measure_result") == "no_signal"}
        geometric = bool(scan["completed"] and end > start and unknown <= real_negatives)
        cap = bool(scan["completed"] and len(after["known"]) == 16)
        if (scan.get("credited_geometric_station") is not geometric
                or scan.get("certified_by_count") is not cap):
            raise ValueError("Scan claims unearned geometric coverage or count certificate")
        if geometric:
            credit_points.append((end, q))
        totals["relative_silence_skips"] += len(categories["relative"])
        totals["count_cap_skips"] += len(categories["count"])
        totals["completed_geometric_scans"] += geometric
        totals["count_certified_scans"] += cap
        totals["no_physical_action_scans"] += bool(scan["completed"] and end == start)
        checked.append({"after_actual_action_count_before": start, "after_actual_action_count_after": end,
                        "physical_action_count": end-start, "count_skips": len(categories["count"]),
                        "credited_geometric_station": geometric, "certified_by_count": cap,
                        "zero_physical_actions": start == end})
    if active and len(consumed) != len(events):
        raise ValueError("Inference is not accounted for by its actual scan prefix")
    if active:
        if "coverage_points_visited" in summary and summary["coverage_points_visited"] != len(credit_points):
            raise ValueError("Coverage visit count includes an uncredited/fake physical station")
        for entry in parameters.get("relocation_log", []):
            count = _count(entry.get("after_actual_action_count"), len(actual), "relocation prefix")
            expected = [list(q) for end, q in credit_points if end <= count]
            if entry.get("executed_discovery_stations") != expected:
                raise ValueError("Relocation history credits an unexecuted/incomplete discovery station")
        if parameters.get("derived_silence_stats", totals) != totals:
            raise ValueError("Derived scan aggregate statistics differ from independently replayed events")
    return clear_requests, checked, totals


def observation_audit(record, cover_cache=None, legacy=None):
    """Raise on any unsupported prefix; return the unweakened legacy certificate.

    `legacy` is the loaded q3-geometric research_v1_physical_audit module, passed
    by the caller so helper-source hashes can be recorded by its existing runner.
    """
    if legacy is None:
        raise ValueError("An independently loaded legacy auditor is required")
    summary = record.get("summary") or {}
    actual, reported, prefixes = _physical_prefixes(record)
    events = summary.get("strategy_parameters", {}).get("inferred_no_signal_constraints", [])
    schedule, previous = defaultdict(list), 0
    for event in events:
        count = _count(event.get("after_actual_action_count"), len(actual), "inference prefix")
        if count < previous:
            raise ValueError("Inferences are not recorded in causal prefix order")
        previous = count
        _channel(event.get("channel"))
        _point(event.get("position"))
        if (event.get("physical_measurement") is not False
                or event.get("inference_kind") != "inferred_no_signal"):
            raise ValueError("Inference claims a physical measurement")
        _same_time(event.get("virtual_time_s"), prefixes[count]["time"], "Inference virtual time")
        schedule[count].append(event)
    clear_requests, scan_checks, totals = _scan_schedule(summary, actual, reported, prefixes, events)
    regions, near, relative_checks = {}, set(), []
    old_count = 0

    def finish_prefix(count):
        nonlocal old_count
        prefix = prefixes[count]
        for event in schedule[count]:
            channel = event["channel"]
            region = regions.get(channel)
            if (channel not in prefix["detected"] - prefix["cleared"]
                    or region is None or not region.vertices):
                raise ValueError("Inference lacks a known uncleared nonempty source region")
            if event.get("method") == RELATIVE:
                relative_checks.append(_check_relative(event, region, actual, count))
            else:
                if event.get("method") not in ("enclosing_disk", "polygon_edges"):
                    raise ValueError("Unknown inferred-silence method")
                if legacy.polygon_distance(region.vertices, tuple(event["position"])) <= 1500.+MARGIN_M:
                    raise ValueError("Old inferred silence lacks its independent >1500m certificate")
                old_count += 1
            region.observe_no_signal(event["position"])
            if not region.vertices:
                raise ValueError("Inferred constraint empties its known source region")
        # All these facts follow from this same actual prefix. Independent
        # channel scans can interleave at one count; none uses later feedback.
        for channel in clear_requests[count]:
            region = regions.get(channel)
            if channel not in near and (not region or not region.vertices
                                        or region.enclosing_disk().radius > 19.9):
                raise ValueError("Clear-certified scan skip lacks its prefix near/MEC certificate")

    finish_prefix(0)
    for count, action in enumerate(actual, 1):
        channel, response = action["channel"], action["response"]
        region = regions.setdefault(channel, legacy.OmniCandidateRegion())
        if action["action"] == "/measure" and channel not in prefixes[count-1]["cleared"]:
            q = (action["position"]["x"], action["position"]["y"])
            if response["measure_result"] == "direction":
                region.observe(q, response["svd_deg"])
            elif response["measure_result"] == "near":
                near.add(channel)
            else:
                region.observe_no_signal(q)
            if channel in prefixes[count]["detected"] and not region.vertices:
                raise ValueError("Actual prefix leaves an empty known source region")
        finish_prefix(count)
    # Construct a read-only observation projection.  Do not even deep-copy an
    # archive's unrelated ground-truth/scenario payload into this proof replay.
    parameters = dict(summary.get("strategy_parameters", {}))
    parameters["inferred_no_signal_constraints"] = [e for e in events if e.get("method") != RELATIVE]
    stripped = {"history": record["history"],
                "summary": {**summary, "strategy_parameters": parameters}}
    result = dict(legacy.observation_audit(stripped, cover_cache))
    if result["inferred_silence_verified"] != old_count or result["inferred_coverage_credits"] != 0:
        raise ValueError("Legacy verification disagrees with independent old-inference replay")
    result.update(derived_audit_version=VERSION, legacy_inferred_silence_verified=old_count,
                  relative_inferred_silence_verified=len(relative_checks),
                  inferred_silence_verified=old_count+len(relative_checks),
                  relative_certificates=relative_checks, derived_scan_checks=scan_checks,
                  derived_scan_totals=totals, relative_inferred_coverage_credits=0,
                  legacy_without_relative_constraints_passed=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="One explicitly authorized, completed JSON(.gz) record")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output overwrite is forbidden")
    from experiments.round2_posthoc_audit import load_helpers
    _, legacy, hashes = load_helpers()
    opener = gzip.open if args.record.suffix == ".gz" else open
    with opener(args.record, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    result = observation_audit(record, {}, legacy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"record": str(args.record), "helper_sha256": hashes, "observations": result},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

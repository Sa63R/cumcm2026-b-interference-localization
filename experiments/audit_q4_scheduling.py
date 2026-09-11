"""Independent scheduling-prefix checks; no truth or policy imports."""
import math
from localization import CandidateRegion


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_scheduling_prefix(record):
    summary = record["summary"]
    parameters = summary["strategy_parameters"]
    config = parameters["q4_r2_scheduling"]
    require(config in {"cap16", "onroute"}, "Unknown scheduling config")
    history = summary["action_history"]
    wire = [a for a in record["history"] if a["action"] in {"/measure", "/clear"}]
    require(len(wire) == len(history), "Wire/phase history length differs")
    points = [tuple(p) for p in summary["coverage_points"]]
    services = parameters.get("early_service_log", [])
    stops = parameters.get("discovery_stop_log", [])
    require(len(services) <= (4 if config == "onroute" else 0), "Too many early services")
    require(len(stops) <= 1, "Repeated discovery count-cap declaration")
    snapshots, known, cleared, near, regions, covers = {}, set(), set(), set(), {}, []
    requested = {0, len(history)}
    for event in services:
        requested.update((event["after_actual_action_count"], event["end_actual_action_count"]))
    requested.update(e["after_actual_action_count"] for e in stops)
    require(all(type(i) is int and 0 <= i <= len(history) for i in requested), "Invalid event prefix")
    position, elapsed = (0., 0.), 0.
    for index in range(len(history)+1):
        if index in requested:
            disks = {c: (tuple(r.enclosing_disk().center), r.enclosing_disk().radius)
                     for c, r in regions.items() if r.vertices}
            snapshots[index] = (set(known), set(cleared), set(near), disks, list(covers), position, elapsed)
        if index == len(history):
            break
        item, raw = history[index], wire[index]
        response = raw["response"]
        require(response.get("accepted") is True, "Unaccepted action in scheduling prefix")
        position = tuple(item["position"])
        raw_point = raw["position"]
        raw_point = (raw_point["x"], raw_point["y"]) if isinstance(raw_point, dict) else tuple(raw_point)
        require(position == raw_point and item["channel"] == raw["channel"]
                and "/"+item["action"] == raw["action"], "Phase history differs from wire")
        elapsed = response["virtual_time_s"]
        require(abs(item["virtual_time_s"]-elapsed) <= 2e-6, "Phase time differs from wire")
        channel = item["channel"]
        if item["action"] == "measure":
            result = response["measure_result"]
            require(item["result"] == result, "Measurement result mismatch")
            if result in {"direction", "near"}:
                known.add(channel)
            if result == "near":
                near.add(channel)
            elif result == "direction":
                require(item["bearing_deg"] == response["svd_deg"], "Bearing mismatch")
                regions.setdefault(channel, CandidateRegion()).observe(position, response["svd_deg"])
            if item.get("phase") == "coverage" and position not in covers:
                covers.append(position)
                require(covers == points[:len(covers)], "Coverage prefix is not the fixed chain")
        else:
            require(item["result"] == response["clear_result"], "Clear result mismatch")
            if response["clear_result"] == "success":
                known.add(channel)
                cleared.add(channel)
    attempted, previous_end, costs, interrupted = set(), 0, [], 0
    for event in services:
        start, end, channel = event["after_actual_action_count"], event["end_actual_action_count"], event["channel"]
        require(previous_end <= start <= end, "Overlapping/unsorted early services")
        require(channel not in attempted, "Repeated early service for one channel")
        attempted.add(channel)
        before_known, before_clear, before_near, disks, visited, current, start_time = snapshots[start]
        require(len(before_known) < 16 and channel in before_known-before_clear-before_near, "Invalid early-service source")
        require(channel in disks and len(visited) < len(points), "Missing region or next coverage station")
        center, radius = disks[channel]
        next_cover = points[len(visited)]
        detour = math.dist(current, center)+math.dist(center, next_cover)-math.dist(current, next_cover)
        require(19.9 < radius <= 40.+1e-7 and detour <= 100.+1e-7, "Early service violates radius/detour gate")
        require(math.dist(current, center)/5.+6. <= 60.+1e-7, "First approach cannot fit service slice")
        require(abs(event["radius_m"]-radius) <= 1e-6 and abs(event["detour_m"]-detour) <= 1e-6,
                "Logged early-service geometry mismatch")
        actual = snapshots[end][6]-start_time
        require(-1e-7 <= actual <= 60.+2e-6 and abs(actual-event["actual_cost_s"]) <= 2e-6
                and event["budget_s"] == 60., "Early service cost/budget mismatch")
        require(all(h["channel"] == channel and h.get("phase") != "coverage" for h in history[start:end]),
                "Service interval contains another task")
        require(type(event["interrupted"]) is bool and type(event["cleared"]) is bool
                and event["cleared"] == (channel in snapshots[end][1]), "Service outcome mismatch")
        require(not event["interrupted"] or not event["cleared"], "Successful service labelled budget-interrupted")
        costs.append(actual)
        interrupted += event["interrupted"]
        previous_end = end
    for event in stops:
        known, cleared, _, _, visited, _, _ = snapshots[event["after_actual_action_count"]]
        require(len(known) == 16 and event["known_channels"] == sorted(known), "Discovery stopped before 16 real known sources")
        require(event["unresolved_channels"] == sorted(known-cleared)
                and event["omitted_cover_stations"] == len(points)-len(visited)
                and event["discovery_only_not_removal"] is True, "Count-cap evidence mismatch")
        require(all(h.get("phase") != "coverage" for h in history[event["after_actual_action_count"]:]),
                "Discovery scans continued after declared stop")
        if record["row"].get("successful"):
            require(len(snapshots[len(history)][1]) == 16, "Discovery count cap falsely used as all-clear certificate")
    return {"passed": True, "early_services": len(services), "early_service_cost_s": sum(costs),
            "discovery_stops": len(stops), "budget_interrupted_services": interrupted,
            "boundary": "Checks observed service interval and gates; the omitted next action is not recorded, so exact budget-expiry choice is not independently reconstructed"}

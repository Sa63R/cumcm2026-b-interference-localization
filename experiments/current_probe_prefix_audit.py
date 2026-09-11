"""Public-prefix audit of the explicitly opened current-probe implementation smoke.

No policy, simulator, case generator, SQLite or hidden source field is used.
The unchanged v1 finite scorer is recomputed, not the candidate wrapper/gates.
This is an implementation audit; scoring results are not policy-performance
bounds. The caller binds helper hashes and performs the generic physical audit.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.round2_derived_silence_audit import _physical_prefixes, observation_audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ledger(record):
    """Reconstruct every accepted action's actual microsecond cost, no truth."""
    previous, tuned, elapsed = (0., 0.), 1, 0
    parts = dict(movement_s=0, switching_s=0, detection_s=0, optical_s=0, removal_s=0)
    for i, action in enumerate(record["history"]):
        response = action["response"]
        require(action["index"] == i and response.get("accepted") is True, "Actual order/acceptance mismatch")
        if action["action"] in ("/measure", "/clear"):
            point = (action["position"]["x"], action["position"]["y"])
            movement = round(math.dist(previous, point)/5*1e6)
            previous = point
            parts["movement_s"] += movement
            elapsed += movement
            if action["action"] == "/measure":
                switch = int(tuned != action["channel"])*1_000_000
                tuned = action["channel"]
                parts["switching_s"] += switch
                parts["detection_s"] += 5_000_000
                elapsed += switch+5_000_000
            else:
                removal = int(response["clear_result"] == "success")*2_000_000
                parts["optical_s"] += 3_000_000
                parts["removal_s"] += removal
                elapsed += 3_000_000+removal
        else:
            require(action["action"] in ("/enter", "/exit"), "Unexpected physical action")
        require(abs(response["virtual_time_s"]-elapsed/1e6) <= 2e-6, "Actual per-prefix time billing mismatch")
    result = {k:v/1e6 for k,v in parts.items()}
    require(result == record["evaluation"]["time_breakdown_s"], "Cost components differ from post-exit ledger")
    return result


def _pure_helpers():
    """Load only the protected baseline's two pure finite-score modules."""
    root = Path(__file__).resolve().parents[2]/"q3-state-search"/"src"/"planning"
    result = []
    for basename in ("radius_probe", "probe_candidates"):
        name = "_current_probe_audit_"+basename
        spec = importlib.util.spec_from_file_location(name, root/(basename+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        result.append(module)
    return result


def _dependencies():
    from experiments import round2_derived_silence_audit
    paths = {Path(__file__).resolve(), Path(round2_derived_silence_audit.__file__).resolve()}
    for name in ("geometry", "localization", "localization.omni", "simulator_client.state",
                 "simulator_client.rules", "planning.disk_cover", "_current_probe_audit_radius_probe",
                 "_current_probe_audit_probe_candidates"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            paths.add(Path(module.__file__).resolve())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def audit_current_probe(record, observations=None, *, legacy=None):
    """Reusable public-record entry; the second argument is the passed old audit.

    For standalone use the legacy module may also be supplied as argument two
    or via the keyword. Unidentified fallback/pending-action shapes fail closed;
    failed episodes are not silently declared complete. Probe re-entry after a
    genuine coverage scan resets the inherited atomic-resolution probe count.
    """
    if observations is not None and not isinstance(observations, dict):
        require(legacy is None, "Conflicting legacy arguments")
        legacy, observations = observations, None
    if legacy is None:
        from localization.omni import OmniCandidateRegion
        legacy = SimpleNamespace(OmniCandidateRegion=OmniCandidateRegion)
    if observations is None:
        require(hasattr(legacy, "observation_audit"), "Standalone audit needs the independent legacy module")
        observations = observation_audit(record, {}, legacy)
    radius_module, candidates_module = _pure_helpers()
    choose_radius_probe, geometry_candidates = radius_module.choose_radius_probe, candidates_module.geometry_candidates
    dependencies = _dependencies()

    def finish(result):
        require(dependencies == _dependencies(), "Auditor/geometry dependencies changed during replay")
        result.update(dependency_sha256=dependencies, dependency_unchanged_during_audit=True)
        return result

    actual, reports, prefixes = _physical_prefixes(record)
    parts = ledger(record)
    params = record["summary"]["strategy_parameters"]
    logs, main_logs = params.get("current_probe_log", []), params.get("probe_search_log", [])
    if not params.get("current_probe_enabled", False):
        require(not logs, "Disabled wrapper produced new decision logs")
        return finish({"passed": True, "scope": "disabled/baseline parent plus supplied legacy physical prefix proof",
                "observations": observations, "costs_s": parts, "localization_records": 0,
                "eligible": 0, "new_current_selected": 0, "records": []})

    # These modules belong to the pinned legacy geometry tree; the model gate
    # independently checks their bytes against the frozen candidate's unchanged
    # v1 files before accepting this score replay.
    from simulator_client.state import Position
    active_ordinals = [i for i,a in enumerate(reports, 1)
                       if a["action"] == "measure" and a.get("phase") == "active_localization"]
    require(len(logs) == len(main_logs) == len(active_ordinals), "Localization logging does not cover actual calls")
    by_prefix = {}
    for row, main, ordinal in zip(logs, main_logs, active_ordinals):
        count = row["after_actual_action_count"]
        require(type(count) is int and count == ordinal-1 and count not in by_prefix,
                "Probe log does not name its immediate actual public prefix")
        require(row["expected_action_ordinal"] == ordinal == row["actual_action_ordinal"], "Probe action ordinal mismatch")
        by_prefix[count] = row, main
    inference_schedule = defaultdict(list)
    for event in params.get("inferred_no_signal_constraints", []):
        inference_schedule[event["after_actual_action_count"]].append(event)
    regions, near, measured, first_bearing = {}, set(), defaultdict(set), {}
    actual_probe_count = Counter()
    checked = []

    def same(field, saved, reconstructed):
        require(saved == reconstructed, "Recomputed "+field+" differs from log")

    def finish_prefix(count):
        for event in inference_schedule[count]:
            regions[event["channel"]].observe_no_signal(event["position"])
        if count not in by_prefix:
            return
        row, main = by_prefix[count]
        channel, index = row["channel"], row["index"]
        prefix = prefixes[count]
        same("real current position", row["current_position"], list(prefix["position"]))
        same("actual prior primary probe count", index, actual_probe_count[channel])
        same("main-log channel", main["channel"], channel)
        same("main-log probe index", main["index"], index)
        require(channel in prefix["detected"]-prefix["cleared"], "Primary channel is not known and uncleared")
        region = regions[channel]
        q = Position(*prefix["position"])
        radius = region.enclosing_disk().radius
        require(channel not in near and radius > 19.9, "Probe bypassed an available direct clear")
        require(region.vertices and all(math.isfinite(x) for v in region.vertices for x in v), "Invalid source region")
        key = (round(q.x, 6), round(q.y, 6))
        fresh = key not in measured[channel]
        maximum = max(math.dist((q.x, q.y), v) for v in region.vertices)
        area = math.fsum(a[0]*b[1]-a[1]*b[0] for a,b in zip(region.vertices, (*region.vertices[1:], region.vertices[0])))
        nondegenerate = len(region.vertices) >= 3 and math.isfinite(area) and area != 0.
        receives = maximum <= 1000.-1e-5
        eligible = index == 0 and nondegenerate and fresh and receives
        reason = ("not_first_probe" if index else "empty_or_degenerate_region" if not nondegenerate
                  else "already_measured_here" if not fresh else "reception_not_certified" if not receives else "eligible")
        same("gate eligibility", row["eligible"], eligible)
        same("gate rejection reason", row["reason"], reason)
        if index == 0 and nondegenerate:
            same("channel-local real freshness", row["fresh_for_channel"], fresh)
            same("all-vertex maximum distance", row["maximum_vertex_distance_m"], maximum)
            same("guaranteed reception", row["guaranteed_reception"], receives)
            same("new-point margin", row["reception_margin_m"], 1e-5)

        old9, nine = choose_radius_probe(region, q, first_bearing[channel], measured[channel], params["probe_uncertainty_weight"])
        require(old9 is not None, "This smoke scorer audit does not replace missing-parent fallback")
        extras, geometry = geometry_candidates(region, mode="axis_quantile", old_best=old9)
        old23, base = choose_radius_probe(region, q, first_bearing[channel], measured[channel], params["probe_uncertainty_weight"], extra_points=extras)
        require(old23 is not None and math.isfinite(base["score_s"]), "Original finite-score comparator unavailable")
        same("original complete 23-point selection", row["baseline_position"], [old23.x, old23.y])
        old_saved = row["baseline_probe_log"] if eligible else main
        for field in ("position", "score_s", "candidates", "proposed_candidates", "hypotheses", "geometry_updates",
                      "evaluated_candidates", "pruned_candidates", "score_kind"):
            same("original "+field, old_saved[field], base[field])
        same("original nine-point winner", old_saved["old_best_position"], [old9.x, old9.y])
        same("original nine-point score", old_saved["old_best_score_s"], nine["score_s"])
        same("original nine-point legal count", old_saved["baseline_candidates"], nine["candidates"])
        same("original nine-point update count", old_saved["baseline_geometry_updates"], nine["geometry_updates"])
        same("unchanged axis geometry", main["axis"], geometry["axis"])
        same("original extra set count", main["extra_points_proposed"], 14)
        expected, final = old23, base
        if eligible:
            # Exact rational proof on every supplied binary64 input vertex.
            threshold2 = Fraction(1000.-1e-5)**2
            require(all(sum((Fraction(v[i])-Fraction((q.x,q.y)[i]))**2 for i in (0,1)) <= threshold2
                        for v in region.vertices), "Exact input-vertex reception certificate failed")
            expected, final = choose_radius_probe(region, q, first_bearing[channel], measured[channel], params["probe_uncertainty_weight"], extra_points=[*extras,q])
            require(expected in (old23, q) and math.isfinite(final["score_s"])
                    and final["score_s"] <= base["score_s"]+1e-9, "Extended finite-choice consistency failed")
            for field, value in (("baseline_score_s",base["score_s"]),("baseline_candidates",base["candidates"]),
                                 ("baseline_proposed_candidates",base["proposed_candidates"]),
                                 ("baseline_geometry_updates",base["geometry_updates"]),
                                 ("selected_score_s",final["score_s"]),
                                 ("score_improvement_s",base["score_s"]-final["score_s"]),
                                 ("added_geometry_updates",final["geometry_updates"]),
                                 ("extra_point_was_duplicate",final["proposed_candidates"]==base["proposed_candidates"]),
                                 ("changed_on_score_tie",expected!=old23 and final["score_s"]==base["score_s"])):
                same(field,row[field],value)
            same("full three-pass updates",main["total_geometry_updates"],nine["geometry_updates"]+base["geometry_updates"]+final["geometry_updates"])
            same("three-pass recorded scorer runtime",main["total_runtime_s"],old_saved["total_runtime_s"]+row["added_scoring_runtime_s"])
        else:
            same("original two-pass updates",main["total_geometry_updates"],nine["geometry_updates"]+base["geometry_updates"])
        for field in ("position", "score_s", "candidates", "proposed_candidates", "hypotheses", "geometry_updates"):
            same("final "+field,main[field],final[field])
        same("actual selected position",row["selected_position"],[expected.x,expected.y])
        same("changed selection",row["changed_point"],expected!=old23)
        activated = eligible and expected==q and expected!=old23
        same("new current activation",row["new_current_selected"],activated)
        action = actual[count]
        require(action["action"]=="/measure" and action["channel"]==channel
                and reports[count].get("phase")=="active_localization"
                and (action["position"]["x"],action["position"]["y"])==(expected.x,expected.y), "Selected probe differs from next actual action")
        same("execution status",row["execution_status"],"accepted")
        same("bound actual position",row["actual_position"],[expected.x,expected.y])
        same("bound actual time",row["actual_virtual_time_s"],action["response"]["virtual_time_s"])
        increment = action["response"]["virtual_time_s"]-prefix["time"]
        bill = round(q.distance_to(expected)/5*1e6)/1e6+5+int(prefix["channel"]!=channel)
        require(abs(increment-bill)<2e-6,"Chosen probe's actual movement/detection/switch bill differs")
        if eligible:
            require(action["response"]["measure_result"] in ("direction","near"), "Guaranteed receiving probe was actually silent")
        next_clear = next((a for a in actual[count+1:] if a["action"]=="/clear" and a["channel"]==channel), None)
        next_clear_time = next_clear["response"]["virtual_time_s"] if next_clear is not None else None
        checked.append({"channel":channel,"index":index,"prefix":count,"reason":reason,"eligible":eligible,
            "new_current_selected":activated,"current_position":[q.x,q.y],"baseline_23_position":[old23.x,old23.y],
            "actual_probe_position":[expected.x,expected.y],"old_score_s":base["score_s"],"selected_score_s":final["score_s"],
            "all_vertex_max_distance_m":maximum,"reception_slack_to_1000_m":1000.-maximum,
            "exact_input_vertex_reception_verified":eligible,"actual_measure_result":action["response"]["measure_result"],
            "actual_probe_bill_s":bill,"real_movement_s":round(q.distance_to(expected)/5*1e6)/1e6,
            "next_same_channel_clear_virtual_time_s":next_clear_time,
            "next_same_channel_clear_result":next_clear["response"]["clear_result"] if next_clear is not None else None})

    finish_prefix(0)
    for count,action in enumerate(actual,1):
        channel,response=action["channel"],action["response"]
        region=regions.setdefault(channel,legacy.OmniCandidateRegion())
        if action["action"]=="/measure":
            # Frozen v1 retries a blocked source only after a real new coverage
            # scan. Such a scan is not replaced with a planned future station.
            if reports[count-1].get("phase")=="coverage":
                actual_probe_count.clear()
            point=(action["position"]["x"],action["position"]["y"])
            measured[channel].add((round(point[0],6),round(point[1],6)))
            if response["measure_result"]=="direction":
                first_bearing.setdefault(channel,response["svd_deg"])
                region.observe(point,response["svd_deg"])
            elif response["measure_result"]=="near":
                near.add(channel)
            else:
                region.observe_no_signal(point)
            if reports[count-1].get("phase")=="active_localization":
                actual_probe_count[channel]+=1
        finish_prefix(count)
    require(len(checked)==len(logs),"Some decision prefixes were not replayed")
    return finish({"passed":True,"scope":"public finite-radius probe replay; unsupported fallback/pending shapes fail closed; not a performance gate",
            "observations":observations,"costs_s":parts,"localization_records":len(checked),
            "eligible":sum(r["eligible"] for r in checked),"new_current_selected":sum(r["new_current_selected"] for r in checked),
            "reason_counts":dict(Counter(r["reason"] for r in checked)),"records":checked})

"""Plot one past relative-silence certificate, without worlds or simulation.

Only the chosen inference and preceding public actions/constraints are used.
The first relative event is the default; it is not selected by final outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_public_prefix(path, event_index):
    raw = json.loads(gzip.decompress(path.read_bytes()))
    # No evaluation, ground-truth, estimates, source count or later feedback is
    # accessed. The gzip is an archive container, not a policy-facing world.
    parameters = raw["summary"]["strategy_parameters"]
    chosen = None
    preceding = []
    relative_count = 0
    for item in parameters["inferred_no_signal_constraints"]:
        if item.get("method") == "relative_actual_negative":
            if relative_count == event_index:
                chosen = dict(item)
                break
            relative_count += 1
        preceding.append(dict(item))
    if chosen is None:
        raise ValueError("Requested relative event is absent")
    limit = chosen["after_actual_action_count"]
    actions = []
    for item in raw["history"]:
        if item["action"] not in ("/measure", "/clear"):
            continue
        if len(actions) >= limit:
            break
        actions.append({"action": item["action"], "position": dict(item["position"]),
                        "channel": item["channel"], "response": dict(item["response"])})
    if len(actions) != limit:
        raise ValueError("Chosen inference exceeds the public action prefix")
    saved = raw["summary"]["action_history"][:limit]
    for a, b in zip(actions, saved):
        response = a["response"]
        assert response["accepted"] is True
        assert a["action"][1:] == b["action"] and a["channel"] == b["channel"]
        assert [a["position"]["x"], a["position"]["y"]] == b["position"]
        key = "measure_result" if a["action"] == "/measure" else "clear_result"
        assert response[key] == b["result"]
        assert response["virtual_time_s"] == b["virtual_time_s"]
        if response.get("measure_result") == "direction":
            assert response["svd_deg"] == b["bearing_deg"]
    if actions[-1]["response"]["virtual_time_s"] != chosen["virtual_time_s"]:
        raise ValueError("Inference virtual time differs from its actual prefix")
    source_hashes = dict(raw["source_sha256"])
    assert source_hashes == raw["source_after_sha256"]
    assert raw["source_unchanged_during_smoke"] is True
    return chosen, preceding, actions, source_hashes


def reconstruct(chosen, preceding, actions, source_root, source_hashes):
    # The two observation geometry modules and their imports are frozen by the
    # archived smoke source map. No candidate policy or simulator is run.
    required = ["src/geometry/__init__.py", "src/localization/__init__.py",
                "src/localization/omni.py"]
    for name in required:
        if sha256(source_root / name) != source_hashes[name]:
            raise ValueError(f"Frozen source mismatch: {name}")
    sys.path.insert(0, str(source_root / "src"))
    from localization.omni import OmniCandidateRegion

    schedule = defaultdict(list)
    for event in preceding:
        if event["after_actual_action_count"] > len(actions):
            raise ValueError("Earlier inference uses later observations")
        schedule[event["after_actual_action_count"]].append(event)
    regions, detected, cleared = {}, set(), set()
    used = []

    def apply_preceding(count):
        for event in schedule[count]:
            channel = event["channel"]
            if channel not in detected - cleared:
                raise ValueError("Past inference has no known active source")
            assert event["physical_measurement"] is False
            regions[channel].observe_no_signal(event["position"])

    apply_preceding(0)
    for count, a in enumerate(actions, 1):
        channel, response = a["channel"], a["response"]
        point = (a["position"]["x"], a["position"]["y"])
        if a["action"] == "/measure" and channel not in cleared:
            region = regions.setdefault(channel, OmniCandidateRegion())
            outcome = response["measure_result"]
            if outcome == "direction":
                detected.add(channel)
                region.observe(point, response["svd_deg"])
            elif outcome == "near":
                detected.add(channel)
            else:
                assert outcome == "no_signal"
                region.observe_no_signal(point)
            if channel == chosen["channel"]:
                used.append({"ordinal": count, "position": list(point), "result": outcome,
                             **({"bearing_deg": response["svd_deg"]} if outcome == "direction" else {})})
        elif a["action"] == "/clear" and response["clear_result"] == "success":
            cleared.add(channel)
        apply_preceding(count)
    channel = chosen["channel"]
    if channel not in detected - cleared:
        raise ValueError("Target channel is not publicly known and active")
    region = regions[channel]
    if len(region.vertices) != chosen["outer_region_vertex_count"]:
        raise ValueError("Rebuilt vertex count differs from certificate")
    ordinal = chosen["witness_action_ordinal"]
    if not 1 <= ordinal <= len(actions):
        raise ValueError("Witness is outside the past prefix")
    witness = actions[ordinal - 1]
    n = [witness["position"]["x"], witness["position"]["y"]]
    assert witness["action"] == "/measure" and witness["channel"] == channel
    assert witness["response"]["measure_result"] == "no_signal"
    assert n == chosen["actual_negative_position"]
    assert witness["response"]["virtual_time_s"] == chosen["witness_virtual_time_s"]
    return list(region.vertices), used, required


def exact_check(vertices, event):
    q, n = [tuple(map(Fraction, event[k])) for k in ("position", "actual_negative_position")]
    delta = tuple(n[i] - q[i] for i in range(2))
    mid = tuple((n[i] + q[i]) / 2 for i in range(2))
    norm2 = sum(x*x for x in delta)
    values = [sum(delta[i] * (Fraction(v[i]) - mid[i]) for i in range(2)) for v in vertices]
    margin = Fraction(event["margin_m"])
    assert norm2 > 0 and all(g > 0 and g*g > margin*margin*norm2 for g in values)
    least = min(values)
    assert Fraction(event["affine_lower_bound_m2"]) <= least
    signed_m = float(least) / math.sqrt(float(norm2))
    assert 0 < event["signed_bisector_distance_lower_m"] <= signed_m
    return {"all_vertices_strictly_on_q_farther_side": True,
            "exact_binary64_fraction_check": True,
            "all_vertices_clear_required_margin": True,
            "minimum_affine_G_fraction": str(least), "delta_norm_squared_fraction": str(norm2),
            "minimum_squared_distance_difference_m2": float(2*least),
            "minimum_signed_bisector_distance_m": signed_m,
            "candidate_conservative_signed_lower_m": event["signed_bisector_distance_lower_m"],
            "required_margin_m": float(margin),
            "minimum_vertex_index": values.index(least),
            "vertex_G_fractions": [str(g) for g in values],
            "proof": "G(s)=(n-q) dot (s-(n+q)/2)>0 on every vertex and hence on convex C. "
                     "For the actual source s_star in C, ||s_star-n||>R follows from the prior "
                     "same-channel no_signal; hence ||s_star-q||>||s_star-n||>R. "
                     "The outer polygon alone does not imply ||s-n||>R for every s in C.",
            "scope": "Exact inequality on supplied binary64 vertices; C retains frozen floating geometry contract"}


def draw(vertices, event, check, output_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Polygon, Rectangle
    import numpy as np

    options = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msjh.ttc"),
               Path("C:/Windows/Fonts/simhei.ttf")]
    font_file = next((p for p in options if p.exists()), None)
    if font_file is None:
        raise RuntimeError("A Chinese font is required; no installation is attempted")
    font = FontProperties(fname=str(font_file))
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "svg.fonttype": "path", "svg.hashsalt": "relative-silence-public-prefix-v1",
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    points = np.asarray(vertices)
    q, n = np.asarray(event["position"]), np.asarray(event["actual_negative_position"])
    delta, midpoint = n-q, (n+q)/2
    unit = delta / np.linalg.norm(delta)
    vertex = points[check["minimum_vertex_index"]]
    foot = vertex - np.dot(unit, vertex-midpoint)*unit
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 8.1), gridspec_kw={"width_ratios": [1.2, 1.]})
    fig.subplots_adjust(left=.065, right=.975, bottom=.27, top=.82, wspace=.22)
    blue, ink, orange = "#087f8c", "#243746", "#c56118"
    fig.suptitle("相对负反馈证书：省去一次可证明冗余的测量", x=.06, ha="left", y=.969,
                 fontsize=21, weight="bold", color=ink)
    fig.text(.061, .91, f"频道 {event['channel']} · C 有 {len(vertices)} 个顶点" +
             f"  |  仅使用第 {event['after_actual_action_count']} 个实际动作及其之前的公开观测",
             fontsize=11, color="#526471")
    all_points = np.vstack((points, q, n))
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center = (lo+hi)/2
    span = max(hi-lo)*1.2
    global_bounds = (center[0]-span/2, center[0]+span/2, center[1]-span/2, center[1]+span/2)
    local_span = max(280., check["minimum_signed_bisector_distance_m"]*7)
    local_center = (vertex+foot)/2 + np.array([local_span*.12, -local_span*.1])
    local_bounds = (local_center[0]-local_span/2, local_center[0]+local_span/2,
                    local_center[1]-local_span/2, local_center[1]+local_span/2)

    for ax, bounds in zip(axes, (global_bounds, local_bounds)):
        x0, x1, y0, y1 = bounds
        xx, yy = np.meshgrid(np.linspace(x0, x1, 180), np.linspace(y0, y1, 180))
        signed = unit[0]*(xx-midpoint[0]) + unit[1]*(yy-midpoint[1])
        ax.contourf(xx, yy, signed, levels=[-1e8, 0, 1e8], colors=["#fff3ef", "#ecf7f3"], zorder=0)
        ax.contour(xx, yy, signed, levels=[0], colors=[orange], linewidths=1.8, linestyles="--", zorder=2)
        ax.add_patch(Polygon(points, closed=True, facecolor="#34cbb4", edgecolor=blue, linewidth=2, alpha=.78, zorder=3))
        ax.scatter(points[:, 0], points[:, 1], s=23, c=blue, zorder=4)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x / m", ylabel="y / m")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#cdd8dc", linewidth=.6, alpha=.55, zorder=0)
        ax.tick_params(labelsize=9)
    axes[0].set_title("A  全局：整个可行区域位于有证书的一侧", loc="left", fontsize=12, pad=13)
    axes[1].set_title("B  局部：最靠近平分线的顶点仍有正间隔", loc="left", fontsize=12, pad=13)
    axes[0].scatter(*n, s=85, c="#bc4035", marker="s", zorder=5)
    axes[0].scatter(*q, s=95, c="#7352a2", marker="X", zorder=5)
    axes[0].plot([n[0], q[0]], [n[1], q[1]], color="#85909b", linewidth=1., zorder=1)
    axes[0].annotate(f"n：第 {event['witness_action_ordinal']} 个动作\n真实测得无信号", n,
                     xytext=(-4, -38), textcoords="offset points", ha="center", fontsize=10, color="#983329")
    axes[0].annotate("q：本来要查询的位置\n本次未发出该测量", q,
                     xytext=(13, 8), textcoords="offset points", fontsize=10, color="#62498a")
    axes[0].annotate("C：该频道源的可行区域\n整个区域都距 q 更远", points.mean(axis=0),
                     xytext=(55, -4), textcoords="offset points", arrowprops={"arrowstyle": "-", "color": blue},
                     fontsize=10, color=blue)
    axes[0].text(.025, .965, "粉色：距 q 比距 n 更近\n绿色：距 q 比距 n 更远", transform=axes[0].transAxes,
                 va="top", fontsize=9, color=ink, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    axes[0].add_patch(Rectangle((local_bounds[0], local_bounds[2]), local_span, local_span,
                               facecolor="none", edgecolor=ink, linewidth=1.2, linestyle=":"))
    axes[0].annotate("局部放大范围", (local_bounds[1], local_bounds[3]), xytext=(12, 13),
                     textcoords="offset points", fontsize=9, color=ink)
    axes[1].plot([vertex[0], foot[0]], [vertex[1], foot[1]], color=orange, linewidth=2.5, zorder=6)
    axes[1].scatter(*foot, s=28, color=orange, zorder=6)
    axes[1].annotate(f"最小垂距 {check['minimum_signed_bisector_distance_m']:.3f} m\n大于 0，整块 C 均满足",
                     (vertex+foot)/2, xytext=(40, 39), textcoords="offset points", fontsize=10,
                     color="#8b4815", arrowprops={"arrowstyle": "->", "color": orange})
    axes[1].annotate("最不利顶点", vertex, xytext=(-78, -12), textcoords="offset points",
                     fontsize=10, color=blue, arrowprops={"arrowstyle": "->", "color": blue})
    axes[1].text(.05, .91, "虚线：n 与 q 的垂直平分线", transform=axes[1].transAxes,
                 color="#8b4815", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    fig.text(.065, .157, r"实际源 $s_*\in C$ 满足：$\|s_*-q\|>\|s_*-n\|>R$，所以在 q 必然无信号。", fontsize=16, color=ink)
    fig.text(.065, .111, "左侧不等式在整个 C 上由顶点检验保证；右侧针对实际源，由此前同频道的真实无信号反馈保证。", fontsize=11, color="#526471")
    fig.text(.065, .071, "图中不含隐藏源位置；C 是观测约束的外包区域，并非真实源位置或点估计。虚线方向与距离均按原坐标绘制。", fontsize=10, color="#526471")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    fig.savefig(outputs[0], dpi=170, facecolor="white", metadata={"Title": "Public-prefix relative-silence certificate"})
    fig.savefig(outputs[1], facecolor="white", metadata={"Title": "Public-prefix relative-silence certificate", "Date": None})
    plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "font_file": str(font_file),
            "font_sha256": sha256(font_file), "global_bounds_m": list(global_bounds),
            "local_bounds_m": list(local_bounds), "output_sha256": {p.name: sha256(p) for p in outputs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--record", type=Path, default=root/"results/round2/derived_silence/model_smoke/combined.json.gz")
    parser.add_argument("--source-root", type=Path, default=root.parent/"q3-r2-derived-silence")
    parser.add_argument("--relative-event-index", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=root/"research/round2/figures/derived_silence_certificate")
    args = parser.parse_args()
    if args.relative_event_index < 0:
        parser.error("Event index must be nonnegative")
    event, preceding, actions, source_hashes = load_public_prefix(args.record, args.relative_event_index)
    vertices, used, required = reconstruct(event, preceding, actions, args.source_root, source_hashes)
    check = exact_check(vertices, event)
    result = {"kind": "experimental_public_prefix_geometry_illustration", "hidden_truth_access": False,
              "simulation_run": False, "selection_rule": "first recorded relative event by default; no outcome selection",
              "input_record": str(args.record.resolve()), "input_sha256": sha256(args.record),
              "script_sha256": sha256(Path(__file__)),
              "source_root": str(args.source_root.resolve()),
              "geometry_source_sha256": {p: source_hashes[p] for p in required},
              "frozen_candidate_source_sha256": source_hashes["src/strategies/derived_silence_state_search.py"],
              "relative_event_index": args.relative_event_index, "event": event,
              "actual_prefix_length": len(actions), "public_prefix_sha256": canonical_sha256(actions),
              "prior_inference_count": len(preceding), "same_channel_public_measurements": used,
              "public_region_vertices": [list(p) for p in vertices], "mathematical_check": check,
              "render": draw(vertices, event, check, args.output_prefix),
              "reproduce_from_q3_round2": "python experiments/plot_derived_silence_certificate.py"}
    output = args.output_prefix.with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "prefix": len(actions), "channel": event["channel"],
                      "witness_ordinal": event["witness_action_ordinal"], "vertices": len(vertices),
                      "minimum_signed_distance_m": check["minimum_signed_bisector_distance_m"],
                      "exact_vertex_inequality_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()

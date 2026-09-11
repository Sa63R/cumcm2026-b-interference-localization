"""Read-only bounded-label reconstruction for validation, never for fitting.

Two interior reception-radius choices per episode expose reconstruction
uncertainty. Final labels are evaluator-only and cannot enter policy state.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geometry import distance


def compatible(p, events):
    if math.hypot(*p) > 1800:
        return None
    low, high = 1000.0, 1500.0
    for e in events:
        q = (e["x_m"], e["y_m"])
        d = distance(p, q)
        response = json.loads(e["response_json"])
        result = response.get("measure_result", response.get("clear_result"))
        if result == "no_signal":
            high = min(high, d - 1e-7)
        elif result == "direction":
            low = max(low, d)
            bearing = math.degrees(math.atan2(p[1] - q[1], p[0] - q[0])) % 360
            if d <= 5 or abs((bearing - response["svd_deg"] + 180) % 360 - 180) > 1.005 + 1e-8:
                return None
        elif result == "near":
            if d > 5 + 1e-8:
                return None
            low = max(low, d)
        elif result == "success" and d > 20 + 1e-8:
            return None
        elif result == "no_target_in_range" and d <= 20:
            return None
    return (low, high) if low <= high else None


def witness(label, events, seed):
    center = (label["x_m"], label["y_m"])
    vertices = label["outer_vertices"]
    candidates = [center]
    if vertices:
        mean = tuple(sum(p[k] for p in vertices) / len(vertices) for k in (0, 1))
        candidates.append(mean)
        candidates += [tuple(0.999 * p[k] + 0.001 * mean[k] for k in (0, 1)) for p in vertices]
        rng = random.Random(seed)
        for _ in range(512):
            a, b = rng.choice(list(zip(vertices, vertices[1:] + vertices[:1])))
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            candidates.append(tuple(mean[k] + u * (a[k] - mean[k]) + v * (b[k] - mean[k]) for k in (0, 1)))
    for p in candidates:
        interval = compatible(p, events)
        if interval is not None:
            return p, interval, label["radius_m"] + distance(p, center)
    raise ValueError("No witness found inside bounded label; cannot reconstruct this episode")


def export(database, count, *, selection_salt="fresh-q3-v1:", excluded_evidence=(),
           case_prefix="validation"):
    """Reconstruct a deterministic validation selection without fitting.

    Defaults preserve the first-round case ordering, seeds, and case IDs.
    Alternative rounds may change only the hash salt, evidence exclusions,
    and public case prefix; witness construction remains identical.
    """
    excluded_evidence = frozenset(excluded_evidence)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    episodes = connection.execute("SELECT * FROM episodes WHERE problem=3 AND complete=1 AND cleared_count=source_total AND trajectory_complete=1").fetchall()
    eligible_before_exclusion = len(episodes)
    episodes = [e for e in episodes if e["evidence_sha256"] not in excluded_evidence]
    episodes = sorted(episodes, key=lambda e: hashlib.sha256((selection_salt + e["case_code"]).encode("utf-8")).hexdigest())
    selected = episodes[:count]
    cases, radii, baselines, evidence, anchors, failures = [], {}, {}, [], {}, []
    groups = []
    for index, e in enumerate(selected):
        labels = connection.execute("SELECT label_json FROM source_estimates WHERE episode_id=? ORDER BY channel", (e["id"],)).fetchall()
        steps = connection.execute("SELECT * FROM steps WHERE episode_id=? AND accepted=1 ORDER BY step_index", (e["id"],)).fetchall()
        sources, uncertainty, historic = [], {}, []
        try:
            if len(labels) != e["source_total"]:
                raise ValueError("Missing source labels")
            for item in labels:
                label = json.loads(item[0])
                if not label["geometry_consistent"] or not label["cleared"]:
                    raise ValueError("Inconsistent or incomplete labels")
                events = [s for s in steps if s["channel"] == label["channel"]
                          and s["step_index"] <= label["clear_step_index"]]
                p, interval, radius = witness(label, events, 923000 + index)
                sources.append((label["channel"], p, interval))
                uncertainty[str(label["channel"])] = radius
                historic += [dict(channel=s["channel"], position=[s["x_m"],s["y_m"]],
                                  response=json.loads(s["response_json"]))
                             for s in events if s["action"] == "measure"]
            # Keep no-signal observations from channels that were absent too.
            known = {s[0] for s in sources}
            historic += [dict(channel=s["channel"], position=[s["x_m"],s["y_m"]],
                              response=json.loads(s["response_json"]))
                         for s in steps if s["action"] == "measure" and s["channel"] not in known]
            for fraction, suffix in ((0.25, "radius-low"), (0.75, "radius-high")):
                case_id = f"{case_prefix}-{index:03d}-{suffix}"
                cases.append(dict(case_id=case_id, problem=3, seed=923000 + index,
                    error_mode="uniform", description="Bounded observation-label reconstruction; evaluator only",
                    sources=[dict(channel=c, x=p[0], y=p[1], reception_radius_m=lo + fraction * (hi - lo),
                                  orientation_deg=None) for c, p, (lo, hi) in sources]))
                radii[case_id] = uncertainty
                baselines[case_id] = dict(recorded_time_s=e["virtual_time_s"], recorded_policy=e["policy"],
                                          source_total=e["source_total"])
                anchors[case_id] = historic
            evidence.append(e["evidence_sha256"])
            groups.append(dict(group_id=f"{case_prefix}-{index:03d}", case_code=e["case_code"],
                               evidence_sha256=e["evidence_sha256"],
                               case_ids=[f"{case_prefix}-{index:03d}-{suffix}"
                                         for suffix in ("radius-low", "radius-high")]))
        except ValueError as exc:
            failures.append(dict(index=index, reason=str(exc)))
    connection.rollback()
    connection.close()
    return dict(cases=cases, metadata=dict(
        kind="bounded-label validation reconstruction, not official replay or ground truth",
        eligible_episodes=len(episodes), selected_episodes=len(selected), reconstructed_episodes=len(evidence),
        selected_by=f"first SHA256({selection_salt}case_code), all DB split labels validation-only",
        selection_salt=selection_salt, selection_encoding="utf-8",
        eligible_before_exclusion=eligible_before_exclusion,
        excluded_eligible_episodes=eligible_before_exclusion - len(episodes),
        excluded_evidence_sha256=sorted(excluded_evidence),
        selected_evidence_sha256=[e["evidence_sha256"] for e in selected],
        requested_groups=count, original_groups=groups, radius_fractions=[0.25, 0.75],
        evidence_sha256=evidence, failures=failures, uncertainty_radii=radii,
        historical_baselines=baselines, anchors=anchors,
        warning="New observation locations use the public-rule research error model. No exact counterfactual official scores are available."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    result = export(args.database, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f)
    print(json.dumps({k: v for k, v in result["metadata"].items()
                      if k in {"eligible_episodes", "selected_episodes", "reconstructed_episodes", "failures"}}))

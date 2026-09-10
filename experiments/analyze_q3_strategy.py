"""Q3 evidence and a constructed non-optimality witness; no official requests."""
import hashlib
import gzip
import json
import math
from pathlib import Path

from simulation import Source, Scenario, LocalResearchSimulator
from strategies import run_search


def main():
    root = Path(__file__).resolve().parents[1]
    # The harness constructs a legal scene; the strategy receives only client().
    sources = tuple(Source(i + 1, math.cos(2 * math.pi * i / 16),
                           math.sin(2 * math.pi * i / 16), 1000)
                    for i in range(16))
    scenario = Scenario("q3-counterexample-sixteen-near-origin", 3, 263,
                        sources, "zero", "Sixteen distinct sources within 1 m of origin")
    simulator = LocalResearchSimulator(scenario)
    with simulator.client() as client:
        search = run_search(client, problem=3, variant="adaptive", active_policy="center")
    evaluation = simulator.evaluation()
    successes = [a for a in search.action_history
                 if a["action"] == "clear" and a["result"] == "success"]
    cap_reached_at = successes[15]["virtual_time_s"]
    assert evaluation["all_cleared"] and len(successes) == 16
    assert abs(cap_reached_at - 199.0) < 1e-6
    assert abs(search.virtual_time_s - 2137.0) < 1e-6

    target_radius, reception = 1800.0, 1000.0
    a_min = target_radius * math.sqrt(3) / 2 - math.sqrt(
        reception**2 - target_radius**2 / 4)
    a_max = math.sqrt(3) * reception
    ring_results = []
    for a in (a_min, 1150., 1200., 1500., 900 * math.sqrt(3)):
        worst = max(a / math.sqrt(3), math.sqrt(
            target_radius**2 + a*a - math.sqrt(3)*a*target_radius))
        ring_results.append({"ring_radius_m": a, "worst_coverage_distance_m": worst,
                             "coverage_margin_m": reception - worst,
                             "open_skeleton_length_m": 6*a,
                             "open_skeleton_movement_s": 6*a/5})

    directory = root / "results/sessions/practice/20260910T122405844918Z"
    official = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    bounds = json.loads((directory / "lower_bounds.json").read_text(encoding="utf-8"))
    groups = json.loads((root / "results/study/summary.json").read_text(encoding="utf-8"))["groups"]
    savings = []
    for trace_path in sorted((root / "results/study/traces").glob("q3-random-*--adaptive_center.json.gz")):
        # Ignore the evaluation block: only submitted actions and accepted
        # clear feedback are needed for this exact prefix truncation.
        trace = json.loads(gzip.decompress(trace_path.read_bytes()))["search"]
        cleared = set()
        first_cap_time = None
        for action in trace["action_history"]:
            if action["action"] == "clear" and action["result"] == "success":
                cleared.add(action["channel"])
                if len(cleared) == 16:
                    first_cap_time = action["virtual_time_s"]
                    break
        saved = trace["virtual_time_s"] - first_cap_time if first_cap_time is not None else 0.0
        savings.append({"trace": trace_path.name, "cap_reached": first_cap_time is not None,
                        "current_time_s": trace["virtual_time_s"], "saved_time_s": saved})
    if len(savings) != 100:
        raise ValueError("Expected the existing 100 paired Q3 random traces")
    saved_total = sum(item["saved_time_s"] for item in savings)
    current_total = sum(item["current_time_s"] for item in savings)
    result = {
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src").rglob("*.py"))},
        "strict_nonoptimality_witness": {
            "origin": "constructed_local_research_case_not_official",
            "sources": 16, "source_channels": list(range(1,17)),
            "distance_from_origin_m": 1,
            "current_policy_time_s": search.virtual_time_s,
            "same_prefix_with_valid_cap_exit_time_s": cap_reached_at,
            "avoidably_spent_after_16_successes_s": search.virtual_time_s - cap_reached_at,
            "proof": "At 16 distinct accepted successes, the public upper source count certifies all-clear; exit adds zero virtual time.",
            "production_policy_changed": False,
        },
        "ring_family_feasible_radius_interval_m": [a_min, a_max],
        "existing_random_trace_cap_exit_analysis": {
            "runs": len(savings), "cap_reached_runs": sum(item["cap_reached"] for item in savings),
            "strict_saving_runs": sum(item["saved_time_s"] > 0 for item in savings),
            "total_saved_s": saved_total, "mean_saved_s": saved_total / len(savings),
            "mean_before_s": current_total / len(savings),
            "mean_after_prefix_truncation_s": (current_total - saved_total) / len(savings),
            "relative_mean_time_reduction_pct": 100 * saved_total / current_total,
            "method": "Exact truncation of legal action prefixes once 16 different source clearances succeeded; no new policy rerun.",
            "per_case": savings,
        },
        "ring_family": ring_results,
        "ring_caveat": "Shorter coverage skeleton alone does not prove shorter full adaptive search.",
        "official_instance": {
            "source_total_gui_verified": 13,
            "time_s": official["state"]["virtual_time_s"],
            "guaranteed_strategy_optimum_lower_s": bounds["conditional_guaranteed_all_clear_lower_s"],
            "a_posteriori_approximation_ratio_upper": official["state"]["virtual_time_s"] / bounds["conditional_guaranteed_all_clear_lower_s"],
            "time_breakdown_s": official["search"]["time_breakdown"],
            "scope": "This instance, guaranteed-all-clear strategies with only the public count bound; neither a uniform competitive ratio nor an achievable speedup.",
        },
        "q3_random_study": [{k: g[k] for k in (
            "strategy", "runs", "all_clear_runs", "mean_virtual_time_s", "mean_measurements",
            "relative_mean_time_reduction_pct", "paired_mean_saving_ci95_s", "mean_time_breakdown_s")}
            for g in groups if g["problem"] == 3 and g["case_kind"] == "random"],
    }
    target = root / "results/validation/q3_strategy_analysis.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"source_sha256", "q3_random_study", "existing_random_trace_cap_exit_analysis"}}, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result["existing_random_trace_cap_exit_analysis"].items()
                      if k != "per_case"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

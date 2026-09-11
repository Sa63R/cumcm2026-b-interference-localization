"""One repeated training scene: exact cache on/off equivalence, no weight update."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from q4_rl.micro_controller import Q4MicroSearch
from q4_rl.micro_network import configure_cpu, load_policy
from q4_rl.scenarios import build_case
from q4_rl.train import training_case_spec
from simulation import LocalResearchSimulator
from experiments.q4_comparison_bounds import common_bound


def diagnose(checkpoint, expected_sha256):
    configure_cpu()
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Frozen BC content mismatch")
    reference = []  # In memory only; never serialize candidate matrices as logs.
    action_indices, reports, results = [], [], []
    lower = None
    for enabled in (False, True):
        base_policy = load_policy(checkpoint, deterministic=True)
        indices = []
        def policy(global_features, candidate_features):
            selected = base_policy(global_features, candidate_features)
            base_policy.records.clear()
            indices.append(selected)
            return selected

        class ObservedSearch(Q4MicroSearch):
            compared = 0
            comparison_wall_s = 0.

            def _features(self, candidates):
                features = super()._features(candidates)
                started = time.perf_counter()
                if enabled:
                    assert (candidates, features) == reference[self.compared], "Candidate/feature mismatch"
                else:
                    reference.append((candidates, features))
                self.compared += 1
                self.comparison_wall_s += time.perf_counter()-started
                return features

        simulator = LocalResearchSimulator(build_case(8006000, split="train", **training_case_spec(8006000)),
                                            max_real_duration_s=180.)
        search = ObservedSearch(simulator.client(), policy=policy, decision_cache=enabled,
                                max_decisions=512, record_transitions=False,
                                action_deadline_epoch=time.time()+175.)
        started, cpu_started = time.perf_counter(), time.process_time()
        report = search.run()
        elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu_started
        evaluation = simulator.evaluation()
        assert evaluation["all_cleared"] and report.completion_certified_under_model
        assert not report.error and not report.exit_error
        if lower is None:
            lower = common_bound(evaluation["ground_truth"])["common_lower_bound_s"]
        assert search.compared == len(reference)
        if enabled:
            assert indices == action_indices
            assert report.action_history == reports[0].action_history
            assert report.learning["micro_steps"] == reports[0].learning["micro_steps"]
            assert report.learning["certified_clear_checks"] == reports[0].learning["certified_clear_checks"]
            for key in ("decision_cost_s", "fallback_cost_s", "uncovered_cost_s", "total_billed_cost_s"):
                assert report.learning[key] == reports[0].learning[key], key
            assert report.virtual_time_s == reports[0].virtual_time_s
        else:
            action_indices = indices
        results.append({"decision_cache": enabled, "compared_decisions": search.compared,
            "controller_wall_s": elapsed, "controller_cpu_s": cpu,
            "comparison_wall_s": search.comparison_wall_s,
            "wall_without_comparison_s": elapsed-search.comparison_wall_s,
            "feature_wall_without_comparison_s": report.learning["feature_wall_time_s"]-search.comparison_wall_s,
            "inference_wall_s": report.learning["inference_wall_time_s"],
            "actual_time_s": report.virtual_time_s, "common_lower_bound_s": lower,
            "time_over_lower_bound": report.virtual_time_s/lower,
            "all_cleared": evaluation["all_cleared"], "failed_clear_count": evaluation["failed_clear_count"],
            "fallback_cost_s": report.learning["fallback_cost_s"]})
        reports.append(report)
    return {"scope": "single repeated training scene CPU diagnostic; not efficacy evidence",
            "seed": 8006000, "checkpoint_sha256": digest, "exact_candidates_features_actions_billing": True,
            "runs": results,
            "wall_speedup_without_comparison": results[0]["wall_without_comparison_s"]/results[1]["wall_without_comparison_s"],
            "raw_cpu_speedup_with_comparison": results[0]["controller_cpu_s"]/results[1]["controller_cpu_s"],
            "timing_limit": "one off/on pair in fixed order; no statistical speed claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Diagnostic output must be new")
    result = diagnose(args.checkpoint, args.sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))

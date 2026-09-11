"""Implementation smoke only on an already-opened development case.

This is not independent performance evidence. Snapshot changing source bytes
with each invocation; do not overwrite records or inspect the validation DB.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "q3-r2-observation-tree"
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def source_hashes():
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO / "src").rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite model development evidence")
    source_before = source_hashes()
    harness_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from simulation import LocalResearchSimulator, random_scenario
    from experiments.training_stress_reliability import ObservationOnlyClient, audit_actions
    from strategies.observation_tree_state_search import run_observation_tree_state_search
    simulator = LocalResearchSimulator(random_scenario(3, 200114),
        max_real_duration_s=1200, max_virtual_duration_s=360000)
    client = ObservationOnlyClient(simulator.client())
    config = json.loads((REPO / "experiments/state_search_candidate_relocating_cover_v1.json").read_text())["kwargs"]["config"]
    started = time.perf_counter()
    result = run_observation_tree_state_search(client, config=config,
        tree_config={"depth": args.depth, "max_searches": 1, "decision_seconds": args.seconds},
        enabled=not args.disabled)
    simulator.finish_for_evaluation()
    evaluation, history = simulator.evaluation(), simulator.observation_history()
    source_after = source_hashes()
    old_audit = json.loads((ROOT / "research/round2/pilot_audit.json").read_text(encoding="utf-8"))
    bound = old_audit["bounds_by_seed"]["200114"]
    canonical = json.dumps(evaluation["ground_truth"], sort_keys=True, ensure_ascii=False,
                           allow_nan=False, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != bound["case_sha256"]:
        raise ValueError("Previously audited development case does not match")
    record = {"scope": "Unfrozen implementation smoke on already-opened development 200114; not independent evidence",
        "source_sha256": source_before, "source_after_sha256": source_after,
        "source_unchanged_during_smoke": source_before == source_after,
        "harness_sha256": harness_before,
        "arguments": {**vars(args), "output": str(args.output)}, "summary": result.as_dict(),
        "evaluation": evaluation, "history": history, "audit": audit_actions(history, evaluation),
        "elapsed_seconds": time.perf_counter()-started,
        "physical_clairvoyant_lower_s": bound["physical_clairvoyant_lower_s"],
        "time_over_physical_lower": result.virtual_time_s / bound["physical_clairvoyant_lower_s"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
    print(json.dumps({"completed": evaluation["all_cleared"], "certified": result.completion_certified_under_model,
        "failed_clears": evaluation["failed_clear_count"], "virtual_time_s": result.virtual_time_s,
        "physical_lower_s": record["physical_clairvoyant_lower_s"], "time_over_lower": record["time_over_physical_lower"],
        "elapsed_seconds": record["elapsed_seconds"], "planning": result.strategy_parameters["observation_tree_log"]},
        allow_nan=False))


if __name__ == "__main__":
    main()

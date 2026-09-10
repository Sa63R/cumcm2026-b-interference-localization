"""Training-only CPU cost and exact neutral transfer audit; no policy fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from research_rl.joint_scan import JointScanRLSearch
from research_rl.route_debt import RouteDebtRLSearch
from research_rl.network import CandidateActorCritic, TorchPolicy, pack_observations
from research_rl.portable_checkpoint import model_tensor_digest
from research_rl.train import initialize_from, source_manifest
from simulation import LocalResearchSimulator, random_scenario
from tests.test_strategy import ObservationOnlyClient


class TimedFeatures:
    def _features(self, candidates, remaining):
        started = time.perf_counter()
        result = super()._features(candidates, remaining)
        self.feature_s += time.perf_counter() - started
        return result


class TimedV3(TimedFeatures, JointScanRLSearch):
    pass


class TimedV4(TimedFeatures, RouteDebtRLSearch):
    pass


def run(model, version, seed):
    simulator = LocalResearchSimulator(random_scenario(3, seed))
    records = []
    def record(f, c, a, t, s, cost):
        records.append(dict(features=np.asarray(f, dtype=np.float32),
            context=np.asarray(c, dtype=np.float32), action=a, teacher=t, cost=cost))
    cls = TimedV3 if version == "v3" else TimedV4
    control = cls(ObservationOnlyClient(simulator.client()),
                  TorchPolicy(model, capture_diagnostics=True), feature_version=version,
                  recorder=record)
    control.feature_s = 0.0
    started = time.perf_counter()
    report = control.run()
    elapsed = time.perf_counter() - started
    evaluation = simulator.evaluation()
    assert evaluation["all_cleared"] and not evaluation["failed_clear_count"]
    assert report.completion_certified_under_model
    return report, records, elapsed, control.feature_s


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("research/route_debt_evidence/migration_cpu.json"))
    parser.add_argument("--seed-start", type=int, default=110501)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.count < 32 or args.repeats < 2 or not 100001 <= args.seed_start <= args.seed_start + args.count - 1 <= 199999:
        parser.error("Use at least 32 training-only worlds and two alternating repeats")
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert payload["feature_schema"]["version"] == "v3"
    old = CandidateActorCritic(payload["hidden"], 60).eval()
    old.load_state_dict(payload["model"])
    new = CandidateActorCritic(payload["hidden"], 76).eval()
    initialize_from(new, payload)
    models = {"v3": old, "v4": new}
    # Warm both kernels; excluded from benchmark rows, still training-only worlds.
    for version in models:
        run(models[version], version, args.seed_start)
    rows = []
    for repeat in range(args.repeats):
        for seed in range(args.seed_start, args.seed_start + args.count):
            order = ["v3", "v4"] if (repeat + seed) % 2 else ["v4", "v3"]
            results = {version: run(models[version], version, seed) for version in order}
            left, a, _, _ = results["v3"]
            right, b, _, _ = results["v4"]
            assert left.action_history == right.action_history
            assert left.virtual_time_s == right.virtual_time_s and len(a) == len(b)
            for x, y in zip(a, b):
                assert np.array_equal(x["features"], y["features"][:, :60])
                assert np.array_equal(x["context"], y["context"])
                assert (x["action"], x["teacher"], x["cost"]) == (y["action"], y["teacher"], y["cost"])
                assert np.isfinite(y["features"]).all()
            with torch.no_grad():
                original = old(*pack_observations(a))
                migrated = new(*pack_observations(b))
                assert all(torch.equal(x, y) for x, y in zip(original, migrated))
            rows.append(dict(seed=seed, repeat=repeat, execution_order=order,
                v3_wall_s=results["v3"][2], v4_wall_s=results["v4"][2],
                v3_feature_s=results["v3"][3], v4_feature_s=results["v4"][3],
                decisions=len(a), max_candidates=max(len(r["features"]) for r in a),
                virtual_time_s=left.virtual_time_s, exact_actions_prefix_logits_value=True))
    summary = {key: statistics.mean(row[key] for row in rows) for key in
               ("v3_wall_s", "v4_wall_s", "v3_feature_s", "v4_feature_s", "decisions")}
    summary["wall_ratio_v4_over_v3"] = summary["v4_wall_s"] / summary["v3_wall_s"]
    summary["paired_mean_extra_ms"] = 1000 * statistics.mean(row["v4_wall_s"]-row["v3_wall_s"] for row in rows)
    result = dict(schema=1, scope="CPU float32, one Torch thread; two runs per version per training world",
        limitation="Numerical equality is tested on this device/kernel; timing includes equal record/diagnostic overhead, excludes batched parity checking and scenario construction. This is not an improved-policy result.",
        checkpoint_name=args.checkpoint.name,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        model_tensor_digest=model_tensor_digest(payload["model"]),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_manifest=source_manifest(), torch_version=torch.__version__, numpy_version=np.__version__,
        cases=args.count, repeats=args.repeats, seed_range=[args.seed_start,args.seed_start+args.count-1],
        all_exact=True, summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

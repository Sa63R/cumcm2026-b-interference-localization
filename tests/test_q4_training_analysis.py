"""Integrity tests for the offline diagnostic; no simulator or training run."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "q4_analyze_training", Path(__file__).resolve().parents[1] / "scripts" / "q4_analyze_training.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def legacy_episode(*, success=True):
    penalty = 5. if success else 360000.
    return {"seed": 8000000, "records": [{"global_features": [0.] * 10,
        "candidate_features": [[1., 0., 0., 0., 0., .005, 1., 1.] + [0.] * 8],
        "action_index": 0, "action_kind": "measure", "cost_s": 5., "fallback_cost_s": 0.,
        "return": -penalty / 1000., "terminal_penalty_s": penalty - 5.}],
        "metrics": {"success": success, "actual_time_s": 5., "penalized_time_s": penalty,
                    "reward_cost_s": penalty, "failed_clear_count": 0, "fallback_cost_s": 0.}}


class TrainingAnalysisTests(unittest.TestCase):
    def test_failed_episode_is_retained_in_total_cost(self):
        rows = [review.analyze_episode(legacy_episode(success=success), batch_label="fixture")
                for success in (True, False)]
        result = review.summary_rows(rows)
        self.assertEqual(result["available_episode_metrics"], 2)
        self.assertEqual(result["reported_task_failures"], 1)
        self.assertEqual(result["mean_penalized_time_s"], 180002.5)
        self.assertIsNone(result["actual_all_clear_ratio_of_sums"])

    def test_legacy_missing_evidence_is_not_fabricated(self):
        row = review.analyze_episode(legacy_episode(), batch_label="fixture")
        self.assertEqual(row["physical_audit"]["status"], "missing_evidence")
        result = review.summary_rows([row])
        self.assertEqual(result["physically_verified_full_clear"], 0)
        self.assertEqual(result["physical_evidence_missing"], 1)
        self.assertIsNone(result["physical"])
        self.assertIsNone(result["mean_lower_bound_s"])

    def test_inconsistent_episode_does_not_create_favorable_subset(self):
        valid = review.analyze_episode(legacy_episode(), batch_label="fixture")
        invalid = {"errors": [{"type": "ValueError"}], "reported_metrics": {"actual_time_s": 999.}}
        summary = review.summary_rows([valid, invalid])
        self.assertEqual(summary["attempts"], 2)
        self.assertEqual(summary["aggregation_status"], "withheld_incomplete_or_inconsistent_batch")
        self.assertNotIn("mean_actual_time_s", summary)

    def test_billing_and_discount_tampering_detected(self):
        for key, value in (("cost_s", 4.), ("return", -.0049), ("terminal_penalty_s", 1.)):
            episode = legacy_episode()
            episode["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                review.analyze_episode(episode, batch_label="fixture")

    def test_public_float32_equivalence_is_conservative(self):
        features = [0.] * 16
        features[4] = 1.
        indistinguishable = copy.deepcopy(features)
        indistinguishable[4] += 1e-9
        distinguishable = copy.deepcopy(features)
        distinguishable[7] = 1.
        self.assertEqual(review.feature_key(features), review.feature_key(indistinguishable))
        self.assertNotEqual(review.feature_key(features), review.feature_key(distinguishable))

    def test_critic_constant_targets_do_not_get_fake_perfect_ev(self):
        self.assertIsNone(review.explained_variance([1., 1.], [1., 1.])["value"])
        self.assertEqual(review.explained_variance([1., 2., 3.], [1., 2., 3.])["value"], 1.)


if __name__ == "__main__":
    unittest.main()

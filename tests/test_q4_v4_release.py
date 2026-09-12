"""Offline release checks for V4 and its exact computation acceleration.

Place this file under the repository's tests/ directory and run it with
unittest discovery. SyntheticClient feeds a local fixture through the real
protocol client's validation; these tests never contact an official simulator.
All request logs use temporary directories and are removed after each test.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
CODE = REPOSITORY
sys.path[:0] = [
    str(CODE / "experiments" / "q4_speedup"),
    str(CODE / "experiments" / "q4_official_practice"),
]

import bootstrap
from bootstrap import b, make_case, RoundedSimulator
from compute_fast import ExactComputeCache
from coverage_compute_fast import CoverageComputeCache
import online
import q4_coverage
import q4_information
import q4_route_cached
import q4_v3_local
import q4_v4_local
import runtime
import run_speedup
from test_adapter import SyntheticClient


def _function_aliases():
    return (
        b.enclosing_circle,
        b.clip_halfplane,
        q4_information.expected_radius_gain,
        online.expected_radius_gain,
        q4_v3_local.optical_cover,
        q4_v4_local.optical_cover,
        q4_route_cached.multi_route,
        online.multi_route,
        q4_coverage.certify,
        runtime.certify,
    )


def _policy_without_diagnostic_time(policy):
    result = dict(policy)
    certificate = result.get("coverage_certificate")
    if certificate is not None:
        result["coverage_certificate"] = {
            key: value for key, value in certificate.items() if key != "wall"
        }
    return result


class V4ReleaseTests(unittest.TestCase):
    def test_complete_sessions_are_identical_for_true_emission_extremes(self):
        # The historical baseline generator clamps the type fraction. The
        # bootstrap fixture explicitly corrects the 0 and 1 extremes here.
        scenarios = [
            ("all_omni", 0.0, "uniform", "hash", 236800000),
            ("all_directional", 1.0, "uniform", "plus", 236800001),
            ("boundary_directional", 1.0, "outward_boundary", "minus", 236800002),
        ]
        self.assertEqual(set(run_speedup.METHODS), {"v4", "v4_fast"})
        certified_cases = 0
        with tempfile.TemporaryDirectory(prefix="q4-release-") as directory:
            for name, fraction, placement, error, seed in scenarios:
                paired = {}
                for method in ("v4", "v4_fast"):
                    with self.subTest(scenario=name, method=method):
                        targets = make_case(seed, fraction, placement)
                        self.assertTrue(all(
                            (target.direction is None) == (fraction == 0.0)
                            for target in targets
                        ))
                        simulator = RoundedSimulator(targets, seed, error, trace=True)
                        state, metadata = run_speedup.prepare_speedup_policy(method, seed + 71)
                        original_aliases = _function_aliases()
                        report = dict(metadata, status="not_entered", mode="synthetic_test_only")
                        request_log = Path(directory) / f"{name}_{method}.jsonl"
                        with SyntheticClient(simulator, request_log) as client:
                            run_speedup.run_speedup_session(client, state, report)
                            self.assertEqual(client.state.session, "exited")
                            self.assertIsNone(client.pending_request)
                            self.assertEqual(client.state.cleared_count, len(targets))
                            self.assertEqual(simulator.clear_count, len(targets))
                            self.assertAlmostEqual(
                                client.state.virtual_time_s, simulator.virtual_seconds
                            )
                            self.assertEqual(report["status"], "policy_completed_and_exited")
                            policy = report["policy"]
                            if len(targets) < 16:
                                self.assertTrue(policy["coverage_certificate"]["ok"])
                                if method == "v4":
                                    certified_cases += 1
                            else:
                                self.assertEqual(policy["stop_certificate"], "source_upper_bound")
                            paired[method] = {
                                "trace": simulator.trace[:],
                                "summary": simulator.summary(),
                                "policy": _policy_without_diagnostic_time(policy),
                                "virtual_time_s": client.state.virtual_time_s,
                            }
                        self.assertEqual(_function_aliases(), original_aliases)
                        self.assertTrue(request_log.is_file())
                self.assertEqual(paired["v4"], paired["v4_fast"], name)
        self.assertGreater(certified_cases, 0, "The fixture must exercise the coverage proof")

    def test_compute_cache_exception_restores_aliases_and_releases_entries(self):
        original_aliases = _function_aliases()
        cache = ExactComputeCache(
            polygon_limit=2, gain_limit=2, route_limit=2, optical_limit=2
        )
        with self.assertRaisesRegex(LookupError, "intentional compute interruption"):
            with cache:
                self.assertIsNot(b.enclosing_circle, original_aliases[0])
                for index in range(5):
                    b.enclosing_circle([(0.0, 0.0), (float(index + 1), 0.0), (0.0, 1.0)])
                self.assertEqual(cache.stats()["circle"]["currsize"], 2)
                polygon = [(0.0, 0.0), (50.0, 0.0), (50.0, 1.0), (0.0, 1.0)]
                expected = q4_v3_local.optical_cover(polygon)
                mutated = q4_v3_local.optical_cover(polygon)
                mutated.reverse()
                mutated.append((999.0, 999.0))
                self.assertEqual(q4_v3_local.optical_cover(polygon), expected)
                with self.assertRaises(RuntimeError):
                    with ExactComputeCache():
                        pass
                raise LookupError("intentional compute interruption")
        self.assertEqual(_function_aliases(), original_aliases)
        with cache:
            self.assertTrue(all(info["currsize"] == 0 for info in cache.stats().values()))
        self.assertEqual(_function_aliases(), original_aliases)

    def test_coverage_cache_exception_restores_aliases_and_preserves_proof(self):
        original_aliases = _function_aliases()
        sites = [(0.0, 0.0)]
        baseline = q4_coverage.certify(sites, keep_leaves=True)
        normalize = lambda result: {key: value for key, value in result.items() if key != "wall"}
        cache = CoverageComputeCache(cache_limit=2)
        with self.assertRaisesRegex(LookupError, "intentional proof interruption"):
            with cache:
                self.assertIsNot(q4_coverage.certify, original_aliases[-2])
                self.assertIs(runtime.certify, q4_coverage.certify)
                accelerated = q4_coverage.certify(sites, keep_leaves=True)
                self.assertEqual(normalize(accelerated), normalize(baseline))
                self.assertFalse(accelerated["ok"])
                self.assertEqual(cache.stats()["calls"], 1)
                self.assertLessEqual(cache.stats()["peak_call_entries"], 2)
                with self.assertRaises(RuntimeError):
                    with CoverageComputeCache():
                        pass
                raise LookupError("intentional proof interruption")
        self.assertEqual(_function_aliases(), original_aliases)
        with cache:
            self.assertEqual(normalize(q4_coverage.certify(sites)), normalize(baseline))
        self.assertEqual(_function_aliases(), original_aliases)

    def test_missing_practice_flag_never_constructs_client(self):
        with mock.patch.object(run_speedup, "SimulatorClient") as factory:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_speedup.main(["--robot-id", "synthetic-only", "--method", "v4_fast"])
            self.assertEqual(error.exception.code, 2)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Refactor equivalence, independent retained planning and removed dependencies."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import candidates as c
from candidates import bench
import ablation_reference as old
import q4_v6_lite as lite
from jammers_local.core import Scenario, Session, LocalClient


def session(case, solver):
    s = Session(Scenario.from_dict(case['scenario']))
    client = LocalClient(s.dispatch)
    client.enter()
    report = solver(bench.Device(client))
    client.exit()
    bench.audit(s, report)
    trace = [(r['path'], r['request'], {k: v for k, v in r['response'].items()
              if k not in ('real_timestamp_ms', 'remaining_real_duration_s')}) for r in s.history]
    return trace, report


class LiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench.verify_sources()
        bench.initialize(1.005)
        cls.cases = json.loads((c.ROOT / 'previous_ablation_plan.json').read_text())['cases']

    def test_exact_combination_trace(self):
        evidence = []
        config = replace(old.FULL, route=False, rb_sharing=False)
        for case in [self.cases[i] for i in [0, 96, 198, 300, 335]]:
            with self.subTest(case=case['key']):
                expected, _ = session(case, lambda device: old.solve(device, config))
                actual, report = session(case, lite.solve_v6_lite)
                self.assertEqual(expected, actual)
                evidence.append(dict(case_key=case['key'], actions=len(actual), full_trace_equal=True,
                                     planner_calls=report['planner_calls'], transit_stops=report['transit_stops']))
        bench.dump(c.ROOT / 'implementation_equivalence.json', evidence)

    def test_rollout_and_transit_retained_without_route_or_rb_engine(self):
        with patch.object(old, 'RBEngine', side_effect=AssertionError('Removed engine called')):
            with patch.object(lite, 'Critic', wraps=lite.Critic) as loaded:
                _, report = session(self.cases[0], lite.solve_v6_lite)
        self.assertEqual(loaded.call_count, 1)
        self.assertTrue(loaded.call_args.args[0].endswith('transit_critic_big_extra.json'))
        self.assertGreater(report['planner_calls'], 0)
        self.assertGreater(report['transit_stops'], 0)
        self.assertGreaterEqual(report['activation']['visited_stations'], 4)

    def test_posterior_failure_keeps_geometry_completion(self):
        with patch.object(lite.SceneFactory, 'pool', side_effect=lite.PosteriorUnavailable('test unavailable')):
            _, report = session(self.cases[0], lite.solve_v6_lite)
        self.assertGreater(report['planner_failures']['test unavailable'], 0)
        self.assertIn(report['stop_certificate'], ['coverage_complete', 'source_upper_bound'])

    def test_removed_modules_not_needed_by_standalone_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'session.json'
            result = subprocess.run([sys.executable, str(c.ROOT / 'run_lite_local.py'),
                                     '--seed', 'q4-lite-import-smoke', '--out', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output.read_text())
        self.assertTrue(data['all_cleared'])
        self.assertEqual(data['removed_modules_loaded'], [])

    def test_invalid_config_rejected_before_device_access(self):
        for values in [dict(minimum_stations=-1), dict(max_transit_stops=65),
                       dict(scenes=0), dict(total_planning_seconds=float('nan'))]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    lite.solve_v6_lite(None, replace(lite.default_config(), **values))


if __name__ == '__main__':
    unittest.main(verbosity=2)

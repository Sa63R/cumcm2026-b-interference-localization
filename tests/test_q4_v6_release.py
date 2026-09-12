"""Run the legacy-named V6 modules in isolated processes, preserving V4 imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT/'experiments/q4_v6'


@unittest.skipUnless(hasattr(__import__('signal'), 'SIGALRM'), 'Local harness requires macOS/Linux SIGALRM')
class V6ReleaseTests(unittest.TestCase):
    def run_python(self, *arguments, env=None):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
            text=True, capture_output=True, timeout=180, env=env)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        return result

    def test_frozen_sources_and_published_v4_bridge(self):
        with tempfile.TemporaryDirectory(prefix='q4-v6-bridge-') as directory:
            env = dict(os.environ, Q4_V6_VALIDATION_OUTPUT=directory)
            self.run_python(V6/'test_bridge.py', env=env)
            evidence = json.loads((Path(directory)/'published_v4_equivalence.json').read_text())
            self.assertEqual(len(evidence), 3)
            self.assertTrue(all(r['trace_equal'] for r in evidence))

    def test_new_session_reproduces_frozen_v6_case(self):
        plan = json.loads((V6/'plan.json').read_text())
        case = plan['cases'][0]
        rows = [json.loads(line) for line in (V6/'main/records.jsonl').read_text().splitlines()]
        expected = next(r for r in rows if r['case_key']==case['key'] and r['method']=='v6')
        with tempfile.TemporaryDirectory(prefix='q4-v6-run-') as directory:
            output = Path(directory)/'run'
            self.run_python(V6/'run_local.py', '--method', 'v6', '--seed',
                            case['scenario']['label'], '--output', output)
            actual = json.loads((output/'summary.json').read_text())[0]
            self.assertIsNone(actual['error'])
            self.assertTrue(actual['all_cleared'])
            self.assertEqual(actual['virtual_time_us'], expected['virtual_time_us'])
            self.assertEqual(actual['action_sha256'], expected['action_sha256'])
            self.assertEqual(actual['cleared_count'], expected['cleared_count'])


if __name__ == '__main__':
    unittest.main()

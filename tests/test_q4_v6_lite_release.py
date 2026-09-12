"""Exercise the shipped Lite CLI in isolation from legacy top-level modules."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LITE = ROOT / 'experiments/q4_v6_lite'


class V6LiteReleaseTests(unittest.TestCase):
    def test_packaged_cli_matches_frozen_random_and_regression_cases(self):
        plan = json.loads((LITE / 'plan.json').read_text())
        records = [json.loads(line) for line in (LITE / 'main/records.jsonl').read_text().splitlines()]
        index = {(r['case_key'], r['method']): r for r in records}
        for key in ['main-0000', 'main-0066']:
            case = next(c for c in plan['cases'] if c['key'] == key)
            expected = index[key, 'lite']
            with self.subTest(case=key), tempfile.TemporaryDirectory(prefix='q4-v6-lite-release-') as directory:
                output = Path(directory) / 'session.json'
                result = subprocess.run([sys.executable, str(LITE / 'run_lite_local.py'),
                                         '--seed', case['scenario']['label'], '--out', str(output)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
                actual = json.loads(output.read_text())
                self.assertTrue(actual['all_cleared'])
                self.assertEqual(actual['removed_modules_loaded'], [])
                self.assertEqual(actual['snapshot']['virtual_time_us'], expected['virtual_time_us'])
                canonical = [dict(path=r['path'], request=r['request'],
                                  response={k: v for k, v in r['response'].items()
                                            if k not in ('real_timestamp_ms', 'remaining_real_duration_s')})
                             for r in actual['session']['history']]
                digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, expected['action_sha256'])


if __name__ == '__main__':
    unittest.main()

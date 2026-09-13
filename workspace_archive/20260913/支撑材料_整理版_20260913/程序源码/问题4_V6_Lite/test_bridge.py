"""Offline HTTP bridge checks; never contacts the official simulator."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'checks/simulator')]
import run_official as runner
from simulator_client import SimulatorClient
from simulator_client.state import ClientState
from jammers_local.core import Scenario
from jammers_local.server import Application, Server


def trace_hash(history):
    trace = [dict(path=r['path'], position=r['request'].get('position'), channel=r['request'].get('channel'),
                  response={k: v for k, v in r['response'].items()
                            if k not in ('real_timestamp_ms', 'remaining_real_duration_s')}) for r in history]
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()


class BridgeTests(unittest.TestCase):
    def test_http_matches_four_frozen_lite_traces(self):
        sites, certificate, _ = runner.prepare()
        evidence = []
        cases = json.loads((ROOT / 'checks/cases.json').read_text(encoding='utf-8'))
        for case in cases:
            with self.subTest(case=case['key']):
                app = Application(Scenario.from_dict(case['scenario']))
                server = Server(app, 0)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                report = {}
                try:
                    with SimulatorClient('local-test', base_url=f'http://127.0.0.1:{server.server_port}',
                                         log_path=OUTPUT / (case['key'] + '.jsonl')) as client:
                        runner.run_session(client, report, sites, certificate)
                    self.assertEqual(trace_hash(app.session.history), case['trace_sha256'])
                    self.assertEqual(app.session.engine.time_us, case['virtual_time_us'])
                    self.assertEqual(report['status'], 'policy_completed_and_exited')
                    self.assertEqual(len(app.session.engine.cleared), len(app.session.engine.sources))
                    evidence.append(dict(case=case['key'], full_trace_equal=True,
                                         actions=len(app.session.history), virtual_time_us=case['virtual_time_us']))
                finally:
                    server.shutdown()
                    server.server_close()
                    worker.join()
        (OUTPUT / 'http_equivalence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')

    def test_missing_robot_id_cannot_connect(self):
        with patch.object(runner, 'SimulatorClient', side_effect=AssertionError('Connection attempted')):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main(['--case-label', 'CHECK', '--output', str(OUTPUT / 'blocked')])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((OUTPUT / 'blocked').exists())

    def test_unknown_enter_is_not_replaced_by_exit(self):
        class UnknownClient:
            state = ClientState(session='active')
            pending_request = {'path': '/enter', 'payload': {'request_id': 'unchanged'}}
            def enter(self):
                raise TimeoutError('Outcome unknown')
            def exit(self):
                raise AssertionError('Must not replace unknown action')
        client = UnknownClient()
        report = {}
        with self.assertRaises(TimeoutError):
            runner.run_session(client, report, [], {})
        self.assertEqual(report['pending_request'], client.pending_request)
        self.assertNotIn('error_exit_response', report)
        self.assertNotIn('error_exit_exception', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BridgeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (OUTPUT / 'test_summary.json').write_text(json.dumps(dict(tests=result.testsRun, errors=len(result.errors),
        failures=len(result.failures), successful=result.wasSuccessful()), indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)

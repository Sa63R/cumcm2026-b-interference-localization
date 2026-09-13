"""Exercise the real entry point against an isolated local fixture, never port 2026."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from test_bridge import Application, BridgeTests, Scenario, Server, trace_hash
import run_official as runner


class EntryModeTests(unittest.TestCase):
    def test_both_modes_and_legacy_aliases_have_identical_complete_actions(self):
        case = json.loads((runner.ROOT / 'checks/cases.json').read_text())[0]
        results = []
        real_client = runner.SimulatorClient
        with tempfile.TemporaryDirectory() as temp:
            for index, (flags, mode) in enumerate([
                (['--mode', 'practice'], 'practice'),
                (['--mode', 'formal'], 'formal'),
                (['--practice-confirmed'], 'practice'),
                (['--formal-confirmed'], 'formal'),
            ]):
                with self.subTest(flags=flags):
                    app = Application(Scenario.from_dict(case['scenario']))
                    server = Server(app, 0)
                    self.assertNotEqual(server.server_port, 2026)
                    worker = threading.Thread(target=server.serve_forever, daemon=True)
                    worker.start()
                    output = Path(temp) / str(index)
                    def isolated_client(*args, **kwargs):
                        return real_client(*args, base_url=f'http://127.0.0.1:{server.server_port}', **kwargs)
                    try:
                        with patch.object(runner, 'SimulatorClient', side_effect=isolated_client):
                            with contextlib.redirect_stdout(io.StringIO()):
                                runner.main(flags + ['--robot-id', 'local-test', '--case-label',
                                                     'OFFLINE-FIXTURE', '--output', str(output)])
                        report = json.loads((output / 'result.json').read_text())
                        self.assertEqual(report['mode'], f'operator_confirmed_q4_{mode}')
                        self.assertEqual(report['status'], 'policy_completed_and_exited')
                        self.assertEqual(len(app.session.engine.cleared), len(app.session.engine.sources))
                        self.assertTrue((output / 'requests.jsonl').stat().st_size)
                        results.append((trace_hash(app.session.history), app.session.engine.time_us))
                    finally:
                        server.shutdown()
                        server.server_close()
                        worker.join()
        self.assertEqual(len(results), 4)
        self.assertEqual(len(set(results)), 1)

    def test_preflight_never_constructs_a_client(self):
        output = io.StringIO()
        with patch.object(runner, 'SimulatorClient', side_effect=AssertionError('Unexpected connection')):
            with contextlib.redirect_stdout(output):
                runner.main(['--check-only'])
        report = json.loads(output.getvalue())
        self.assertEqual(report['status'], 'preflight_passed')
        self.assertIs(report['simulator_contacted'], False)

    def test_invalid_arguments_do_not_connect_or_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            base = ['--robot-id', 'local-test', '--case-label', 'OFFLINE', '--output', str(output)]
            for flags in [[], ['--mode', 'invalid'], ['--mode', 'formal', '--practice-confirmed']]:
                with self.subTest(flags=flags):
                    with patch.object(runner, 'SimulatorClient', side_effect=AssertionError('Unexpected connection')):
                        with contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as error:
                                runner.main(base + flags)
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EntryModeTests)
    suite.addTest(BridgeTests('test_unknown_enter_is_not_replaced_by_exit'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

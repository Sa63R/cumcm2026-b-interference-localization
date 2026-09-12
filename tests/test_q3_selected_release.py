"""Offline packaging and protocol checks; no official simulator requests."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/q3_selected'
sys.path.insert(0, str(PACKAGE))
import q3_runtime as runtime
from q3_optical import ExperimentalOpticalAgent

spec = importlib.util.spec_from_file_location('q3_selected_runner', PACKAGE/'run_practice.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
b, np = runtime.b, runtime.np


class AnnexSimulator(b.ToySimulator):
    """Local fixture with rounded bearings and clear-without-retuning."""
    def _error(self, channel):
        return super()._error(channel) * math.radians(1) / b.A

    def detect(self, channel):
        answer = super().detect(channel)
        if answer.kind == 'bearing':
            answer = b.Observation('bearing', math.radians(round(math.degrees(answer.bearing), 2) % 360))
        return answer

    def clear(self, channel):
        self.optical += 1
        success = channel in self._live and b.norm(self._sources[channel].xy-self.pos) <= 20
        if success:
            self._live.remove(channel)
            self.clears += 1
        else:
            self.failed += 1
        self._log('clear', channel=channel, success=success)
        return success


class SyntheticClient(runtime.SimulatorClient):
    def __init__(self, simulator, path):
        self.simulator = simulator
        super().__init__('synthetic-only', log_path=path)

    def _exchange(self, action, timeout):
        extra = {}
        if action.path == '/enter':
            extra = dict(max_virtual_duration_s=360000, max_real_duration_s=1200, remaining_real_duration_s=1200)
        elif action.path == '/exit':
            extra = dict(exit_reason='user_exit')
        else:
            data = action.payload
            self.simulator.move(np.array([data['position']['x'], data['position']['y']]))
            if action.path == '/measure':
                answer = self.simulator.detect(data['channel'])
                extra = dict(measure_result={'none':'no_signal', 'strong':'near', 'bearing':'direction'}[answer.kind])
                if answer.kind == 'bearing':
                    extra['svd_deg'] = round(math.degrees(answer.bearing), 2) % 360
            else:
                extra = dict(clear_result='success' if self.simulator.clear(data['channel']) else 'no_target_in_range')
        return 200, dict(accepted=True, virtual_time_s=self.simulator.virtual_time,
                         real_timestamp_ms=time.time_ns()//1_000_000, **extra)


class SelectedReleaseTests(unittest.TestCase):
    def setUp(self):
        b.A = math.radians(1.005)

    def test_preserved_sources_and_optical_algorithm_identity(self):
        manifest = json.loads((PACKAGE/'SOURCE_MANIFEST.json').read_text())
        for relative, record in manifest['unchanged_files'].items():
            self.assertEqual(hashlib.sha256((PACKAGE/relative).read_bytes()).hexdigest(), record['sha256'])
        def definitions(path):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and n.name in ('area_points', 'ExperimentalOpticalAgent')}
        self.assertEqual(definitions(PACKAGE/'q3_optical.py'),
                         definitions(PACKAGE/'vendor/optical_experiment_original.py'))

    def test_complete_sessions_match_direct_local_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            for seed, boundary in [(5022, False), (5027, False), (93005, True)]:
                for method in runtime.METHODS:
                    with self.subTest(seed=seed, method=method):
                        sources = b.make_case(seed, boundary)
                        direct = AnnexSimulator(sources, seed)
                        agent = runtime.make_agent(direct, method)
                        agent.run()
                        simulator = AnnexSimulator(sources, seed)
                        with SyntheticClient(simulator, Path(directory)/f'{seed}-{method}.jsonl') as client:
                            client.enter()
                            result = runtime.run_policy(client, method)
                            client.exit()
                            self.assertFalse(simulator._live)
                            self.assertFalse(direct._live)
                            self.assertEqual(client.state.cleared_count, len(sources))
                            self.assertEqual(result['cleared'], sorted(agent.cleared))
                            self.assertEqual(result['absent'], sorted(agent.absent))
                            self.assertTrue(result['complete_channel_certificate'])
                            self.assertAlmostEqual(client.state.virtual_time_s, direct.virtual_time, places=6)
                            self.assertAlmostEqual(client.state.time_breakdown.total_s, simulator.virtual_time, places=6)
                            self.assertEqual(simulator.failed, direct.failed)
                            self.assertEqual(simulator.detects, direct.detects)
                            self.assertIsNone(client.pending_request)

    def test_clear_keeps_radio_and_failed_attempt_costs_three_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = AnnexSimulator([b.Source(1, np.array([300.,400.]),1200.)],1)
            with SyntheticClient(sim,Path(directory)/'charge.jsonl') as client:
                client.enter()
                device=runtime.OfficialDevice(client)
                device.move([300.,400.])
                self.assertEqual(device.detect(3).kind,'none')
                self.assertAlmostEqual(client.state.virtual_time_s,106.)
                self.assertTrue(device.clear(1))
                self.assertEqual(device.channel,3)
                self.assertFalse(device.clear(1))
                self.assertAlmostEqual(client.state.virtual_time_s,114.)
                client.exit()

    def test_optical_failure_preserves_region_and_obeys_attempt_limits(self):
        sim = AnnexSimulator(b.make_case(5022),5022)
        agent = runtime.make_agent(sim, 'optical')
        # Use a known geometrical region and guaranteed misses, independent of
        # the source distribution, to check the actual failure-control path.
        from types import SimpleNamespace
        region=np.array([[-10.,-10.],[10.,-10.],[10.,10.],[-10.,10.]])
        agent.tracks[1]=SimpleNamespace(poly=region.copy(),radius=15.)
        with mock.patch.object(sim,'clear',return_value=False) as clear:
            self.assertFalse(agent._attempt(1))
            self.assertEqual(agent.trials[1],1)
            self.assertFalse(agent._attempt(1))
            self.assertEqual(clear.call_count,1)
            sim.move(np.array([1.,0.]))
            self.assertFalse(agent._attempt(1))
            sim.move(np.array([2.,0.]))
            self.assertFalse(agent._attempt(1))
            self.assertEqual(clear.call_count,2)
            self.assertEqual(agent.trials[1],2)
            self.assertNotIn(1,agent.cleared)
            np.testing.assert_array_equal(agent.tracks[1].poly,region)

    def test_origin_scan_and_optical_parameters(self):
        sources=[b.Source(1,np.array([1700.,0.]),1000.)]
        sim=AnnexSimulator(sources,0)
        origin=runtime.make_agent(sim,'v3_origin20')
        origin.scan_stop()
        self.assertEqual(sim.detects,20)
        optical=runtime.make_agent(AnnexSimulator(sources,0),'optical')
        self.assertEqual((optical.threshold,optical.max_radius,optical.cap),(.4,80.,2))
        self.assertEqual(optical.initial_channels,0)

    def test_missing_practice_confirmation_never_constructs_client(self):
        with mock.patch.object(runner,'SimulatorClient') as factory:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main(['--method','optical','--robot-id','synthetic-only'])
            self.assertEqual(error.exception.code,2)
            factory.assert_not_called()

    def test_runner_writes_complete_session_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            sim=AnnexSimulator(b.make_case(5022),5022)
            def client_factory(robot_id, log_path):
                self.assertEqual(robot_id,'synthetic-only')
                return SyntheticClient(sim,log_path)
            with mock.patch.object(runner,'SimulatorClient',side_effect=client_factory):
                with contextlib.redirect_stdout(io.StringIO()):
                    folder=runner.main(['--method','optical','--robot-id','synthetic-only',
                                        '--practice-confirmed','--output',directory])
            result=json.loads((folder/'result.json').read_text())
            self.assertEqual(result['status'],'policy_completed_and_exited')
            self.assertEqual(result['version'],runtime.VERSION)
            self.assertEqual(result['parameters']['cap'],2)
            self.assertIsNone(result['pending_request'])
            self.assertIn('q3_optical.py',result['source_sha256'])
            self.assertTrue((folder/'requests.jsonl').is_file())
            self.assertFalse(sim._live)


if __name__ == '__main__':
    unittest.main(verbosity=2)

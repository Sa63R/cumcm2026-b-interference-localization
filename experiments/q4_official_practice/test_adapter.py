"""Synthetic protocol/geometry checks. These are NOT official practice runs."""
from __future__ import annotations
import math
from pathlib import Path
import tempfile
import time
import unittest
import sys
from runtime import b, CODE, OfficialDevice, SimulatorClient, configure_geometry, prepare_policy, run_policy
sys.path.insert(0, str(CODE))
from tests.fake_simulator import FakeSimulator


class RoundedLocalSimulator(b.LocalSimulator):
    def _error(self, channel):
        # Keep physical error at +/-1 deg while the strategy uses +/-1.005.
        return super()._error(channel) * math.radians(1.) / b.DELTA


class SyntheticClient(SimulatorClient):
    """Feeds public-shaped responses through the real client's validation.

    Hidden state is test-fixture-only. No network request is made here.
    """
    def __init__(self, sim, path):
        self._sim = sim
        super().__init__('synthetic-only', log_path=path)

    def _exchange(self, action, timeout):
        extra = {}
        if action.path == '/enter':
            extra = dict(max_virtual_duration_s=360000, max_real_duration_s=1200,
                         remaining_real_duration_s=1200)
        elif action.path == '/exit':
            extra = dict(exit_reason='user_exit')
        else:
            fields = action.payload
            self._sim.move((fields['position']['x'], fields['position']['y']))
            if action.path == '/measure':
                obs = self._sim.detect(fields['channel'])
                extra = dict(measure_result={'bearing':'direction', 'none':'no_signal', 'strong':'near'}[obs.status])
                if obs.status == 'bearing':
                    extra['svd_deg'] = round(math.degrees(obs.theta), 2) % 360
            else:
                extra = dict(clear_result='success' if self._sim.clear(fields['channel']) else 'no_target_in_range')
        return 200, dict(accepted=True, virtual_time_s=self._sim.virtual_seconds,
                         real_timestamp_ms=time.time_ns() // 1_000_000, **extra)


class AdapterTests(unittest.TestCase):
    def test_transport_move_channel_and_idempotent_clear(self):
        with tempfile.TemporaryDirectory() as directory, FakeSimulator() as fixture:
            with SimulatorClient('test-team', base_url=fixture.base_url, log_path=Path(directory)/'http.jsonl') as client:
                client.enter()
                device = OfficialDevice(client)
                device.move((300., 400.))
                self.assertEqual(client.state.position.x, 0.)
                self.assertEqual(device.detect(3).status, 'none')
                self.assertEqual(client.state.virtual_time_s, 106.)
                fixture.drop_once_paths.add('/clear')
                self.assertTrue(device.clear(1))
                self.assertEqual(device.channel, 3)
                self.assertEqual(client.state.virtual_time_s, 111.)
                self.assertEqual(fixture.state['clear_count'], 1)
                clears = [p for method, path, p in fixture.requests if path == '/clear']
                self.assertEqual(len(clears), 2)
                self.assertEqual(clears[0], clears[1])
                self.assertFalse(device.clear(1))
                self.assertEqual(client.state.cleared_count, 1)
                self.assertEqual(client.state.virtual_time_s, 114.)
                client.exit()

    def test_rounding_extreme_is_inside_conservative_polygon(self):
        configure_geometry()
        self.assertLess(math.sqrt(21/16 - b.COS_D + math.sin(b.DELTA)/2), b.KAPPA)
        for angle in (.0049, .9951, 89.9951, 179.0049, 359.0049, 359.9951):
            true = math.radians(angle)
            source = b.mul(1200., b.unit(true))
            for error in (-1., 1.):
                reported = math.radians(round((angle+error) % 360, 2) % 360)
                axis = b.unit(reported)
                along = b.dot(source, axis)
                lateral = b.dot(source, (-axis[1], axis[0]))
                self.assertLessEqual(abs(lateral), along*b.TAN_D + 1e-7)

    def test_v4_full_tasks_with_rounded_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in ('v4',):
                for index, (fraction, place, error) in enumerate([
                    (0., 'uniform', 'hash'), (.5, 'uniform', 'hash'),
                    (1., 'uniform', 'plus'), (1., 'outward_boundary', 'minus')]):
                    with self.subTest(method=method, error=error, fraction=fraction):
                        state, planner, _ = prepare_policy(method)
                        sim = RoundedLocalSimulator(b.make_case(109500000+index, fraction, place), 109500000+index, error)
                        with SyntheticClient(sim, Path(directory)/f'{method}_{index}.jsonl') as client:
                            client.enter()
                            report = run_policy(client, state, planner)
                            self.assertEqual(sim.clear_count, len(sim._targets))
                            self.assertEqual(client.state.cleared_count, sim.clear_count)
                            self.assertAlmostEqual(client.state.virtual_time_s, sim.virtual_seconds)
                            self.assertIn(report['stop_certificate'], ('source_upper_bound', 'coverage_complete'))
                            client.exit()


if __name__ == '__main__':
    unittest.main(verbosity=2)

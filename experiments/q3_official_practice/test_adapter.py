"""Check official response mapping and exact parity with paired local results."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from runtime import CODE, COMPARISON, METHODS, SimulatorClient, OfficialDevice, b, run_policy
sys.path.insert(0, str(COMPARISON))
import run_comparison as local

class SyntheticClient(SimulatorClient):
    def __init__(self, sim, path):
        self.sim = sim
        super().__init__('synthetic-only', log_path=path)
    def _exchange(self, action, timeout):
        extra = {}
        if action.path == '/enter':
            extra = dict(max_virtual_duration_s=360000, max_real_duration_s=1200, remaining_real_duration_s=1200)
        elif action.path == '/exit':
            extra = dict(exit_reason='user_exit')
        else:
            d = action.payload
            self.sim.move(local.np.array([d['position']['x'], d['position']['y']]))
            if action.path == '/measure':
                obs = self.sim.detect(d['channel'])
                extra = dict(measure_result={'none':'no_signal', 'strong':'near', 'bearing':'direction'}[obs.kind])
                if obs.kind == 'bearing':
                    extra['svd_deg'] = round(local.math.degrees(obs.bearing), 2) % 360
            else:
                extra = dict(clear_result='success' if self.sim.clear(d['channel']) else 'no_target_in_range')
        return 200, dict(accepted=True, virtual_time_s=self.sim.virtual_time,
                         real_timestamp_ms=time.time_ns()//1_000_000, **extra)

class Tests(unittest.TestCase):
    def test_nine_policies_match_local_execution(self):
        local.configure('annex')
        with tempfile.TemporaryDirectory() as d:
            for mode in METHODS:
                with self.subTest(mode=mode):
                    seed=5022
                    sim=local.AnnexSimulator(b.make_case(seed), seed)
                    with SyntheticClient(sim, Path(d)/f'{mode}.jsonl') as c:
                        c.enter()
                        result=run_policy(c, mode)
                        self.assertFalse(sim._live)
                        self.assertEqual(result['cleared'], sorted(s.channel for s in b.make_case(seed)))
                        expected=local.work(('annex','random',seed,mode,'hash',0,''))
                        self.assertTrue(expected['success'])
                        self.assertAlmostEqual(c.state.virtual_time_s,expected['virtual_seconds'],places=6)
                        self.assertAlmostEqual(c.state.time_breakdown.total_s,sim.virtual_time,places=6)
                        c.exit()
    def test_clear_keeps_radio_and_failed_attempt_costs_three_seconds(self):
        local.configure('annex')
        with tempfile.TemporaryDirectory() as d:
            sim=local.AnnexSimulator([b.Source(1, local.np.array([300.,400.]),1200.)], 1)
            with SyntheticClient(sim, Path(d)/'test.jsonl') as c:
                c.enter(); device=OfficialDevice(c)
                device.move([300.,400.])
                self.assertEqual(c.state.position.x, 0.)
                self.assertEqual(device.detect(3).kind,'none')
                self.assertAlmostEqual(c.state.virtual_time_s,106.)
                self.assertTrue(device.clear(1))
                self.assertEqual(device.channel,3)
                self.assertFalse(device.clear(1))
                self.assertAlmostEqual(c.state.virtual_time_s,114.)
                c.exit()

if __name__=='__main__': unittest.main(verbosity=2)

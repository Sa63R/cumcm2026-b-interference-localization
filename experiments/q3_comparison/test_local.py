import math
import unittest
import numpy as np
import run_comparison as r
from test_optimized import inside


class AnnexTests(unittest.TestCase):
    def setUp(self):
        r.configure('annex')

    def test_annex_example_time_and_unchanged_clear_channel(self):
        sim = r.AnnexSimulator([], 0)
        sim.move(np.array([300., 400.])); sim.detect(1)
        self.assertAlmostEqual(sim.virtual_time, 105)
        sim.detect(2)
        self.assertAlmostEqual(sim.virtual_time, 111)
        sim.move(np.array([300., 0.])); self.assertFalse(sim.clear(3))
        self.assertEqual(sim.channel, 2)
        self.assertAlmostEqual(sim.virtual_time, 194)
        sim.detect(2)
        self.assertAlmostEqual(sim.virtual_time, 199)
        self.assertEqual(sim.switches, 1)

    def test_success_cost_channel_and_twenty_metre_boundary(self):
        s = r.b.Source(7, np.array([20., 0.]), 1000.)
        sim = r.AnnexSimulator([s], 0)
        self.assertTrue(sim.clear(7))
        self.assertEqual(sim.channel, 1)
        self.assertEqual(sim.virtual_time, 5)
        self.assertFalse(sim.clear(7))
        self.assertEqual(sim.virtual_time, 8)

    def test_error_fixed_and_bounded_after_rounding(self):
        xy = 1200 * np.array([math.cos(.234567), math.sin(.234567)])
        for noise in ['hash', 'plus', 'minus', 'alternating', 'smooth']:
            sim = r.AnnexSimulator([r.b.Source(1, xy, 1500.)], 7, noise)
            first, second = sim.detect(1), sim.detect(1)
            self.assertEqual(first.bearing, second.bearing)
            error = (first.bearing - .234567 + math.pi) % (2 * math.pi) - math.pi
            self.assertLessEqual(abs(error), 1.005 * r.DEG + 1e-12)
            degrees = math.degrees(first.bearing)
            self.assertAlmostEqual(degrees, round(degrees, 2))

    def test_all_policies_truth_containment_audit(self):
        # Test harness reads truth after each completed observation; the actual
        # policy receives only PublicBackend and never receives audit values.
        for mode in r.MODES:
            for seed, stress, noise in [(5000, False, 'hash'), (5011, False, 'plus'),
                                        (91000, True, 'minus'), (91001, True, 'smooth')]:
                with self.subTest(mode=mode, seed=seed, noise=noise):
                    sim = r.AnnexSimulator(r.b.make_case(seed, stress), seed, noise)
                    agent = r.make_agent(sim, mode)
                    original_observe = agent.observe
                    def audited_observe(*args, **kw):
                        original_observe(*args, **kw)
                        for c, track in agent.tracks.items():
                            self.assertTrue(inside(track.poly, sim._sources[c].xy))
                            self.assertLessEqual(r.b.norm(sim._sources[c].xy - track.center), track.radius + 1e-4)
                    agent.observe = audited_observe
                    agent.run()
                    self.assertFalse(sim._live)
                    self.assertFalse(agent.unseen)
                    if mode != 'optical':
                        self.assertEqual(sim.failed, 0)

    def test_unexecuted_scan_does_not_certify_absence(self):
        sim = r.AnnexSimulator(r.b.make_case(4), 4)
        agent = r.make_agent(sim, 'v3')
        agent.scan_stop()
        self.assertEqual(sim.detects, 0)
        self.assertFalse(agent.full_scans)
        self.assertEqual(len(agent.unseen), 20)

    def test_disabled_new_features_reproduce_v3(self):
        from new_methods import PlanningAgent
        for seed in [0, 7, 5011, 91000]:
            sims = [r.AnnexSimulator(r.b.make_case(seed), seed) for _ in range(2)]
            a = r.v.make_agent(sims[0])
            z = PlanningAgent(r.v.PublicBackend(sims[1]))
            a.run(); z.run()
            self.assertAlmostEqual(sims[0].virtual_time, sims[1].virtual_time, places=7)
            self.assertEqual(sims[0].detects, sims[1].detects)

    def test_hypothetical_scenes_do_not_change_evidence(self):
        from new_methods import PlanningAgent
        sim = r.AnnexSimulator(r.b.make_case(0), 0)
        a = PlanningAgent(r.v.PublicBackend(sim), scenario=True, future_cover=True)
        a.scan_stop()
        sim.move(a.cover[0]); a.scan_stop(0)
        before = (sim.virtual_time, set(a.unseen), set(a.absent), len(a.full_scans), list(a.remaining_cover))
        nodes = a.task_nodes()
        a.sample_worlds()
        if nodes:
            a.choose_action(nodes, a.order(a.env.pos, [n[2] for n in nodes]))
        after = (sim.virtual_time, set(a.unseen), set(a.absent), len(a.full_scans), list(a.remaining_cover))
        self.assertEqual(before, after)


if __name__ == '__main__':
    r.v.warmup()
    unittest.main(verbosity=2)

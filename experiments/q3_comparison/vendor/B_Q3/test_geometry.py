"""Run with: python -m unittest -v test_geometry.py"""
import math
import unittest
import numpy as np
from q3_solver import A, COVER, Agent, Source, ToySimulator, first_polygon, mec, wedge


class GeometryTests(unittest.TestCase):
    def test_seven_point_coverage(self):
        angles = np.linspace(0, 2*math.pi, 10001)
        for radius in [0, 100, 500, 999, 1200, 1500, 1800]:
            points = radius*np.column_stack([np.cos(angles), np.sin(angles)])
            d = np.linalg.norm(points[:, None, :]-np.array(COVER)[None, :, :], axis=2)
            self.assertLessEqual(float(d.min(axis=1).max()), 999+1e-7)

    def test_first_triangle_radius(self):
        _, r = mec(first_polygon(np.array([0., 0.]), 0.))
        self.assertAlmostEqual(r, 1500/(2*math.cos(A)**2), places=5)

    def test_diameter_40_is_not_safe(self):
        p = np.array([[0., 0.], [40., 0.], [20., 20*math.sqrt(3)]])
        _, r = mec(p)
        self.assertAlmostEqual(r, 40/math.sqrt(3), places=5)
        self.assertGreater(r, 20)

    def test_sector_contraction(self):
        rng = np.random.default_rng(421)
        for _ in range(100):
            g = rng.uniform(-1, 1, 2)*900
            p = np.array([0., 0.])
            theta = math.atan2(g[1], g[0])+rng.uniform(-A, A)
            poly = first_polygon(p, theta)
            c, r = mec(poly)
            for _ in range(6):
                if np.linalg.norm(g-c) <= 5:
                    break
                theta = math.atan2(g[1]-c[1], g[0]-c[0])+rng.uniform(-A, A)
                poly = wedge(poly, c, theta)
                nc, nr = mec(poly)
                self.assertLessEqual(nr, r/(2*math.cos(A))+1e-4)
                self.assertLessEqual(np.linalg.norm(g-nc), nr+1e-5)
                c, r = nc, nr

    def test_spatially_fixed_error(self):
        env = ToySimulator([Source(1, np.array([100., 100.]), 1000)], seed=10)
        one, two = env.detect(1), env.detect(1)
        self.assertEqual(one.kind, 'bearing')
        self.assertEqual(one.bearing, two.bearing)

    def test_origin_strong_signal_and_channel_certificate(self):
        sources = [Source(c, np.zeros(2), 1000.) for c in range(1, 11)]
        env = ToySimulator(sources, seed=11)
        agent = Agent(env)
        agent.run()
        self.assertEqual(env.clears, 10)
        self.assertEqual(agent.absent, set(range(11, 21)))
        self.assertEqual(env.failed, 0)

    def test_sixteen_source_early_certificate(self):
        sources = [Source(c, np.zeros(2), 1000.) for c in range(1, 17)]
        env = ToySimulator(sources, seed=12)
        agent = Agent(env)
        agent.run()
        self.assertEqual(env.clears, 16)
        self.assertEqual(len(agent.remaining_cover), 7)
        self.assertEqual(env.failed, 0)


if __name__ == '__main__':
    unittest.main()

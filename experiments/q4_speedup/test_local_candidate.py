"""Geometry and feedback tests for the new local cover optimization."""
import math
import random
import unittest
from local_candidate import optical_cells, cell_clear, b
from q4_v4_local import LocalConfig
from q4_v4_solver import Belief


class CellGeometryTests(unittest.TestCase):
    def test_cells_partition_rotated_thin_polygons(self):
        # Includes narrow wedges and slanted parallelograms at many rotations.
        for angle in [0.,.17,1.3,2.8]:
            e=b.unit(angle);v=(-e[1],e[0])
            for length,width in [(25,2),(120,8),(350,20)]:
                poly=[b.add(b.mul(x,e),b.mul(y,v)) for x,y in [(0,-width/2),(length,-width/4),(length,width/2),(0,width/4)]]
                cells=optical_cells(poly)
                self.assertTrue(cells)
                def area(p):
                    return abs(sum(a[0]*z[1]-a[1]*z[0] for a,z in zip(p,p[1:]+p[:1])))/2
                # clip_halfplane expands boundaries by its 1e-8 tolerance,
                # causing harmless micrometre-scale overlap, never a gap.
                self.assertAlmostEqual(area(poly),sum(map(area,cells)),delta=1e-4)
                from q4_fast_local import nearest_clear_point
                for cell in cells:
                    for here in [(-500,120),(50,50),(300,-600)]:
                        point=nearest_clear_point(cell,here)
                        self.assertTrue(all(b.dist(point,p)<=19.5+1e-6 for p in cell))

    def test_every_strip_can_be_cleared_and_failure_never_completes(self):
        poly=[(0,-3),(300,-3),(300,3),(0,3)]
        cfg=LocalConfig(optical_cover_limit=20)
        for x in range(0,301,5):
            for mode in ('near_cells','nearest_cells'):
                sim=b.LocalSimulator([b.Target(1,(x,0),1000,None)],543)
                belief=Belief(1,(-100,0),b.Observation('bearing',0.),poly[:])
                report=cell_clear(sim,belief,cfg,mode)
                self.assertEqual(sim.clear_count,1)
                self.assertEqual(report['optical_attempts'],sim.optical_count)
        class AlwaysMiss:
            position=(-100.,0.)
            def move(self,p): self.position=p
            def clear(self,c): return False
        with self.assertRaisesRegex(RuntimeError,'without clear success'):
            cell_clear(AlwaysMiss(),Belief(1,(-100,0),b.Observation('bearing',0.),poly[:]),cfg,'nearest_cells')

if __name__=='__main__': unittest.main()

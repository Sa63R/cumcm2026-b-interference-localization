import itertools
import math
import random
import unittest
from radius_direction import (RadiusPrior, marginalize, visibility_probability,
                              count_and_subset_sampler)

class GeometryMarginalTests(unittest.TestCase):
    def test_unobserved_evidence(self):
        m=marginalize((0,0))
        self.assertAlmostEqual(m.mixture_evidence,1)
        self.assertAlmostEqual(m.posterior_directional,.5)
    def test_same_ray_negative_upper_radius(self):
        self.assertEqual(visibility_probability((0,0),[(500,0)],[(1200,0)],(1300,0)),0.)
        self.assertAlmostEqual(visibility_probability((0,0),[(500,0)],[(1200,0)],(1100,0)),.5)
    def test_type_prior_updates_after_detection(self):
        self.assertAlmostEqual(marginalize((0,0),[(500,0)]).posterior_directional,1/3)
        self.assertAlmostEqual(visibility_probability((0,0),[(500,0)],[],(-500,0)),2/3)
    def test_near_opposite_negative_implies_directional(self):
        m=marginalize((0,0),[(500,0)],[(-500,0)])
        self.assertAlmostEqual(m.posterior_directional,1.)
        self.assertAlmostEqual(visibility_probability((0,0),[(500,0)],[(-500,0)],(0,500)),.5)
    def test_static_repeat(self):
        self.assertAlmostEqual(visibility_probability((0,0),[(500,0)],[],(500,0)),1)
        self.assertAlmostEqual(visibility_probability((0,0),[(500,0)],[(-500,0)],(-500,0)),0)
        self.assertEqual(marginalize((0,0),[(500,0)]),marginalize((0,0),[(500,0),(500,0)]))
    def test_contradiction_not_success(self):
        with self.assertRaises(ValueError):
            visibility_probability((0,0),[(500,0)],[(500,0)],(0,0))
    def test_zero_distance(self):
        self.assertEqual(marginalize((0,0),[],[(0,0)]).mixture_evidence,0)
        self.assertAlmostEqual(marginalize((0,0),[(0,0)]).mixture_evidence,1)
    def test_radius_atoms_closed_receive_boundary(self):
        rp=RadiusPrior(0,((1000,1),))
        self.assertAlmostEqual(marginalize((0,0),[(1000,0)],directional_prior=0,radius_prior=rp).mixture_evidence,1)
        self.assertEqual(marginalize((0,0),[],[(1000,0)],directional_prior=0,radius_prior=rp).mixture_evidence,0)
        rp2=RadiusPrior(0,((1500,1),))
        self.assertAlmostEqual(marginalize((0,0),[(1500,0)],directional_prior=0,radius_prior=rp2).mixture_evidence,1)
    def test_evidence_monotone(self):
        old=marginalize((100,200),[(500,500)]).mixture_evidence
        new=marginalize((100,200),[(500,500)],[(1000,1000)]).mixture_evidence
        self.assertLessEqual(new,old)
    def test_rotation_invariance(self):
        g=(100.,50.);p=[(500.,100.),(200.,500.)];n=[(-100.,-600.),(1300.,800.)]
        base=marginalize(g,p,n).mixture_evidence
        for a in [.1,.7,1.5,3.14159,5.9]:
            def rot(q):return (q[0]*math.cos(a)-q[1]*math.sin(a),q[0]*math.sin(a)+q[1]*math.cos(a))
            self.assertAlmostEqual(marginalize(rot(g),list(map(rot,p)),list(map(rot,n))).mixture_evidence,base,places=10)

class CardinalityTests(unittest.TestCase):
    def test_prior_recovered_without_observations(self):
        p,_=count_and_subset_sampler([1.]*20,0)
        for n in range(10,17):self.assertAlmostEqual(p[n],1/7)
    def test_known_sixteen_implies_no_more(self):
        p,s= count_and_subset_sampler([.5]*4,16)
        self.assertEqual(p,{16:1.})
        self.assertEqual(s(random.Random(1)),(16,()))
    def test_exhaustive_small(self):
        values=[.1,.3,.7,1.]
        p,s=count_and_subset_sampler(values,2,total_channels=6,min_count=3,max_count=5)
        exact={n:0. for n in range(3,6)}
        for bits in itertools.product([0,1],repeat=4):
            n=2+sum(bits)
            if n in exact:
                exact[n]+=math.prod(v for v,on in zip(values,bits) if on)/math.comb(6,n)/3
        z=sum(exact.values())
        for n in exact:self.assertAlmostEqual(p[n],exact[n]/z)
        rng=random.Random(12)
        for _ in range(1000):
            n,ix=s(rng)
            self.assertEqual(len(ix),n-2)
            self.assertEqual(len(ix),len(set(ix)))
    def test_no_mass_fails_closed(self):
        with self.assertRaises(ValueError):count_and_subset_sampler([0.]*20,0)
    def test_extreme_small_likelihoods(self):
        p,s=count_and_subset_sampler([1e-100]*20,0)
        self.assertAlmostEqual(p[10],1.)
        self.assertEqual(s(random.Random(1))[0],10)
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):RadiusPrior(.5)
        with self.assertRaises(ValueError):marginalize((0,0),directional_prior=1.1)

if __name__=='__main__':unittest.main(verbosity=2)

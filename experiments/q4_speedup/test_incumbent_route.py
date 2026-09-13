"""Legality, projection and MRO tests for retained routes."""
import copy
import random
from types import SimpleNamespace
import unittest
from incumbent_route import (project_and_insert,route_length,IncumbentMixin,
    build_incumbent,FlexibleState,LocalState)


class DummyBase:
    def __init__(self):
        self.visited=[0];self.remaining={1,2,3};self.pending={};self.actions=1
        self.sites=[(0.,0.),(1.,0.),(2.,0.),(10.,0.)]
        self.fresh=[('site',1),('site',2),('site',3)]
    def ordered_actions(self,device):return self.fresh[:]
    def report(self):return {}


class DummyState(IncumbentMixin,DummyBase):
    pass


class IncumbentTests(unittest.TestCase):
    def test_project_insert_has_exact_action_set_and_minimum_increment(self):
        rng=random.Random(773921)
        for _ in range(200):
            keys=[('site',i) for i in range(7)]+[('target',i) for i in range(1,9)]
            points={a:(rng.uniform(-30,30),rng.uniform(-30,30)) for a in keys}
            previous=rng.sample(keys,rng.randrange(1,len(keys)))
            survivors=[a for a in previous if rng.random()<.7]
            newly=rng.choice([a for a in keys if a not in previous])
            legal=survivors+[newly];position=(rng.uniform(-5,5),rng.uniform(-5,5))
            order,inserted=project_and_insert(previous,legal,points,position)
            self.assertEqual(set(order),set(legal));self.assertEqual(len(order),len(legal))
            self.assertEqual(inserted,[newly])
            self.assertEqual([a for a in order if a!=newly],survivors)
            brute=[survivors[:i]+[newly]+survivors[i:] for i in range(len(survivors)+1)]
            self.assertAlmostEqual(route_length(position,order,points),min(route_length(position,q,points) for q in brute))

    def test_retain_shorter_route_tie_and_state_unchanged(self):
        state=DummyState();device=SimpleNamespace(position=(0.,0.))
        good=state.ordered_actions(device)
        state.fresh=[('site',3),('site',1),('site',2)]
        before=copy.deepcopy((state.remaining,state.pending,state.sites))
        self.assertEqual(state.ordered_actions(device),good)
        self.assertEqual(before,(state.remaining,state.pending,state.sites))
        state.fresh=good[:]
        self.assertEqual(state.ordered_actions(device),good)
        self.assertEqual(state.incumbent_stats['ties'],1)

    def test_shift_site_and_clear_project_out_new_target_inserted(self):
        state=DummyState();device=SimpleNamespace(position=(0.,0.))
        state.ordered_actions(device)
        # A completed shift keeps its site identity, then removes that site.
        state.sites[1]=(.5,0.);state.remaining.remove(1)
        target=SimpleNamespace(center=lambda:(4.,0.));state.pending[11]=target
        state.fresh=[('site',2),('target',11),('site',3)]
        order=state.ordered_actions(device)
        self.assertEqual(set(order),set(state.fresh));self.assertNotIn(('site',1),order)
        del state.pending[11];state.fresh=[('site',2),('site',3)]
        order=state.ordered_actions(device)
        self.assertEqual(order,state.fresh)
        self.assertEqual(state.incumbent_stats['inserted_actions'],1)

    def test_mro_applies_route_before_flexible_geometry(self):
        for name in ('flex_incumbent','cells_flex_incumbent'):
            state=build_incumbent(name);mro=type(state).mro()
            self.assertLess(mro.index(FlexibleState),mro.index(IncumbentMixin))
            if name=='cells_flex_incumbent':
                self.assertLess(mro.index(IncumbentMixin),mro.index(LocalState))

if __name__=='__main__':unittest.main()

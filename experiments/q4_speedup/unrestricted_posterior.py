"""Ranking-only prior that permits zero sources of either emission type."""
import bootstrap
import math
from posterior import Worlds as BaseWorlds, ConditionalSimulator as BaseSimulator
from bootstrap import b

class Worlds(BaseWorlds):
    def sample(self,rng):
        n,ix=self.select(rng)
        channels=sorted(set(self.state.pending)|self.state.cleared)+[self.unknown[i] for i in ix]
        targets=[self.pools[c].target(c,self.state.history.get(c,[]),rng,c in self.state.cleared) for c in channels]
        assert len(targets)==n
        return targets,rng.randrange(1<<60)

class ConditionalSimulator(BaseSimulator):
    def _error(self,c):
        if (c,tuple(self.position)) in self.old_bearings:
            return super()._error(c)
        return super()._error(c)*math.radians(1.)/b.DELTA
    def detect(self,c):
        obs=super().detect(c)
        if obs.status=='bearing':
            return b.Observation('bearing',math.radians(round(math.degrees(obs.theta),2)%360))
        return obs

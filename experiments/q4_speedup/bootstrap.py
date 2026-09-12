from pathlib import Path
import sys, math, random
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'q4_comparison'),str(HERE.parent/'q4_official_practice')]
import common
import q4_baseline as b
from runtime import configure_geometry
configure_geometry()

SCENARIOS = [
    ('all_omni',0.,'uniform','hash'),
    ('mixed25',.25,'uniform','hash'),
    ('mixed50',.5,'uniform','hash'),
    ('mixed75',.75,'uniform','hash'),
    ('all_directional',1.,'uniform','hash'),
    ('boundary_directional',1.,'outward_boundary','hash'),
    ('boundary_mixed',.5,'outward_boundary','hash'),
    ('plus1',.5,'uniform','plus'),
    ('minus1',.5,'uniform','minus'),
    ('clustered',.75,'clustered','hash'),
]

def make_case(seed,fraction,placement):
    ts=b.make_case(seed,fraction,'uniform' if placement=='clustered' else placement)
    rng=random.Random(seed+31000000)
    if fraction == 0.:
        for t in ts:t.direction=None
    elif fraction == 1.:
        for t in ts:
            if t.direction is None:
                t.direction=b.unit(math.atan2(t.position[1],t.position[0]) if placement=='outward_boundary' else rng.random()*2*math.pi)
    if placement=='clustered':
        centers=[b.mul(1350*math.sqrt(rng.random()),b.unit(2*math.pi*rng.random())) for _ in range(3)]
        for i,t in enumerate(ts):
            t.position=b.add(centers[i%3],b.mul(160*math.sqrt(rng.random()),b.unit(2*math.pi*rng.random())))
    return ts

class RoundedSimulator(b.LocalSimulator):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.last_clear=None
    def _error(self,c):
        return super()._error(c)*math.radians(1.)/b.DELTA
    def detect(self,c):
        obs=super().detect(c)
        if obs.status=='bearing':
            obs=b.Observation('bearing',math.radians(round(math.degrees(obs.theta),2)%360))
        return obs
    def clear(self,c):
        ok=super().clear(c)
        if ok:self.last_clear=self.virtual_seconds
        return ok

class PublicDevice:
    def __init__(self,device):self.__device=device
    @property
    def position(self):return self.__device.position
    @property
    def channel(self):return self.__device.channel
    def move(self,p):return self.__device.move(p)
    def detect(self,c):return self.__device.detect(c)
    def clear(self,c):return self.__device.clear(c)

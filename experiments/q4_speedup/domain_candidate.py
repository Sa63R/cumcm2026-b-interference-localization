"""Refine the baseline's 64-tangent source disk enclosure to 128 tangents."""
import bootstrap
import math
from bootstrap import b
from online import State
from q4_v4_solver import Belief

NORMALS=tuple(b.unit(2*math.pi*i/128) for i in range(128))

def clip_domain(poly):
    if not poly or max(map(b.norm,poly)) <= 1800.+1e-8:
        return poly
    # Intersection of tangent halfplanes CONTAINS the closed 1800 m disk.
    out=poly
    for n in NORMALS:
        if max(b.dot(n,v) for v in out)<=1800.+1e-7:
            continue
        out=b.clip_halfplane(out,n,1800.+1e-7)
        if not out:
            raise RuntimeError('Bearing feedback conflicts with the known source domain')
    return out

class DomainBelief(Belief):
    def tighten(self):
        self.poly=clip_domain(self.poly)
        super().tighten()

class DomainMixin:
    def discovery(self,device):
        for c in sorted(self.unknown,key=lambda x:(x!=device.channel,x)):
            if len(self.cleared)+len(self.pending)==16:break
            obs=device.detect(c)
            if obs.status=='none':
                self.negatives[c].append(device.position)
                continue
            self.unknown.remove(c)
            poly=b.initial_polygon(device.position,obs.theta,1500.) if obs.status=='bearing' else []
            tr=DomainBelief(channel=c,anchor=device.position,obs=obs,poly=poly,
                measured=[device.position],positives=[device.position],negatives=self.negatives[c][:],
                negative_enabled=self.config.directional_negatives,direction_sectors=self.config.direction_sectors)
            tr.tighten();self.pending[c]=tr
        self.scanpoints.append(device.position)

class CandidateState(DomainMixin,State):
    pass

"""Local RF/optical candidates. Uses only the public observation interface."""
from __future__ import annotations
import math
import sys
import copy
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'q4_comparison'))
import common
import q4_baseline as b
from online import State, HistoryDevice
from q4_v4_solver import V4Config
from q4_v4_local import localize_step
from q4_fast_local import nearest_clear_point
from q4_v3_local import optical_cover


def optical_cells(poly, radius=19.5):
    """Partition the outer polygon into certified strip cells.

    The same rectangle cover as V4 supplies the strip axis and count. Clipping
    creates convex cells whose union equals the original polygon. Each cell is
    inside its original radius-19.5 disk, so projecting toward the robot is safe.
    """
    centers=optical_cover(poly,radius)
    if not centers:
        return []
    if len(centers)==1:
        return [poly]
    axis=b.sub(centers[-1],centers[0])
    axis=b.mul(1/b.norm(axis),axis)
    cuts=[b.dot(axis,b.mul(.5,b.add(a,z))) for a,z in zip(centers,centers[1:])]
    cells=[]
    for i in range(len(centers)):
        cell=poly[:]
        if i:
            cell=b.clip_halfplane(cell,b.mul(-1,axis),-cuts[i-1])
        if i<len(cuts):
            cell=b.clip_halfplane(cell,axis,cuts[i])
        if cell:
            assert all(b.dist(centers[i],p)<=radius+1e-6 for p in cell)
            cells.append(cell)
    return cells


def polygon_area(poly):
    return abs(sum(a[0]*z[1]-a[1]*z[0] for a,z in zip(poly,poly[1:]+poly[:1])))*.5


def cell_clear(device,belief,config,mode):
    if belief.obs.status=='strong' or b.enclosing_circle(belief.poly)[1]<=b.CLEAR_CERT_RADIUS:
        return None
    centers=optical_cover(belief.poly)
    if not centers or len(centers)>config.optical_cover_limit:
        return None
    cells=optical_cells(belief.poly)
    count=len(cells)
    if mode=='near_cells':
        if b.dist(device.position,nearest_clear_point(cells[-1],device.position)) < b.dist(device.position,nearest_clear_point(cells[0],device.position)):
            cells.reverse()
    attempts=0
    while cells:
        if mode=='greedy_cells':
            costs=[b.dist(device.position,nearest_clear_point(cell,device.position))/5+3 for cell in cells]
            j=max(range(len(cells)),key=lambda i:polygon_area(cells[i])/costs[i])
        elif mode=='nearest_cells':
            j=min(range(len(cells)),key=lambda i:b.dist(device.position,nearest_clear_point(cells[i],device.position)))
        else:
            j=0
        cell=cells.pop(j)
        q=nearest_clear_point(cell,device.position)
        if not all(b.dist(q,p)<=19.5+1e-6 for p in cell):
            raise RuntimeError('Invalid optical cell certificate')
        device.move(q)
        attempts+=1
        if device.clear(belief.channel):
            return dict(stages=getattr(belief,'rf_rounds',0),pair_failures=getattr(belief,'pair_failures',0),certificate='optical_cells_confirmed',optical_attempts=attempts,cover_size=count,negative_updates=belief.negative_updates)
    raise RuntimeError('Certified optical cells exhausted without clear success')


class CandidateState(State):
    """Interface-compatible local parameter candidate; geometry remains outer."""
    def __init__(self, config=None, local_mode='params', local_parameters=None, **kwargs):
        config = copy.deepcopy(config) if config is not None else V4Config()
        if local_parameters:
            for key, value in local_parameters.items():
                if not hasattr(config.local, key):
                    raise ValueError('Unknown local parameter: ' + key)
                setattr(config.local, key, value)
        self.local_mode=local_mode
        super().__init__(config=config, **kwargs)

    def execute(self,raw_device,action):
        if self.local_mode=='params' or action[0]!='target':
            return super().execute(raw_device,action)
        if self.local_mode not in ('near_cells','greedy_cells','nearest_cells'):
            raise ValueError('Unknown local mode: '+self.local_mode)
        device=HistoryDevice(raw_device,self.history)
        self.actions+=1
        if self.actions>len(self.sites)+16*14+1+self.extra_actions:
            raise RuntimeError('Progress invariant violated')
        k=action[1]
        tr=self.pending[k]
        stats=cell_clear(device,tr,self.config.local,self.local_mode)
        if stats is None:
            stats=localize_step(device,tr,self.config.local,on_probe=(lambda:self.shared(device,exclude=k)) if self.config.shared_at_probes else None)
        if stats.get('complete') is False:
            self.replans+=1
            return
        self.localizations.append(dict(channel=k,**stats))
        self.cleared.add(k)
        self.archived[k]=tr
        del self.pending[k]
        if len(self.cleared)<16 and self.config.shared_at_clears:
            self.shared(device)

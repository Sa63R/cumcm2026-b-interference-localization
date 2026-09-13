"""Retain a known feasible route when heuristic recomputation is worse.

Only the ordering of existing actions changes. The previous route is projected
onto the current remaining sites/pending channels, and newly discovered actions
are inserted at their smallest Euclidean distance increment. All source and
coverage safety state remains owned by the original State implementation.
"""
from __future__ import annotations
import math
import bootstrap
from bootstrap import b
from online import State
from coverage_candidate import FlexibleState
from local_candidate import CandidateState as LocalState


def route_length(position,order,points):
    """The same open Euclidean center-route objective as the original router."""
    previous=position
    lengths=[]
    for action in order:
        point=points[action]
        lengths.append(b.dist(previous,point))
        previous=point
    return sum(lengths)


def project_and_insert(previous,legal_actions,points,position):
    """Delete finished actions, retain survivors once, insert new actions.

    New-action and insertion ties follow deterministic input/index order. The
    function never reads truth or changes State. It always returns exactly the
    supplied legal action set once each, including after a completed shift.
    """
    legal=set(legal_actions)
    if len(legal)!=len(legal_actions):
        raise ValueError('Duplicate legal actions')
    seen=set();order=[]
    for action in previous:
        if action in legal and action not in seen:
            order.append(action);seen.add(action)
    inserted=[]
    for action in legal_actions:
        if action in seen:
            continue
        point=points[action]
        increments=[]
        for index in range(len(order)+1):
            before=position if index==0 else points[order[index-1]]
            increment=b.dist(before,point)
            if index<len(order):
                after=points[order[index]]
                increment+=b.dist(point,after)-b.dist(before,after)
            increments.append(increment)
        index=min(range(len(increments)),key=lambda i:(increments[i],i))
        order.insert(index,action);seen.add(action);inserted.append(action)
    if len(order)!=len(legal) or set(order)!=legal:
        raise RuntimeError('Projected incumbent route lost or duplicated an action')
    return order,inserted


class IncumbentMixin:
    def __init__(self,*args,**kwargs):
        self.incumbent_order=None
        self.incumbent_stats=dict(comparisons=0,retained=0,fresh_selected=0,ties=0,
                                  inserted_actions=0,projected_out_actions=0,
                                  total_immediate_route_saved_m=0.,max_immediate_route_saved_m=0.)
        self.incumbent_decisions=[]
        super().__init__(*args,**kwargs)

    def ordered_actions(self,device):
        fresh=super().ordered_actions(device)
        # The origin scan is mandatory before a full pending/site route exists.
        if not self.visited:
            self.incumbent_order=None
            return fresh
        legal=[('site',i) for i in sorted(self.remaining)]+[('target',c) for c in sorted(self.pending)]
        if len(fresh)!=len(legal) or set(fresh)!=set(legal):
            raise RuntimeError('Fresh route is not the complete legal action set')
        points={action:(self.sites[action[1]] if action[0]=='site' else self.pending[action[1]].center()) for action in legal}
        if not all(all(math.isfinite(x) for x in point) for point in points.values()):
            raise RuntimeError('Nonfinite action center')
        if self.incumbent_order is None:
            self.incumbent_order=list(fresh)
            return fresh
        projected,inserted=project_and_insert(self.incumbent_order,legal,points,device.position)
        old_length=route_length(device.position,projected,points)
        new_length=route_length(device.position,fresh,points)
        retain=old_length<=new_length
        chosen=projected if retain else list(fresh)
        self.incumbent_stats['comparisons']+=1
        self.incumbent_stats['retained' if retain else 'fresh_selected']+=1
        self.incumbent_stats['ties']+=int(old_length==new_length)
        self.incumbent_stats['inserted_actions']+=len(inserted)
        self.incumbent_stats['projected_out_actions']+=sum(action not in points for action in self.incumbent_order)
        saved=max(0.,new_length-old_length)
        self.incumbent_stats['total_immediate_route_saved_m']+=saved
        self.incumbent_stats['max_immediate_route_saved_m']=max(self.incumbent_stats['max_immediate_route_saved_m'],saved)
        self.incumbent_decisions.append(dict(action_index=self.actions,retained=retain,
            incumbent_length_m=old_length,fresh_length_m=new_length,
            incumbent_first=projected[0] if projected else None,
            fresh_first=fresh[0] if fresh else None,inserted=inserted))
        # Store plain site tags BEFORE outer FlexibleState optionally changes
        # the returned first action to ('shift', site_id, position).
        self.incumbent_order=list(chosen)
        return list(chosen)

    def report(self):
        return dict(super().report(),incumbent=dict(self.incumbent_stats,decisions=self.incumbent_decisions))


class IncumbentState(IncumbentMixin,State):
    pass


class FlexIncumbentState(FlexibleState,IncumbentMixin,State):
    pass


class CellsFlexIncumbentState(FlexibleState,IncumbentMixin,LocalState):
    pass


def build_incumbent(method):
    if method=='incumbent':
        return IncumbentState()
    if method=='flex_incumbent':
        return FlexIncumbentState()
    if method=='cells_flex_incumbent':
        return CellsFlexIncumbentState(local_mode='nearest_cells',local_parameters={'optical_cover_limit':12})
    raise ValueError(method)

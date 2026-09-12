"""R12 first-probe alternative with independent decision/action ownership."""
import math
import time

from planning.q4_anchor_first_hit import (LIMITS, ModelUnavailable, anchor_candidates,
    radio_prefix, rank_anchor_probes)
from simulator_client.state import Position
from .q4_joint_continuation import Q4JointContinuation


CONFIG='anchor_first_hit'
MAX_NEW_PROBES=40


def xy(p):
    return [p.x,p.y] if p is not None else None


def preview_probe(region, first_bearing, current, observed):
    """Pure equivalent of the mode=state_pruned/R9 geometric five-point rule."""
    region=region.copy()
    circle=region.enclosing_disk(); center=Position.coerce(circle.center)
    angle=math.radians(first_bearing); scale=min(180.,max(25.,circle.radius*.5))
    perpendicular=(-math.sin(angle),math.cos(angle))
    offsets=((0.,0.),(perpendicular[0]*scale,perpendicular[1]*scale),
        (-perpendicular[0]*scale,-perpendicular[1]*scale),
        (math.cos(angle)*scale,math.sin(angle)*scale),
        (-math.cos(angle)*scale,-math.sin(angle)*scale))
    candidates=[Position(center.x+x,center.y+y) for x,y in offsets]
    fresh=[p for p in candidates if (round(p.x,6),round(p.y,6)) not in observed]
    if not fresh:return None
    return candidates[0] if candidates[0] in fresh else min(fresh,key=current.distance_to)


class AnchorFirstHit(Q4JointContinuation):
    def __init__(self,client,max_actions,max_active_probes,*,max_expansions,config=CONFIG):
        if config!=CONFIG:raise ValueError('Unknown anchor-first-hit configuration')
        super().__init__(client,max_actions,max_active_probes,max_expansions=max_expansions)
        self.anchor_first_hit_log=[]
        self.actual_new_probes=0
        self._anchor_pending=None
        self.report.strategy_parameters.update(anchor_first_hit_config=config,
            anchor_first_hit_log=self.anchor_first_hit_log,
            anchor_first_hit_limits=dict(LIMITS,actual_new_probe_limit=MAX_NEW_PROBES),
            anchor_first_hit_scope='Only resolver index0, canonical radius>40 and no ready auxiliary; original preview plus two real-direction-anchor offsets; physical R12 safety and all fallback points retained',
            anchor_first_hit_model='Assumed finite spatial prior and continuous conditional radio model; one radio then expected first optical hit on a branch-common route; no real feedback or geometry is inferred')

    def _parent_probe(self,channel,index,event,status):
        event.update(parent_fallback=True,status=status)
        try:
            chosen=super()._next_probe(channel,index)
            event['selected']=xy(chosen)
            return chosen
        except Exception as error:
            event.update(status='parent_clear' if channel in self.cleared else 'interrupted',
                         interruption_type=type(error).__name__,interruption_reason=str(error))
            raise
        finally:
            event['end_actual_action_count']=len(self.report.action_history)

    def _next_probe(self,channel,index):
        began=time.perf_counter(); context=self._joint_context
        prefix=len(self.report.action_history)
        event=dict(id=len(self.anchor_first_hit_log),resolver_id=context['event']['id'] if context else None,
            channel=channel,index=index,after_actual_action_count=prefix,end_actual_action_count=prefix,
            current_position=xy(self.client.state.position),current_channel=self.client.state.current_channel,
            canonical_radius_m=None,model_region_kind=None,model_vertices=None,
            baseline=None,anchor=None,candidates=[],model=None,selected=None,
            changed=False,parent_fallback=False,status='pending',executed_measure=False,
            r8_clear_before_measure=False,actual_new_probes_before=self.actual_new_probes,
            actual_new_probes_after=self.actual_new_probes,actual_result=None,
            actual_cost_s=0.,decision_wall_s=0.,action_wall_s=0.)
        self.anchor_first_hit_log.append(event)
        try:
            if not context or context['channel']!=channel:
                return self._parent_probe(channel,index,event,'outside_resolver')
            if index!=0:
                return self._parent_probe(channel,index,event,'not_first_probe')
            canonical=self.regions.get(channel)
            if (channel not in self.detected or channel in self.cleared or channel in self.near_points
                    or canonical is None or not canonical.vertices or not canonical.observations):
                return self._parent_probe(channel,index,event,'no_live_positive_region')
            circle=canonical.enclosing_disk();event['canonical_radius_m']=circle.radius
            if not math.isfinite(circle.radius) or circle.radius<=40.:
                return self._parent_probe(channel,index,event,'canonical_not_wide')
            aux=context['region']
            if aux is not None and aux.enclosing_disk().radius<=19.9:
                return self._parent_probe(channel,index,event,'auxiliary_ready')
            if self.actual_new_probes>=MAX_NEW_PROBES:
                return self._parent_probe(channel,index,event,'actual_new_probe_limit')
            region=aux if aux is not None else canonical
            event.update(model_region_kind='auxiliary' if aux is not None else 'canonical',
                         model_vertices=[list(p) for p in region.vertices])
            baseline=preview_probe(region,self.first_bearings[channel],self.client.state.position,
                                   self.observed_positions.get(channel,set()))
            event['baseline']=xy(baseline)
            if baseline is None:
                return self._parent_probe(channel,index,event,'no_parent_probe')
            try:
                candidates,anchor=anchor_candidates(self.report.action_history,channel,
                    self.client.state.position,baseline,self.observed_positions.get(channel,set()))
                event.update(anchor=anchor,candidates=[list(p) for p in candidates])
                selected,model=rank_anchor_probes(region,radio_prefix(self.report.action_history,channel),
                    candidates=candidates,current=self.client.state.position,
                    first_bearing=self.first_bearings[channel],channel=channel,
                    current_channel=self.client.state.current_channel)
                event['model']=model
            except ModelUnavailable as error:
                event['unavailable_reason']=str(error)
                return self._parent_probe(channel,index,event,'model_unavailable')
            if selected==0:
                return self._parent_probe(channel,index,event,'original_best')
            chosen=Position.coerce(candidates[selected])
            event.update(selected=xy(chosen),changed=True,status='planned')
            self._anchor_pending=event
            return chosen
        finally:
            event['decision_wall_s']=time.perf_counter()-began

    def _perform(self,action,position,channel,phase):
        event=self._anchor_pending
        owns=(event is not None and action=='measure' and phase=='active_localization'
              and channel==event['channel'] and xy(Position.coerce(position))==event['selected'])
        if not owns:return super()._perform(action,position,channel,phase)
        began=time.perf_counter(); before=self.client.state.virtual_time_s
        try:
            return super()._perform(action,position,channel,phase)
        except Exception as error:
            event.update(interruption_type=type(error).__name__,interruption_reason=str(error))
            raise
        finally:
            end=len(self.report.action_history)
            actual=self.report.action_history[event['after_actual_action_count']:end]
            measurements=[a for a in actual if a['action']=='measure' and a['channel']==channel
                and a['phase']=='active_localization' and a['position']==event['selected']]
            event['executed_measure']=bool(measurements)
            event['r8_clear_before_measure']=any(a['phase']=='speculative_clear_before_probe' for a in actual)
            if measurements:
                self.actual_new_probes+=1
                event.update(status='measured',actual_result=measurements[-1]['result'])
            else:
                event['status']='cleared_by_parent' if channel in self.cleared else 'interrupted'
            event.update(end_actual_action_count=end,actual_new_probes_after=self.actual_new_probes,
                actual_cost_s=self.client.state.virtual_time_s-before,action_wall_s=time.perf_counter()-began)
            self._anchor_pending=None


def run_q4_anchor_first_hit(client,*,problem=4,config=CONFIG,max_actions=20000,
                            max_active_probes=6,max_expansions=200):
    if type(problem) is not int or problem!=4 or config!=CONFIG:
        raise ValueError('Q4 only; config must be anchor_first_hit')
    for value,low,high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low<=value<=high:
            raise ValueError('Invalid integer execution budget')
    return AnchorFirstHit(client,max_actions,max_active_probes,max_expansions=max_expansions,config=config).run()

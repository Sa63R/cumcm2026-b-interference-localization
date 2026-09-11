"""Bounded directional actions and separate compatible risk witnesses.

Witnesses are deliberately selected stress examples, never posterior samples.
Their costs never enter the nominal mean or confirmation standard error.
"""
from dataclasses import replace
import math
import time

from geometry import distance
from strategies.q3_belief import _history_by_channel, _radius_interval
from strategies.q3_fresh_stepper import Action
from strategies.q3_round3 import Round3Rollout, Proposal, plain, digest


def compatible_witnesses(policy, proposal, nominal_worlds, deadline, count=4):
    if not nominal_worlds or count<=0:
        return [],[]
    events,mandatory=_history_by_channel(policy.history,deadline)
    optional=sorted(set(range(1,21))-mandatory)
    if not optional:
        return [],[]
    p=tuple(policy.pending.position);q=tuple(proposal.actions[0].position)
    positions=[]
    # Public deterministic search candidates, not fitted distributions.
    for radius in (600.,1000.,1400.,1799.):
        for i in range(48):
            theta=2*math.pi*i/48
            positions.append((radius*math.cos(theta),radius*math.sin(theta)))
    for radius in (1000.,1100.,1250.,1499.):
        for i in range(64):
            theta=2*math.pi*i/64
            positions.append((p[0]+radius*math.cos(theta),p[1]+radius*math.sin(theta)))
    world=nominal_worlds[0]
    present={s.channel for s in world.sources}
    channels=[c for c in optional if c in present or len(world.sources)<16]
    feasible=[]
    for channel in channels:
        for index,s in enumerate(positions):
            if index%32==0 and time.perf_counter()>=deadline:
                raise TimeoutError('Risk witness construction deadline')
            interval=_radius_interval(s,events[channel])
            if interval is not None:
                lo,hi=interval
                feasible.append((channel,s,lo,hi))
    if not feasible:
        return [],[]
    modes=['lost_visibility','east','north','west','south']
    witnesses=[];labels=[];seen=set()
    for mode in modes:
        choices=[]
        for channel,s,lo,hi in feasible:
            if mode=='lost_visibility':
                lower=max(lo,distance(s,p));upper=min(hi,distance(s,q))
                if upper-lower<=1e-6:continue
                radius=lower+.02*(upper-lower)
                score=distance(s,q)-distance(s,p)
            else:
                radius=lo+.02*(hi-lo)
                score={'east':s[0],'west':-s[0],'north':s[1],'south':-s[1]}[mode]
            choices.append((score,channel,s,radius))
        if not choices:continue
        _,channel,s,radius=max(choices,key=lambda x:(x[0],-x[1],x[2]))
        key=(channel,s,radius)
        if key in seen:continue
        seen.add(key)
        sources=[]
        from simulation.cases import Source
        for source in world.sources:
            if source.channel==channel:continue
            interval=_radius_interval((source.x,source.y),events[source.channel])
            if interval is None:raise ValueError('Nominal world incompatible with public history')
            lo,hi=interval
            sources.append(replace(source,reception_radius_m=lo+.02*(hi-lo)))
        sources.append(Source(channel,s[0],s[1],radius))
        candidate=replace(world,case_id=f'risk-{world.seed}-{mode}',
            seed=world.seed ^ (10571*(len(witnesses)+1)),sources=tuple(sorted(sources,key=lambda x:x.channel)),
            description='Public-history-compatible risk witness, not a posterior draw')
        witnesses.append(candidate);labels.append(mode)
        if len(witnesses)>=count:break
    return witnesses,labels


class ExplorationRollout(Round3Rollout):
    def __init__(self,*,risk_limit_s=120.0,**kwargs):
        super().__init__(**kwargs)
        self.risk_limit_s=risk_limit_s
        self.used_events=set()

    def event_reason(self,policy):
        if not policy.history or len(policy.history)==self.last_planned_action:
            return None
        latest=policy.history[-1];reason=None
        if latest['action']=='clear' and latest['result']=='success' and not policy.active():
            reason='all_detected_cleared'
        elif latest['result'] in ('direction','near'):
            channel=latest['channel']
            if not any(h['channel']==channel and h['result'] in ('direction','near','success') for h in policy.history[:-1]):
                reason='new_source_discovered'
        if reason is None and distance(policy.position,policy.pending.position)>=350:
            reason='cross_region_departure'
        if (reason is None and latest['phase']=='cover' and
                distance(policy.position,policy.pending.position)>=25):
            reason='leaving_cover_point'
        if reason is None:return None
        key=(len(policy.history),reason,tuple(policy.pending.position),policy.pending.channel)
        if key in self.used_events:return None
        self.used_events.add(key)
        return reason

    def direction_proposals(self,policy):
        # Different directions are exposed only at a real departure boundary.
        departure=distance(policy.position,policy.pending.position)
        if departure<25 or (departure<350 and policy.active()):
            return []
        anchors=policy.needed_anchors()
        if len(anchors)<2 or not policy.unknown():return []
        angle=math.atan2(policy.position[1],policy.position[0])
        tau=2*math.pi
        def order(item,sign):
            delta=(sign*(math.atan2(item[2][1],item[2][0])-angle))%tau
            return (tau if delta<1e-8 else delta,distance(policy.position,item[2]))
        cw=sorted(anchors,key=lambda item:order(item,-1))
        ccw=sorted(anchors,key=lambda item:order(item,1))
        result=[];seen=set()
        for label,ordered in (('explore_clockwise',cw),('explore_counterclockwise',ccw)):
            target=next((a[2] for a in ordered if distance(policy.position,a[2])>25 and a[2] not in seen),None)
            if target is None:continue
            seen.add(target)
            actions=tuple(Action('measure',target,i,'rollout_explore') for i in policy.unknown()
                          if target not in policy.channels[i].measurements)
            if actions:result.append(Proposal(label,actions))
        return result

    def risk_check(self,policy,proposal,worlds,deadline,record):
        if self.witness_mode=='off' or distance(policy.position,proposal.actions[0].position)<350:
            record['risk_checked']=False
            return True
        witnesses,labels=compatible_witnesses(policy,proposal,worlds,deadline)
        record.update(risk_checked=True,risk_worlds=plain(witnesses),risk_labels=labels,
                      risk_rule='separate diagnostic/veto; never mixed into nominal mean',
                      risk_mode=self.witness_mode,risk_limit_s=self.risk_limit_s)
        baseline=Proposal('baseline',(policy.pending,),'baseline')
        best=record['best_index'];deltas=[]
        for i,world in enumerate(witnesses):
            a=self._evaluate_proposal(policy,baseline,world,deadline,f'risk-{i}',0,'risk')
            b=self._evaluate_proposal(policy,proposal,world,deadline,f'risk-{i}',best,'risk')
            deltas.append(b-a)
        loss=max([0.0]+deltas)
        record.update(risk_delta_s=deltas,risk_max_positive_delta_s=loss,risk_witness_count=len(witnesses),
                      risk_veto=self.witness_mode=='veto' and loss>self.risk_limit_s)
        if record['risk_veto']:
            self.stats['risk_vetoes']+=1
            return False
        return True

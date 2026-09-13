"""Optional public-feedback opportunity clears at existing robot stops.

Geometric certificates and actual clear success remain authoritative. Uniform
polygon area is only a speculative ranking assumption; failed trials neither
shrink a belief nor remove a pending channel. No simulator internals are read.
"""
from __future__ import annotations
import math
import bootstrap
from bootstrap import b
from online import State,HistoryDevice


def area(poly):
    return abs(sum(a[0]*z[1]-a[1]*z[0] for a,z in zip(poly,poly[1:]+poly[:1])))*.5


def approximate_hit_fraction(poly,here,radius=19.5):
    """Area inside an inscribed 32-gon; a prior for ranking, not a proof."""
    total=area(poly)
    if total<1e-8:
        return 0.
    clipped=poly[:]
    apothem=radius*math.cos(math.pi/32)
    for i in range(32):
        normal=b.unit(2*math.pi*i/32)
        clipped=b.clip_halfplane(clipped,normal,b.dot(normal,here)+apothem)
        if not clipped:
            return 0.
    return min(1.,area(clipped)/total)


class OpportunisticMixin:
    def __init__(self,*args,speculative_probability=0.,max_trials_per_channel=4,
                 min_trial_separation=20.,after_every_clear=True,**kwargs):
        self.speculative_probability=speculative_probability
        self.max_trials_per_channel=max_trials_per_channel
        self.min_trial_separation=min_trial_separation
        self.after_every_clear=after_every_clear
        self._opportunity_running=False
        self.opportunity_trial_points={}
        self.opportunity_stats=dict(certified_attempts=0,certified_clears=0,
                                    speculative_attempts=0,speculative_clears=0)
        super().__init__(*args,**kwargs)

    def execute(self,device,action):
        if self.after_every_clear:
            device=_ClearStopDevice(device,self,action[1] if action[0]=='target' else None)
        return super().execute(device,action)

    def shared(self,device,exclude=None):
        super().shared(device,exclude)
        self.clear_opportunities(device,exclude)

    def clear_opportunities(self,device,exclude=None):
        if self._opportunity_running:
            return
        self._opportunity_running=True
        try:
            self._clear_opportunities(device,exclude)
        finally:
            self._opportunity_running=False

    def _clear_opportunities(self,device,exclude=None):
        for channel in sorted(self.pending):
            if channel==exclude:
                continue
            belief=self.pending[channel]
            if belief.obs.status=='strong':
                certain=b.dist(device.position,belief.anchor)<=14.5
            else:
                certain=bool(belief.poly) and all(b.dist(device.position,p)<=19.5 for p in belief.poly)
            if not certain:
                if not self.speculative_probability or belief.obs.status=='strong':
                    continue
                previous=self.opportunity_trial_points.get(channel,[])
                if len(previous)>=self.max_trials_per_channel:
                    continue
                if any(b.dist(device.position,p)<self.min_trial_separation for p in previous):
                    continue
                center,radius=b.enclosing_circle(belief.poly)
                if b.dist(device.position,center)>radius+20:
                    continue
                probability=approximate_hit_fraction(belief.poly,device.position)
                if probability<self.speculative_probability:
                    continue
            category='certified' if certain else 'speculative'
            self.opportunity_stats[category+'_attempts']+=1
            self.opportunity_trial_points.setdefault(channel,[]).append(device.position)
            if not device.clear(channel):
                if certain:
                    raise RuntimeError('Certified opportunity clear failed')
                continue
            self.opportunity_stats[category+'_clears']+=1
            self.localizations.append(dict(channel=channel,
                stages=getattr(belief,'rf_rounds',0),pair_failures=getattr(belief,'pair_failures',0),
                certificate='opportunistic_'+category+'_confirmed',
                negative_updates=belief.negative_updates))
            self.cleared.add(channel)
            self.archived[channel]=belief
            del self.pending[channel]

    def report(self):
        result=super().report()
        result['opportunities']=self.opportunity_stats
        return result


class OpportunisticState(OpportunisticMixin,State):
    pass


class _ClearStopDevice:
    """Also inspect unsuccessful optical-cover stops, without recursive trials."""
    def __init__(self,device,owner,active_channel):
        self._device=device
        self._owner=owner
        self._active_channel=active_channel
    @property
    def position(self):return self._device.position
    @property
    def channel(self):return self._device.channel
    def move(self,p):return self._device.move(p)
    def detect(self,c):return self._device.detect(c)
    def clear(self,c):
        success=self._device.clear(c)
        self._owner.clear_opportunities(HistoryDevice(self._device,self._owner.history),self._active_channel if self._active_channel is not None else c)
        return success

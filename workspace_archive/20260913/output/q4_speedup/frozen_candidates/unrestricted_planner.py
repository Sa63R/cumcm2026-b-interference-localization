"""Experimental rollout with unrestricted emission types and rounded future bearings."""
from __future__ import annotations
import copy, math, random, statistics, time
from dataclasses import dataclass, asdict
import common
import q4_baseline as b
from q4_coverage import certify
from online import analytic_gain, finish
from unrestricted_posterior import Worlds, ConditionalSimulator

@dataclass
class PlanningConfig:
    coarse_scenarios:int=4
    verify_scenarios:int=4
    base_candidates:int=3
    max_decisions:int=4
    action_interval:int=5
    seconds_per_decision:float=3.0
    particles:int=32
    unknown_particles:int=128
    min_saved_per_source:float=.5
    max_validation_regret:float=30.
    extra_detour_m:float=75.
    shared:bool=False
    dynamic:bool=False

class Planner:
    def __init__(self,seed,config=None):
        self.seed=seed;self.config=config or PlanningConfig();self.calls=0;self.last_action=-999
        self.stats=dict(attempts=0,accepted=0,timeouts=0,posterior_fallbacks=0,invalid_rollouts=0,
                        candidates=0,rollout_runs=0,shared_candidates=0,shift_candidates=0,
                        coverage_checks=0,coverage_passed=0,wall_seconds=0.,decisions=[])

    def candidates(self,state,device,ordered):
        result=ordered[:self.config.base_candidates]
        here=device.position
        typ,k=ordered[0]
        destination=state.sites[k] if typ=='site' else state.pending[k].center()
        if self.config.shared and len(state.pending)>=2:
            length=b.dist(here,destination)
            direction=b.mul(1/max(length,1e-9),b.sub(destination,here));side=(-direction[1],direction[0])
            possible=[]
            for f in (.3,.6):
                mid=b.add(here,b.mul(f,b.sub(destination,here)))
                for lateral in (0.,30.,-30.):
                    q=b.add(mid,b.mul(lateral,side))
                    if b.dist(here,q)<50: continue
                    detour=b.dist(here,q)+b.dist(q,destination)-length
                    if detour>self.config.extra_detour_m: continue
                    gains=[]
                    for tr in state.pending.values():
                        if tr.obs.status=='strong': continue
                        if min(b.dist(q,p) for p in tr.measured)<state.config.share_min_distance: continue
                        center,radius=b.enclosing_circle(tr.poly)
                        if b.dist(q,center)>state.config.channel_potential_radius+radius: continue
                        gain=analytic_gain(tr.poly,tr.positives,tr.negatives,q)
                        if gain>=state.config.share_min_gain: gains.append(gain)
                    if len(gains)>=2: possible.append((sum(gains)/(1+detour),q))
            if possible:
                q=max(possible)[1];result.append(('shared',q));self.stats['shared_candidates']+=1
        if self.config.dynamic and state.unknown:
            # Use actual current/along-route positions to replace a planned
            # scan. Passing the test authorizes a revised plan, not early exit.
            site_keys=[a[1] for a in ordered if a[0]=='site' and a[1]!=0]
            site_keys.sort(key=lambda j:b.dist(here,state.sites[j]))
            found=False
            for j in site_keys[:2]:
                old=state.sites[j];d=b.dist(here,old)
                if d<1.: continue
                for shift in (d, min(d,150.), min(d,50.), min(d,20.)):
                    q=b.add(old,b.mul(shift/d,b.sub(here,old)))
                    plan=state.scanpoints+[state.sites[i] for i in sorted(state.remaining) if i!=j]+[q]
                    self.stats['coverage_checks']+=1
                    cert=certify(plan,max_depth=16)
                    if cert['ok']:
                        result.append(('shift',j,q));self.stats['shift_candidates']+=1;self.stats['coverage_passed']+=1
                        found=True
                        break
                if found: break
        return result

    def score(self,state,device,candidates,worlds,rng,count,deadline):
        costs=[[] for _ in candidates]
        for _ in range(count):
            targets,error_seed=worlds.sample(rng)
            for j,action in enumerate(candidates):
                if time.perf_counter()>=deadline: raise TimeoutError('decision budget')
                env=ConditionalSimulator(copy.deepcopy(targets),error_seed,state.history,device.position,device.channel)
                clone=copy.deepcopy(state)
                clone.execute(env,action)
                # Strictly use original V4 continuation after the candidate.
                clone.analytic=False
                finish(clone,env,deadline=deadline)
                if any(not t.cleared for t in env._targets.values()): raise RuntimeError('Continuation failed to clear a sampled source')
                costs[j].append(env.virtual_seconds/len(targets))
                self.stats['rollout_runs']+=1
        return costs

    def choose(self,state,device,ordered):
        cfg=self.config
        if not state.visited or len(ordered)<2 or self.calls>=cfg.max_decisions or state.actions-self.last_action<cfg.action_interval:
            return ordered[0]
        # Avoid expensive inference once no unresolved known source remains.
        if not state.pending and not cfg.dynamic: return ordered[0]
        self.calls+=1;self.last_action=state.actions;self.stats['attempts']+=1
        start=time.perf_counter();deadline=start+cfg.seconds_per_decision
        rng=random.Random(self.seed+self.calls*1000003)
        record=dict(action_index=state.actions,baseline=ordered[0],selected=ordered[0])
        try:
            candidates=self.candidates(state,device,ordered)
            self.stats['candidates']+=len(candidates)
            worlds=Worlds(state,rng,cfg.particles,cfg.unknown_particles)
            costs=self.score(state,device,candidates,worlds,rng,cfg.coarse_scenarios,deadline)
            means=list(map(statistics.mean,costs));winner=min(range(len(candidates)),key=lambda j:means[j])
            record['coarse_means']=means
            if winner==0 or means[0]-means[winner]<cfg.min_saved_per_source:
                record['reason']='coarse_no_gain';return ordered[0]
            # Rebuild location quadrature and independently sample validation
            # scenes, while pairing candidate and baseline within each scene.
            validate_rng=random.Random(self.seed+self.calls*2000003+991)
            verify=Worlds(state,validate_rng,cfg.particles,cfg.unknown_particles)
            values=self.score(state,device,[candidates[0],candidates[winner]],verify,validate_rng,cfg.verify_scenarios,deadline)
            delta=[z-a for a,z in zip(*values)]
            mean=statistics.mean(delta);se=statistics.stdev(delta)/math.sqrt(len(delta)) if len(delta)>1 else math.inf
            record.update(validation_delta=delta,validation_mean=mean)
            if mean+se < -cfg.min_saved_per_source and max(delta)<=cfg.max_validation_regret:
                self.stats['accepted']+=1;record['selected']=candidates[winner];record['reason']='validated'
                return candidates[winner]
            record['reason']='validation_rejected';return ordered[0]
        except TimeoutError:
            self.stats['timeouts']+=1;record['reason']='timeout';return ordered[0]
        except ValueError as e:
            self.stats['posterior_fallbacks']+=1;record['reason']='posterior_fallback';record['detail']=str(e);return ordered[0]
        except RuntimeError as e:
            self.stats['invalid_rollouts']+=1;record['reason']='invalid_rollout';record['detail']=str(e);return ordered[0]
        finally:
            elapsed=time.perf_counter()-start;self.stats['wall_seconds']+=elapsed
            record['wall_seconds']=elapsed;self.stats['decisions'].append(record)

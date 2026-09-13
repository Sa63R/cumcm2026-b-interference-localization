"""Generate continuous sensing proposals using inexpensive local rollouts.

This is a PROPOSAL heuristic, not the final value function: clear the selected
source with feedback-driven V4, then add travel to the next route entry. Final
candidate comparison can still use full, all-channel closed-loop rollout.
"""
from __future__ import annotations
import copy,math,random,statistics
from dataclasses import replace
import q4_baseline as b
from q4_state import Action,Engine
from q4_v4_local import localize_step
from q4_belief import ConditionedSimulator,keypoint


def local_candidates(engine,c,pool,ordered,rng,samples=12,top=2):
    st=engine.state;tr=st.pending[c];position=engine.device.position
    if tr.obs.status=='strong' or not tr.poly:return []
    from q4_v3_local import optical_cover
    cover=optical_cover(tr.poly)
    if cover and len(cover)<=engine.config.local.optical_cover_limit:return []
    upper=min(1500.,max(b.dist(tr.anchor,p) for p in tr.poly))
    e=b.unit(tr.obs.theta);v=(-e[1],e[0]);mean=pool.mean()
    xs=[b.dot(e,b.sub(p,tr.anchor)) for p in tr.poly];lo,hi=min(xs),max(xs)
    # Coordinate proposals, not just reordering a fixed list of stations.
    aa=[Action('target',c)]
    for axial,lateral in [(.3,.08),(.5,.035),(.65,.06),(.8,.04),(.4,.15),(.65,.15)]:
        aa.append(Action('pair',c,axial=axial,lateral=lateral))
    for f in [.25,.5,.75]:
        t=lo+f*(hi-lo)
        for side in [-1,1]:
            q=b.add(tr.anchor,b.add(b.mul(t,e),b.mul(side*max(15.,.035*upper),v)))
            if min(b.dist(q,p) for p in tr.measured)>20:aa.append(Action('probe',c,q))
    # Uniform conditional type/position samples from this target's RB pool.
    history={(c,keypoint(rec.point)):rec.obs for rec in st.rf[c]}
    w=[]
    po=.5*pool.omni_evidence;pd=.5*pool.dir_evidence
    for _ in range(samples):
        kind=1 if rng.random()<po/(po+pd) else 2
        g,r,u=pool.sample(kind,rng,__import__('radius_direction').RadiusPrior(.95,((1000.,.05),)))
        w.append((g,r,u,rng.randrange(1<<60)))
    next_action=next((a for a in ordered if not (a.kind=='target' and a.key==c)),None)
    exitpoint=None if next_action is None else (st.sites[next_action.key] if next_action.kind=='site' else st.pending[next_action.key].center())
    cfg=replace(engine.config,shared_at_probes=False,shared_at_clears=False)
    scores=[]
    for action in aa:
        vals=[]
        for g,r,u,seed in w:
            sim=ConditionedSimulator([b.Target(c,g,r,u)],seed,history)
            sim.position=position;sim.channel=engine.device.channel
            state=st.clone();state.pending={c:state.pending[c]};state.remaining.clear()
            state.cleared=set();state.cleared_tracks={};state.unknown=set()
            child=Engine(sim,cfg,state)
            child.execute(action)
            while c in child.state.pending:child.execute(Action('target',c))
            value=sim.virtual_seconds+(b.dist(sim.position,exitpoint)/5 if exitpoint else 0)
            vals.append(value)
        scores.append(statistics.mean(vals))
    ranked=sorted(range(1,len(aa)),key=lambda j:scores[j])
    return [(aa[j],scores[0]-scores[j]) for j in ranked[:top]]

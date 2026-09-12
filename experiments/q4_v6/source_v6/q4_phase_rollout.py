"""Select a coverage-preserving root rotation by full feedback-only continuations.
The source disk is rotationally invariant: rotating ALL 21 stations about its
center preserves the original all-direction continuous coverage proof.
"""
import random,math,statistics,time
from q4_belief import SceneFactory,PosteriorUnavailable
from q4_state import Engine

class PhaseEngine(Engine):
    def __init__(self,device,cfg,angles=8,scenes=12,validation=12,min_gain=1.):
        super().__init__(device,cfg);self.angles=angles;self.scenes=scenes;self.validation=validation;self.min_gain=min_gain;self.root_planned=False;self.phase_record={}
    def plan_phase(self):
        if self.root_planned or not self.state.visited:return
        self.root_planned=True
        if not self.state.pending:return
        tic=time.perf_counter();factory=SceneFactory(128,331391);rng=random.Random(53593)
        sites=self.state.sites[:];candidates=[]
        for k in range(self.angles):
            a=math.pi/2*k/self.angles;c=math.cos(a);s=math.sin(a)
            candidates.append([(c*x-s*y,s*x+c*y) for x,y in sites])
        def value(world,points):
            sim=world.device(self.device.position,self.device.channel);st=self.state.clone();st.sites=points[:]
            Engine(sim,self.config,st).run_base();return sim.virtual_seconds/world.count
        try:
            prepared=factory.prepare(self.state);worlds=factory.sample(self.state,self.scenes,rng,prepared)
            vals=[[value(w,ps) for w in worlds]for ps in candidates]
            diffs=[[v-b for v,b in zip(vs,vals[0])]for vs in vals]
            means=[statistics.mean(d) for d in diffs];j=min(range(len(vals)),key=lambda k:means[k])
            adopted=False;review=None
            if j and means[j]<-self.min_gain:
                reviews=factory.sample(self.state,self.validation,rng,prepared)
                ds=[value(w,candidates[j])-value(w,sites) for w in reviews]
                review=statistics.mean(ds)
                if review<-self.min_gain:
                    self.state.sites=candidates[j];adopted=True
            self.phase_record=dict(estimated_deltas=means,chosen=j,adopted=adopted,validation_delta=review,seconds=time.perf_counter()-tic)
        except PosteriorUnavailable as e:self.phase_record=dict(fallback=str(e),seconds=time.perf_counter()-tic)
    def ordered_actions(self):
        self.plan_phase();return super().ordered_actions()

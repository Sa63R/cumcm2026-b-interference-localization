"""Compare three archived V4 examples with the current runtime."""
import json,math,time
from pathlib import Path
import common
import q4_baseline as b
from q4_v4_solver import solve_v4
out=[]
for seed in [90210000,90210001,90210002]:
    old=json.loads((common.ROOT/'vendor_v4/results_v4'/f'LOCAL_ONLY_v4_{seed}.json').read_text())
    sim=b.LocalSimulator(b.make_case(seed,.5),seed,trace=True)
    start=time.perf_counter();solve_v4(sim);wall=time.perf_counter()-start
    divergent=None
    for i,(a,z) in enumerate(zip(old['actions'],sim.trace)):
        if a['action']!=z['action'] or math.dist(a['position'],z['position'])>1e-5 or a['channel']!=z['channel'] or a.get('status')!=z.get('status'):
            divergent=dict(index_zero_based=i,archived_action=a,current_action=z);break
    out.append(dict(seed=seed,archived_virtual_seconds=old['row']['virtual_seconds'],current_virtual_seconds=sim.virtual_seconds,wall_seconds=wall,all_cleared=sim.clear_count==len(sim._targets),same_virtual_time_to_1e_6=abs(old['row']['virtual_seconds']-sim.virtual_seconds)<1e-6,first_discrete_divergence=divergent))
(common.ROOT/'archive_replay_audit.json').write_text(json.dumps(dict(note='Observed runtime reproduction check; mismatch cause not established. Final comparisons use V4 freshly run on the same local runtime.',examples=out),indent=2,ensure_ascii=False))
print(json.dumps(out,indent=2,ensure_ascii=False))

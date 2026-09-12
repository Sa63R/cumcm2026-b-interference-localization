"""Extract immutable existing results for paper figures. No policy/simulator calls."""
from pathlib import Path
import json, gzip, hashlib, csv

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
DATA = OUT / 'data'
DATA.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value):
    (DATA/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

base=ROOT/'q3-state-search/research/final_results_v1'
all_stats={}
for split in ['random','stress']:
    p=base/f'report-{split}/comparison.json'
    c=read(p)
    all_stats[split]={'methods':c['methods'],'comparisons':c['comparisons'],
       'source':str(p),'source_sha256':sha(p),'protocol_sha256':c['protocol_sha256']}
save('four_methods.json',all_stats)
with (DATA/'four_methods.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['partition','method','n','success','mean_s','p95_s','movement_s','detection_s','switching_s','optical_s','removal_s','saving_s','ci_low_s','ci_high_s'])
    for part,c in all_stats.items():
        for key,m in c['methods'].items():
            cmp=c['comparisons'].get(key,{})
            costs=m['mean_components_s']
            w.writerow([part,key,m['runs'],m['successful_runs'],m['raw_mean_total_time_s'],m['raw_p95_total_time_s'],*[costs[k] for k in ['movement_s','detection_s','switching_s','optical_s','removal_s']],cmp.get('mean_seconds_saved',''),*cmp.get('saving_ci95_s',['',''])])

TRAJ=ROOT/'q3-geometric/research/final_trajectory_v1'
ARCH=ROOT/'q3-v1-artifacts/final-evaluations-complete-0930/v1-selection/evaluations/final_random'
for label in ['median','state_worst','rl_worst']:
    p=TRAJ/f'final_random_{label}.json'; c=read(p)
    seed=c['seed']; truths=[]; archive_refs=[]
    for name in ['baseline','state-future-cover','rl-gae095-u512','geo-future-cover']:
        q=ARCH/name/f'case-{seed}.json.gz'
        a=json.loads(gzip.decompress(q.read_bytes()))
        assert a['row']['case_sha256']==c['case_sha256']
        assert a['evaluation']['all_cleared']
        truths.append(a['evaluation']['ground_truth'])
        archive_refs.append({'source':str(q),'sha256':sha(q)})
    assert all(t==truths[0] for t in truths)
    save(f'trajectory_{label}.json',{'seed':seed,'selection':c['selection'],
        'case_sha256':c['case_sha256'],'ground_truth':truths[0],
        'plotted':c['plotted'],'source':str(p),'source_sha256':sha(p),
        'archives':archive_refs,'truth_use':'post-termination visualization only'})
print('Prepared source-backed aggregates and three paired trajectory cases.')

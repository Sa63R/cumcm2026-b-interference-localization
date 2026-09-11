"""Four opened development worlds; implementation smoke, not model selection."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'research/theory_v1')]
from research_rl.cpu_runtime import require_cpu
require_cpu()
import torch
from research_rl.train import main as train
from experiments.research_v1_eval import run_case
from simulation import Scenario, Source
from audit_eval_bounds import audit_record


def main():
    batch=ROOT/'results/paired_rl_state_20260911'
    manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8-sig'))
    parent=Path(manifest['policies']['rl_trial1']['spec']['kwargs']['checkpoint'])
    parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    assert parent_sha=='3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4'
    output=ROOT/'results/rl/range-smoke'
    torch.set_num_threads(1)
    args=['--output',str(output/'train'),'--initialize-from',str(parent),'--feature-version','v3',
          '--hidden','96','--architecture','mlp','--group-alpha','0','--probe-candidates','range_probes',
          '--device','cpu','--workers','0','--num-threads','1','--scenario-start','1990001',
          '--scenario-end','1990004','--max-attempted-episodes','4','--seed','97027',
          '--bc-episodes','0','--updates','2','--episodes-per-update','2','--epochs','1',
          '--minibatch','128','--lr','0.0001','--gae-lambda','0.95','--entropy-coef','0.005',
          '--max-wall-s','120']
    if not (output/'train/latest.pt').exists():
        assert not output.exists(), 'Use a fresh immutable smoke output'
        assert train(args)==0
    else:
        prior=torch.load(output/'train/latest.pt',map_location='cpu',weights_only=False)
        assert prior['state']['initialization']['sha256']==parent_sha
        assert prior['state']['update']==2 and prior['state']['episodes']==4
        assert prior['action_schema']['name']=='range_probes'
    endpoints={'parent':parent,'initialized':output/'train/initialized.pt','trained_smoke':output/'train/latest.pt'}
    bounds={(r['policy'],int(r['seed'])):r for r in csv.DictReader((batch/'per_case.csv').open(encoding='utf-8-sig'))}
    protocol={'limits':{'real_seconds_per_case':300,'virtual_seconds_per_case':360000,'max_actions':10000}}
    result={'scope':'Implementation smoke only, four already opened development cases; no performance promotion',
            'cpu_only':True,'training_seeds':[1990001,1990002,1990003,1990004],
            'development_seeds':[2100001,2100002,2100003,2100004],
            'parent_sha256':parent_sha,'probe_schema':'range_probes','source_sha256':{},'endpoints':{}}
    for path in sorted((ROOT/'src/research_rl').glob('*.py')):
        result['source_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,path in endpoints.items():
        folder=output/name;folder.mkdir(parents=True,exist_ok=True)
        rows=[]
        for seed in result['development_seeds']:
            spec={'name':name,'entrypoint':'research_rl:run_rl_search',
                  'kwargs':{'checkpoint':str(path),'device':'cpu','num_threads':1}}
            # Preserve the already opened world bytes. Cross-platform sin/cos
            # regeneration can change float.hex and the fixed observation error.
            original_path=batch/'remote_trial1'/f'case-{seed}.json.gz'
            assert hashlib.sha256(original_path.read_bytes()).hexdigest()==manifest['remote_records'][str(seed)]['record_sha256']
            original=json.loads(gzip.decompress(original_path.read_bytes()))
            config=original['evaluation']['ground_truth']
            # Truth is supplied only to the engine; the actor receives its
            # ordinary public client/geometry, as in the frozen fair comparison.
            case=Scenario(**{**config,'sources':tuple(Source(**s) for s in config['sources'])})
            record=run_case(case,spec,protocol)
            audit_record(record)
            expected=bounds[('rl_trial1',seed)]
            assert record['row']['case_sha256']==expected['case_sha256']
            raw=json.dumps(record,ensure_ascii=False).encode()
            target=folder/f'case-{seed}.json.gz';target.write_bytes(gzip.compress(raw,mtime=0))
            r=record['row'];lower=float(expected['primary_lower_bound_s'])
            learning=record['summary']['learning']
            rows.append({'seed':seed,'time_s':r['virtual_time_s'],'lower_bound_s':lower,
                'time_over_lower':r['virtual_time_s']/lower,'successful':r['successful'],
                'failed_clear_count':r['failed_clear_count'],'measurement_count':r['measurement_count'],
                'new_family_measurements':learning.get('range_probe_measurements',0),
                'record_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        result['endpoints'][name]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'records':rows,'mean_time_s':statistics.mean(r['time_s'] for r in rows),
            'total_time_over_total_lower':sum(r['time_s'] for r in rows)/sum(r['lower_bound_s'] for r in rows),
            'all_cleared_zero_failed':all(r['successful'] and r['failed_clear_count']==0 for r in rows)}
    payload=torch.load(endpoints['trained_smoke'],map_location='cpu',weights_only=False)
    result['training']={k:payload['state'][k] for k in ('update','episodes','attempted_episodes','optimizer_steps')}
    dest=ROOT/'research/range_probes';dest.mkdir(parents=True,exist_ok=True)
    (dest/'smoke.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ('mean_time_s','total_time_over_total_lower','all_cleared_zero_failed')} for k,v in result['endpoints'].items()},indent=2))


if __name__=='__main__':main()

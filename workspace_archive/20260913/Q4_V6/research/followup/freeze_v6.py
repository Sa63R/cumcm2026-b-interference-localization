"""Write a new, explicit evaluation manifest before inspecting its cases."""
import argparse,dataclasses,datetime,json,pathlib,platform,hashlib
from q4_v6 import default_config
from validate_v6 import hashes
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results_v6');p.add_argument('--main-cases',type=int,default=100);p.add_argument('--stress-cases',type=int,default=60);p.add_argument('--ablation-cases',type=int,default=20);p.add_argument('--seed-base',type=int,default=191100000);a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(exist_ok=True,parents=True)
    f=out/'frozen_config.json'
    if f.exists():raise SystemExit('Refusing to overwrite an existing frozen manifest; choose another output folder.')
    record={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'configuration':dataclasses.asdict(default_config()),'source_hashes':hashes(),
        'evaluation':{k:getattr(a,k)for k in ['main_cases','stress_cases','ablation_cases','seed_base']},
        'ablation_rule':'First 20 cases in each of nine groups, fixed before observing results.',
        'selection_rule':'Lowest average complete T/N on the common 90-case development set, seed base 157100000. No official cases used.',
        'data_notes':'The three additional test distributions also occur in offline training; these are independent cases, NOT unseen distribution claims.',
        'selected_training_cases':{'transit':2400,'route':480,'union':2400},
        'training_seed_bases':[171100000,181100000],
        'other_experimental_training_seed_base':185100000,
        'development_seed_bases':[147100000,151100000,155100000,157100000],
        'python_version':platform.python_version(),'local_only':True,'failed_optical_seconds':3}
    f.write_text(json.dumps(record,ensure_ascii=False,indent=2));print(f)

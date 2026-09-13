from pathlib import Path
import argparse,sys,json,time,statistics
p=argparse.ArgumentParser();p.add_argument('--wait',type=float,default=0);args=p.parse_args()
if args.wait:time.sleep(args.wait)
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent/'q4_v6_lite_official_20260913'))
import utm_guest as u
root=r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913\practice_100each_20260913'
result={}
for name in ('status.json','completed.json'):
 raw=u.call('file','pull',u.VM,root+'\\'+name)
 if not raw:continue
 data=json.loads(raw.decode('utf-8-sig'));(base/name).write_bytes(raw)
 if name=='status.json':result.update({k:data[k] for k in ('utc','status','completed','q3_completed','q4_completed','index','phase','message')})
 else:
  if isinstance(data,dict):data=[data]
  for q in (3,4):
   rows=[r for r in data if r['question']==q]
   if rows:result[f'Q{q}']={'runs':len(rows),'sources':sum(r['total'] for r in rows),'all_cleared':all(r['cleared']==r['total'] for r in rows),'pooled_s_per_source':round(sum(r['virtual_s'] for r in rows)/sum(r['cleared'] for r in rows),4),'mean_wall_s':round(statistics.mean(r['wall_s'] for r in rows),4),'latest':rows[-1]['case']}
print(json.dumps(result,ensure_ascii=False))
with (base/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')

from pathlib import Path
import argparse,sys,json,hashlib,zipfile
p=argparse.ArgumentParser();p.add_argument('count',type=int);args=p.parse_args()
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent/'q4_v6_lite_official_20260913'));import utm_guest as u
root=r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913'
name=f'practice_checkpoint_{args.count:03d}'
meta=u.call('file','pull',u.VM,root+'\\'+name+'_status.json');status=json.loads(meta.decode('utf-8-sig'))
assert status['status']=='complete' and status['runs']==args.count
raw=u.call('file','pull',u.VM,status['archive'])
assert len(raw)==status['bytes'] and hashlib.sha256(raw).hexdigest()==status['sha256']
archive=base/(name+'.zip');archive.write_bytes(raw)
dest=base/name;dest.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 manifest=json.loads(z.read('export_manifest.json'))
 for name in z.namelist():
  target=dest/name
  assert target.resolve().is_relative_to(dest.resolve())
  data=z.read(name)
  if name in manifest['files']:
   info=manifest['files'][name];assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256']
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
print(json.dumps({'downloaded':args.count,'bytes':len(raw),'files':len(manifest['files']),'folder':str(dest)},ensure_ascii=False))

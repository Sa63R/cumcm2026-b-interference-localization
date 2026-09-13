"""Send one reviewed UI action to the isolated Windows desktop worker."""
import argparse,json,time,uuid
from pathlib import Path
from utm_guest import call,VM
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['snapshot','maximize','click','quit']);ap.add_argument('--x',type=int);ap.add_argument('--y',type=int);a=ap.parse_args()
if a.action=='click' and (a.x is None or a.y is None):ap.error('click needs x and y')
ident=str(time.time_ns())+'_'+uuid.uuid4().hex[:6]
root=r'C:\Users\baiwc\Downloads\Q3Practice'
request=dict(id=ident,action=a.action,x=a.x,y=a.y)
call('file','push',VM,root+'\\desktop_requests\\'+ident+'.json',data=json.dumps(request).encode())
start=time.monotonic()
while time.monotonic()-start<25:
    time.sleep(.5)
    raw=call('file','pull',VM,root+'\\desktop_results\\'+ident+'.json')
    try:r=json.loads(raw.decode('utf-8-sig'))
    except (ValueError,UnicodeError):continue
    folder=Path(__file__).resolve().parent/'results'/'desktop';folder.mkdir(parents=True,exist_ok=True)
    (folder/(ident+'.json')).write_text(json.dumps(r,indent=2))
    if r['status']=='ok':
        image=call('file','pull',VM,r['image'])
        path=folder/(ident+'.png');path.write_bytes(image)
        r['local_image']=str(path)
        (folder/'latest.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,ensure_ascii=False));break
else:raise SystemExit('Timed out waiting for isolated guest desktop worker; inspect its status.')

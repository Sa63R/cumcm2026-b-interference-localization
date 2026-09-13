"""Ask Windows to archive committed cases, download, hash-check and extract."""
from pathlib import Path
import hashlib,json,time,zipfile
from utm_guest import call,ps,VM

here=Path(__file__).resolve().parent
cfg=json.loads((here/'bulk_config.json').read_text())
guest=r'C:\Users\baiwc\Downloads\Q3Practice\batches'+'\\'+cfg['batch_id']
folder=here/'results/bulk'/cfg['batch_id'];folder.mkdir(parents=True,exist_ok=True)
before=json.loads(call('file','pull',VM,guest+'\\status.json').decode('utf-8-sig'))
minimum=before['completed']
ps("& C:\\Windows\\System32\\cmd.exe /c 'C:\\Users\\baiwc\\Downloads\\Q4Practice\\python\\python.exe C:\\Users\\baiwc\\Downloads\\Q3Practice\\collect_bulk.py > C:\\Users\\baiwc\\Downloads\\Q3Practice\\collect_bulk.log 2>&1'")
deadline=time.monotonic()+180
while time.monotonic()<deadline:
    time.sleep(2)
    raw=call('file','pull',VM,guest+'\\collection.json')
    try:info=json.loads(raw.decode('utf-8-sig'))
    except (ValueError,UnicodeError):continue
    if info['completed']>=minimum:break
else:raise RuntimeError('Guest collection did not finish; inspect collect_bulk.log')
data=call('file','pull',VM,info['archive'])
assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256']
archive=folder/Path(info['archive'].replace('\\','/')).name
archive.write_bytes(data)
extract=folder/'evidence_files'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    idx=json.loads(z.read('file_index.json'))
    for item in idx:
        content=z.read(item['path'])
        assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256']
    for name in z.namelist():
        target=(extract/name).resolve();assert target.is_relative_to(extract.resolve())
        content=z.read(name)
        if target.exists() and name not in {'completed.jsonl','bulk_config.json','status_at_collection.json','file_index.json'}:
            assert target.read_bytes()==content, 'Previously archived evidence changed: '+name
            continue
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
info['local_archive']=str(archive);info['extracted_to']=str(extract);info['verified']=True
(folder/'collection.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,ensure_ascii=False,indent=2))

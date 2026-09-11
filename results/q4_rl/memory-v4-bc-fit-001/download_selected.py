"""Approved object-store reads only, fixed BC batches 1 and 16, no hot latest."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

spec=importlib.util.spec_from_file_location('exchange',Path('scripts/q4_object_exchange.py'))
exchange=importlib.util.module_from_spec(spec)
spec.loader.exec_module(exchange)
parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['prepare','episodes'])
parser.add_argument('--jobs',nargs='+',choices=['g1_h64','g3_h64','g1_h128','g3_h128'],required=True)
args=parser.parse_args()
client=exchange.connection(Path('../AGENTS.md'))  # Consumed in memory; never print its contents.
bucket='bucket-c20250204-pool01'
task='q4-rl-memory-v4-20260912'
base=Path('handoff/v4-bc-fit-readback')
started=time.perf_counter()

def execute(job):
    directory=base/job
    directory.mkdir(exist_ok=True)
    manifest_path=directory/'OBJECT_READBACK.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'task':task,'job':job,'selection':'first and sixteenth complete BC batches, fixed before reading losses','objects':[]}
    indexed={item['name']:item for item in manifest['objects']}
    prefix=f'lianghao/bwc/shumo/{task}/runs/train-memory-v4/{job}/training/'
    def fetch(name, expected=None):
        if '/' in name or '\\' in name or name=='latest.pt':
            raise ValueError('Forbidden object name')
        target=directory/name
        if target.exists():
            raw=target.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            if name not in indexed or digest!=indexed[name]['sha256']:
                raise ValueError('Existing local evidence cannot be overwritten')
        else:
            response=client.get_object(Bucket=bucket,Key=prefix+name)
            try: raw=response['Body'].read()
            finally: response['Body'].close()
            if len(raw)!=response['ContentLength']:
                raise ValueError('Object size mismatch')
            digest=hashlib.sha256(raw).hexdigest()
            if expected is not None and digest!=expected:
                raise ValueError('Linked episode hash mismatch')
            with target.open('xb') as stream: stream.write(raw)
            entry={'name':name,'bytes':len(raw),'sha256':digest,'etag':response.get('ETag')}
            manifest['objects'].append(entry)
            indexed[name]=entry
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        if expected is not None and digest!=expected:
            raise ValueError('Linked episode hash mismatch')
        return raw
    if args.phase=='prepare':
        fetch('warmstart.pt')
        selected=[]
        for number in [1,16]:
            progress=json.loads(fetch(f'progress-{number:06d}.json'))
            if progress['phase']!='imitation' or progress['batch']!=number or progress['episodes']!=number*16:
                raise ValueError('The fixed BC endpoint is not complete')
            name=progress['raw_attempt']
            index=json.loads(gzip.decompress(fetch(name)))
            if index['format']!='q4-training-episode-index-v1' or len(index['episodes'])!=16:
                raise ValueError('Selected complete teacher index is not ready')
            selected.append({'batch':number,'index':name,'episodes':index['episodes']})
        (directory/'SELECTION.json').write_text(json.dumps({'selected':selected},indent=2)+'\n',encoding='utf-8')
    else:
        verified=json.loads((directory/'MODEL_VERIFIED.json').read_text())
        if verified['BC_episodes']!=256 or verified['ppo_batches']!=0 or verified['pending_batch'] is not None:
            raise ValueError('Model must first be verified as frozen complete BC256')
        selection=json.loads((directory/'SELECTION.json').read_text())
        for selected in selection['selected']:
            for entry in selected['episodes']:
                rows=json.loads(gzip.decompress(fetch(entry['path'],entry['sha256'])))
                if len(rows)!=1 or rows[0]['seed']!=entry['seed']:
                    raise ValueError('Linked teacher identity mismatch')
                del rows
    print(json.dumps({'job':job,'phase':args.phase,'objects':len(manifest['objects']),
        'bytes':sum(item['bytes'] for item in manifest['objects']),'elapsed_s':time.perf_counter()-started}),flush=True)

for job in args.jobs:
    try: execute(job)
    except Exception as exc:
        print(json.dumps({'job':job,'phase':args.phase,'not_ready_or_failed':True,'error_type':type(exc).__name__}),flush=True)

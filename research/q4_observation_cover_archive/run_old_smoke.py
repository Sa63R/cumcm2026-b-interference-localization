"""Authorized interface QA: only old621001, once per fixed ring configuration."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.run_q4_round2 import one,hashes
from experiments.audit_q4_observation_cover import audit_full


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    output=Path(__file__).with_name('old-smoke')
    output.mkdir(exist_ok=False)
    (output/'records').mkdir()
    source=hashes()
    write(output/'source-sha256.json',source)
    with zipfile.ZipFile(output/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(ROOT/name,name)
    rows=[]
    for config in ('ring_28','ring_31'):
        label='compact_'+config
        spec={'entrypoint':'strategies.q4_observation_cover:run_q4_observation_cover',
              'kwargs':{'config':config,'max_expansions':200}}
        record=one(621001,'pilot',label,spec,source)
        data=json.dumps(record,ensure_ascii=False,allow_nan=False).encode()
        with (output/'records'/f'{label}-621001.json.gz').open('xb') as stream:
            stream.write(gzip.compress(data,mtime=0))
        try:
            audit=audit_full(record)
        except Exception as error:
            audit=dict(passed=False,error=str(error),traceback=traceback.format_exc())
        write(output/f'{label}-audit.json',audit)
        row=dict(record['row'],raw_json_sha256=hashlib.sha256(data).hexdigest(),audit_passed=audit['passed'])
        row['time_per_source_s']=row['virtual_time_s']/row['source_total']
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
    write(output/'summary.json',dict(scope='Only old621001 interface QA; not candidate selection or independent performance evidence',rows=rows))


if __name__=='__main__':main()

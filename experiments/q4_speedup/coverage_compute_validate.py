"""One paired 100-input proof comparison, excluding wall from equality only."""
import bootstrap
import gzip,hashlib,json,statistics,time
from pathlib import Path
import q4_coverage
import coverage_candidate
from coverage_compute_fast import CoverageComputeCache

ROOT=Path(__file__).resolve().parents[3]/'output/q4_speedup/schedule_dev/coverage_compute100'


def main():
    manifest=json.loads((ROOT/'inputs.json').read_text());inputs=manifest['inputs'];assert len(inputs)==100
    original=q4_coverage.certify;original_alias=coverage_candidate.certify
    normalize=lambda result:{k:v for k,v in result.items() if k!='wall'}
    # The optional context updates already-imported aliases and always restores
    # them. Each certificate releases its own bounded hull cache on return.
    try:
        with CoverageComputeCache(cache_limit=2) as context:
            self_result=q4_coverage.certify([(0.,0.)])
            assert coverage_candidate.certify is q4_coverage.certify
            assert not self_result['ok'] and context.stats()['peak_call_entries']<=2
            try:
                with CoverageComputeCache():pass
            except RuntimeError:pass
            else:raise AssertionError('nested context accepted')
            raise LookupError('expected restoration check')
    except LookupError:pass
    assert q4_coverage.certify is original and coverage_candidate.certify is original_alias
    rows=[];expected_calls=0
    with (ROOT/'cases.jsonl').open('w') as log,gzip.open(ROOT/'identical_proofs.jsonl.gz','wt') as proofs:
        for i,item in enumerate(inputs):
            sites=[tuple(p) for p in item['sites']]
            kwargs=dict(max_depth=item['max_depth'],keep_leaves=item['keep_leaves'])
            pair={}
            for fast in ([False,True] if i%2==0 else [True,False]):
                begin_cpu=time.process_time();begin_wall=time.perf_counter()
                if fast:
                    with CoverageComputeCache() as context:result=q4_coverage.certify(sites,**kwargs)
                else:result=original(sites,**kwargs)
                wall=time.perf_counter()-begin_wall;cpu=time.process_time()-begin_cpu
                pair[fast]=dict(result=result,cpu=cpu,wall=wall,cache=context.stats() if fast else None)
                assert q4_coverage.certify is original and coverage_candidate.certify is original_alias
            old,new=pair[False],pair[True]
            if normalize(old['result'])!=normalize(new['result']):
                (ROOT/f'mismatch_{i}.json').write_text(json.dumps(pair,indent=2))
                raise AssertionError(f'Proof fields differ for input {i}')
            proof=normalize(old['result']);text=json.dumps(proof,separators=(',',':'))
            row=dict(index=i,source=item['source'],site_count=len(sites),**kwargs,equal_except_wall=True,
                     ok=proof['ok'],kind='ok' if proof['ok'] else ('witness' if 'witness' in proof else 'unresolved'),
                     accepted=proof['accepted'],outside=proof['outside'],subdivided=proof['subdivided'],deepest=proof['max_depth'],
                     proof_sha256=hashlib.sha256(text.encode()).hexdigest(),baseline_cpu_s=old['cpu'],accelerated_cpu_s=new['cpu'],
                     baseline_wall_s=old['wall'],accelerated_wall_s=new['wall'],cache=new['cache'])
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush();proofs.write(json.dumps(dict(index=i,proof=proof))+'\n')
            if (i+1)%20==0:print('certificates compared',i+1,flush=True)
    base_cpu=sum(r['baseline_cpu_s'] for r in rows);fast_cpu=sum(r['accelerated_cpu_s'] for r in rows)
    base_wall=sum(r['baseline_wall_s'] for r in rows);fast_wall=sum(r['accelerated_wall_s'] for r in rows)
    differences=[r['baseline_cpu_s']-r['accelerated_cpu_s'] for r in rows]
    mean=statistics.mean(differences);se=statistics.stdev(differences)/(len(rows)**.5)
    result=dict(inputs=len(rows),all_fields_except_wall_identical=True,ok=sum(r['ok'] for r in rows),
                witness=sum(r['kind']=='witness' for r in rows),unresolved=sum(r['kind']=='unresolved' for r in rows),
                keep_leaves_inputs=sum(r['keep_leaves'] for r in rows),
                accepted_boxes=sum(r['accepted'] for r in rows),outside_boxes=sum(r['outside'] for r in rows),
                subdivided_boxes=sum(r['subdivided'] for r in rows),
                baseline_cpu_s=base_cpu,accelerated_cpu_s=fast_cpu,cpu_reduction=1-fast_cpu/base_cpu,speedup=base_cpu/fast_cpu,
                baseline_wall_s=base_wall,accelerated_wall_s=fast_wall,wall_reduction=1-fast_wall/base_wall,
                paired_normal95_cpu_saved_s=[mean-1.96*se,mean+1.96*se],
                cache_hits=sum(r['cache']['hits'] for r in rows),cache_misses=sum(r['cache']['misses'] for r in rows),
                peak_call_entries=max(r['cache']['peak_call_entries'] for r in rows),cache_limit=512,
                lifecycle_checks=dict(alias_restoration=True,exception_restoration=True,nested_rejection=True,per_call_cache_release=True),
                code_sha256=hashlib.sha256((Path(__file__).parent/'coverage_compute_fast.py').read_bytes()).hexdigest(),
                original_code_sha256=hashlib.sha256(Path(q4_coverage.__file__).read_bytes()).hexdigest(),
                note='Nonexclusive machine. One paired serial comparison, alternating baseline/accelerated order. 90 inputs come from 286 actual unmodified cells_flex certificate trials in 3 development cases; 10 edge cases cover low depth, empty/degenerate/duplicate and scaled/rotated layouts. This validates certificate computation only, not an end-to-end rollout speed claim.')
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    (ROOT/'coverage_compute_fast_tested.py').write_bytes((Path(__file__).parent/'coverage_compute_fast.py').read_bytes())
if __name__=='__main__':main()

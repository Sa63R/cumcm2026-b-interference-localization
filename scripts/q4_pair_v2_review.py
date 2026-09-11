"""Read-only archived-panel audit and descriptive micro fallback diagnosis.

No training, new feedback, case generation or checkpoint execution is performed.
Writes only a new independent review artifact; archived results stay unchanged.
"""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.audit_q4_state import wire_audit

FOLDER=ROOT/'results/q4_rl/server-pair-v2-evaluation-001'
OUT=ROOT/'results/q4_rl/pair-v2-independent-review-001.json'
MICRO=('micro_rule512','micro_bc512','micro_ppo512')
COMPONENTS=('movement_s','detection_s','switching_s','optical_s','removal_s')
EXPECTED_SUMMARY='f51e0ebc97afcbdf271e14e8aaf2ec26a3584d85797a72df5bdecee5f4e4314f'


def sha(blob): return hashlib.sha256(blob).hexdigest()
def digest(value): return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode())
def read(name): return json.loads((FOLDER/name).read_bytes())
def xy(value): return (value['x'],value['y']) if isinstance(value,dict) else tuple(value)
def quantile(values,q):
    v=sorted(values);a=q*(len(v)-1);lo=math.floor(a)
    return v[lo]*(1-(a-lo))+v[math.ceil(a)]*(a-lo)
def percentiles(values): return {label:quantile(values,q) for label,q in [('median',.5),('p90',.9),('p95',.95),('max',1.)]} if values else None


def paired_means(rows,reference,target):
    baseline={r['case_id']:r for r in rows if r['strategy']==reference}
    groups=defaultdict(list)
    for r in rows:
        if r['strategy']==target: groups[r['seed']].append((baseline[r['case_id']]['penalized_time_s'],r['penalized_time_s']))
    seeds=sorted(groups)
    means={seed:(statistics.mean(b for b,c in groups[seed]),statistics.mean(c for b,c in groups[seed])) for seed in seeds}
    rng=random.Random(4260911);saved=[];relative=[]
    for _ in range(5000):
        selected=rng.choices(seeds,k=len(seeds))
        b=statistics.mean(means[k][0] for k in selected)
        c=statistics.mean(means[k][1] for k in selected)
        saved.append(b-c);relative.append(1-c/b)
    b=statistics.mean(v[0] for v in means.values());c=statistics.mean(v[1] for v in means.values())
    return dict(reference=reference,target=target,clusters=len(seeds),mean_saved_s=b-c,
        relative_mean_saving=1-c/b,mean_saved_ci95_s=[quantile(saved,.025),quantile(saved,.975)],
        relative_saved_ci95=[quantile(relative,.025),quantile(relative,.975)])


def arm_metrics(rows):
    output={}
    for method in sorted({r['strategy'] for r in rows}):
        group=[r for r in rows if r['strategy']==method]
        total=sum(r['penalized_time_s'] for r in group)
        fallback=[r['fallback_virtual_s'] for r in group]
        output[method]=dict(runs=len(group),seed_clusters=len({r['seed'] for r in group}),
            successful=sum(r['successful'] for r in group),failed_clears=sum(r['failed_clear_count'] for r in group),
            means={k:statistics.mean(r[k] for r in group) for k in ('virtual_time_s','measurement_count','fallback_virtual_s',
                'process_cpu_s','program_runtime_s','audit_runtime_s','lower_bound_runtime_s')+COMPONENTS},
            ratio_of_sums=total/sum(r['common_lower_bound_s'] for r in group),
            mean_individual_ratio=statistics.mean(r['penalized_time_over_lower_bound'] for r in group),
            time_percentiles=percentiles([r['penalized_time_s'] for r in group]),
            fallback_count=sum(v>1e-6 for v in fallback),fallback_percentiles_all=percentiles(fallback),
            fallback_percentiles_positive=percentiles([v for v in fallback if v>1e-6]),
            fallback_sum_share=sum(fallback)/total)
    return output


def micro_boundary(record):
    report=record['summary'];learning=report['learning'];history=report['action_history'];row=record['row']
    steps=learning['micro_steps']
    assert len(steps)==learning['decisions']
    assert all(s['accepted_requests']==1 and s['end_actual_action_index']==s['before_actual_action_index']+1 for s in steps)
    for s in steps:
        action=history[s['before_actual_action_index']]
        assert action['action']==('measure' if s['kind']=='measure' else 'clear')
        assert action['channel']==s['channel'] and action['position']==s['position']
    for check in learning['certified_clear_checks']:
        action=history[check['actual_action_index']]
        assert action['action']=='clear' and action['channel']==check['channel'] and action['result']=='success'
        assert check['result']=='success' and check['max_distance_upper_m']<=check['operational_radius_m']-check['guard_m']
    assert math.isclose(sum(s['cost_s'] for s in steps),learning['decision_cost_s'],abs_tol=1e-5)
    assert math.isclose(learning['decision_cost_s']+learning['fallback_cost_s']+learning['uncovered_cost_s'],row['virtual_time_s'],abs_tol=1e-5)
    if not learning['fallback_reason']: return None
    cut=steps[-1]['end_actual_action_index'] if steps else 0
    known,cleared=set(),set();actual=defaultdict(set);roles=Counter();scans=Counter()
    for item,step in zip(history[:cut],steps):
        c=item['channel']
        if item['action']=='measure':
            roles[step['role']]+=1
            scans['already_known' if c in known else 'not_yet_known']+=1
            actual[c].add(xy(item['position']))
            if item['result'] in ('near','direction'): known.add(c)
        elif item['result']=='success': known.add(c);cleared.add(c)
    points={xy(p) for p in report['coverage_points']}
    unknown=set(range(1,21))-known
    pending={c:len(points-actual[c]) for c in unknown}
    discovery=len(known)==16 or all(v==0 for v in pending.values())
    found_all=len(known)==row['source_total']
    return dict(case_id=row['case_id'],strategy=row['strategy'],seed=row['seed'],family=row['family'],
        source_mode=row['source_mode'],source_total=row['source_total'],reason=learning['fallback_reason'],
        decisions=learning['decisions'],known=len(known),cleared=len(cleared),
        known_unresolved=len(known-cleared),posthoc_all_real_sources_found=found_all,
        public_discovery_certified=discovery,pending_unknown_channels=sum(v>0 for v in pending.values()),
        pending_unknown_pairs=sum(pending.values()),prefix_measure_roles=dict(roles),prefix_measure_known_status=dict(scans),
        prefix_grid_clears=sum(s['kind']=='grid_clear' for s in steps),
        prefix_certified_clears=sum(s['kind']=='certified_clear' for s in steps),
        fallback_s=row['fallback_virtual_s'],virtual_time_s=row['virtual_time_s'],
        time_over_lower_bound=row['time_over_lower_bound'])


def aggregate_boundaries(items):
    result={}
    for mode,predicate in [('all',lambda x:True),('n16',lambda x:x['source_total']==16),('n_lt16',lambda x:x['source_total']<16)]:
        result[mode]={}
        for method in MICRO:
            selected=[x for x in items if x['strategy']==method and predicate(x)]
            if not selected: continue
            categories=Counter()
            for x in selected:
                if not x['posthoc_all_real_sources_found']: categories['not_all_real_sources_found']+=1
                elif not x['public_discovery_certified']: categories['all_real_sources_found_but_empty_channel_proof_pending']+=1
                else: categories['public_discovery_complete_but_uncleared']+=1
            role=Counter();status=Counter()
            for x in selected: role.update(x['prefix_measure_roles']);status.update(x['prefix_measure_known_status'])
            result[mode][method]=dict(runs=len(selected),seed_clusters=len({x['seed'] for x in selected}),
                reasons=dict(Counter(x['reason'] for x in selected)),exclusive_cut_categories=dict(categories),
                means={k:statistics.mean(x[k] for x in selected) for k in ('decisions','known','cleared','known_unresolved',
                    'pending_unknown_channels','pending_unknown_pairs','prefix_grid_clears','prefix_certified_clears','fallback_s')},
                prefix_measure_roles_total=dict(role),prefix_measure_known_status_total=dict(status),
                examples=sorted(selected,key=lambda x:-x['fallback_s'])[:2])
    return result


def main():
    began=time.perf_counter()
    if OUT.exists(): raise ValueError('Review output already exists; keep prior evidence')
    summary_blob=(FOLDER/'summary.json').read_bytes();assert sha(summary_blob)==EXPECTED_SUMMARY
    summary=json.loads(summary_blob);rows=summary['rows'];methods=sorted(summary['summaries'])
    assert summary['bootstrap']==dict(samples=5000,seed=4260911,unit='paired seed cluster')
    expected={(seed,family,mode,method) for seed in range(8101000,8101032) for family in
        ('random','minimum_radius','boundary_outward','cluster','positive_error','negative_error','alternating_error','narrow_strip')
        for mode in ('mixed','all_directional') for method in methods}
    assert len(rows)==3584 and len(methods)==7 and len(expected)==len(rows)
    assert {(r['seed'],r['family'],r['source_mode'],r['strategy']) for r in rows}==expected
    indexed={(r['case_id'],r['strategy']):r for r in rows}
    manifest,evidence,readback=read('manifest.json'),read('evidence.json'),read('OBJECT_READBACK.json')
    assert manifest['bootstrap_samples']==5000 and manifest['workers']==6 and manifest['numerical_threads_per_worker']==1
    assert manifest['stage']=='development' and not manifest['formal_simulator'] and not manifest['practice_simulator']
    assert sha((FOLDER/'manifest.json').read_bytes())==evidence['manifest_sha256']
    assert sha(summary_blob)==evidence['summary_sha256']
    assert digest(manifest)==read('freeze.json')['manifest_sha256']
    assert sha((FOLDER/'specs.original.json').read_bytes())==manifest['raw_specs_sha256']
    assert read('specs.original.json')==manifest['specs']
    with zipfile.ZipFile(FOLDER/'source.zip') as archive:
        assert archive.testzip() is None
        for name,h in manifest['source_sha256'].items(): assert sha(archive.read(name))==h
        assert sha(archive.read('research/q4_rl/protocol.json'))==manifest['protocol_sha256']
    weights={}
    for name,h in manifest['artifact_sha256'].items():
        local=ROOT/'results/q4_rl/server-pair-v2-complete-001'/name.removeprefix('runs/train-pair-v2/')
        assert sha(local.read_bytes())==h
        weights[name]=h
    boundary=[];wire_count=0;checked_bytes=0;checked_objects=0
    original_component_digests={}
    for obj in readback['objects']:
        name=obj['name'];path=FOLDER/name
        assert path.resolve().is_relative_to(FOLDER.resolve())
        blob=path.read_bytes();assert len(blob)==obj['bytes'] and sha(blob)==obj['sha256']
        checked_bytes+=len(blob);checked_objects+=1
        if not name.startswith('records/'): continue
        assert sha(blob)==evidence['record_sha256'][path.name]
        record=json.loads(gzip.decompress(blob));row=record['row'];key=(row['case_id'],row['strategy'])
        assert row==indexed[key] and record['spec']==manifest['specs'][row['strategy']]
        assert row['successful'] and row['audit_passed'] and row['all_cleared'] and not row['errors']
        assert record['audit']['passed'] and record['summary']['completion_certified_under_model']
        assert digest(record['evaluation']['ground_truth'])==row['case_sha256']
        assert record['common_lower_bound']['common_lower_bound_s']==row['common_lower_bound_s']
        assert math.isclose(row['virtual_time_s'],sum(row[k] for k in COMPONENTS),abs_tol=1e-6)
        assert row['penalized_time_s']==row['virtual_time_s']
        assert math.isclose(row['time_over_lower_bound'],row['virtual_time_s']/row['common_lower_bound_s'],rel_tol=1e-12)
        other=original_component_digests.setdefault(row['case_id'],(row['case_sha256'],row['common_lower_bound_s']))
        assert other==(row['case_sha256'],row['common_lower_bound_s'])
        assert row['stage']=='development'
        wire_audit(record);wire_count+=1
        if row['strategy'] in MICRO:
            item=micro_boundary(record)
            if item: boundary.append(item)
        if wire_count%512==0: print(json.dumps(dict(archived_records_verified=wire_count)),flush=True)
    assert checked_objects==readback['files']==3591 and checked_bytes==readback['bytes']==421434097
    assert wire_count==3584 and len(evidence['record_sha256'])==3584
    groups={}
    for name,predicate in [('all',lambda r:True),('random',lambda r:r['family']=='random'),('stress',lambda r:r['family']!='random')]:
        chosen=[r for r in rows if predicate(r)]
        groups[name]=dict(arms=arm_metrics(chosen),paired_to_r8={m:paired_means(chosen,'r8',m) for m in methods if m!='r8'},
            learning_differences=[paired_means(chosen,a,b) for a,b in (
                ('macro_bc512','macro_ppo512'),('micro_bc512','micro_ppo512'),
                ('macro_rule512','macro_ppo512'),('micro_rule512','micro_ppo512'))])
    strata={key:dict(arms={m:{k:v for k,v in stats.items() if k in ('mean_actual_elapsed_time_s','p95_penalized_time_s','ratio_of_sums','failed_clear_total')}
        for m,stats in value['summaries'].items()},paired={m:{k:v for k,v in stats.items() if k in
        ('relative_mean_saving','relative_saving_ci95','p95_ratio','mean_saved_s','mean_saving_ci95_s')}
        for m,stats in value['paired'].items()}) for key,value in summary['strata'].items()}
    review=dict(summary_sha256=EXPECTED_SUMMARY,review_source_sha256=sha(Path(__file__).read_bytes()),
        scope='Independent offline archive audit, no new scene or policy execution; N grouping and boundary labels exploratory',
        validation=dict(readback_objects=checked_objects,readback_bytes=checked_bytes,raw_rows_and_wire_audits=wire_count,
            source_files=len(manifest['source_sha256']),source_crc_and_sha256=True,freeze_manifest_and_specs=True,
            frozen_weights_sha256=weights,panel_pairing=True,physical_audits=True,
            completion_audit_scope='All stored completion audits checked; physical wire replay repeated; expensive directional completion proof not rerun for every record',
            micro_atomic_steps_and_safe_clear_records=True,elapsed_wall_s=time.perf_counter()-began),
        groups=groups,strata=strata,by_source_count={name:arm_metrics([r for r in rows if predicate(r) and r['strategy'] in MICRO])
            for name,predicate in [('n16',lambda r:r['source_total']==16),('n_lt16',lambda r:r['source_total']<16)]},
        micro_fallback_boundaries=aggregate_boundaries(boundary),
        cautions=['32 independent clusters, not 512 independent cases; N16 subgroup has only four clusters',
            '512-case mixture has seven stress families and one random family; report random separately',
            'Source-count and boundary categories are posthoc diagnostics; never supplied to policy',
            'Macro and micro training throughput differs (816 vs 48 PPO episodes), so architecture and optimization exposure are confounded',
            'All-clear does not mean zero failed optical clears; count failures and complete tails',
            'No development result is independent confirmation; no new checkpoint selection from this panel'])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=OUT.relative_to(ROOT).as_posix(),validation=review['validation'],
        fallback_groups={group:{method:{k:v for k,v in details.items() if k!='examples'} for method,details in arms.items()}
                         for group,arms in review['micro_fallback_boundaries'].items()}),ensure_ascii=False),flush=True)


if __name__=='__main__': main()

"""Only archived real R15 prefixes and public geometry; no policy/case run."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
COMMIT='89d1399b48cdc1c6a7f486b1cccb0cafc55287c2'
LABEL='compact_opportunistic_bounded'
BASE='results/q4_opportunistic_discovery/'
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.audit_q4_joint_continuation import wire_prefix
from planning.q4_directional_cover import certify_directional_cover,verify_directional_cover_certificate

OPTIONS=dict(arena_radius=1800.,reception_radius=1000.,range_margin_m=1e-5,
             orientation_margin_m=1e-7,max_depth=14,max_cells=20000,include_leaves=True)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def git(path):return subprocess.check_output(['git','show',f'{COMMIT}:{path}'],cwd=ROOT)
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def skip_value(text,at):
    # Lexical subtree skipping copied from the archived R13 observation reader;
    # evaluation/row values are never JSON decoded or consulted.
    if text[at] not in '[{"':
        while at<len(text) and text[at] not in ',}] \t\r\n':at+=1
        return at
    stack=[];quoted=text[at]=='"';escaped=False
    if not quoted:stack.append(text[at])
    at+=1
    while at<len(text):
        c=text[at];at+=1
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':
                quoted=False
                if not stack:return at
        elif c=='"':quoted=True
        elif c in '[{':stack.append(c)
        elif c in ']}':
            require(stack and (stack.pop(),c) in (('[',']'),('{','}')),'Invalid skipped JSON structure')
            if not stack:return at
    raise ValueError('Unterminated skipped JSON value')


def observation_record(data):
    text=gzip.decompress(data).decode('utf-8');decoder=json.JSONDecoder();at=len(text)-len(text.lstrip())
    require(text[at]=='{','Record must be object');at+=1;out={};keys=set()
    while True:
        while text[at].isspace() or text[at]==',':at+=1
        if text[at]=='}':break
        key,at=decoder.raw_decode(text,at)
        require(key not in keys,'Repeated top-level field');keys.add(key)
        while text[at].isspace():at+=1
        require(text[at]==':','Invalid record field');at+=1
        while text[at].isspace():at+=1
        if key in {'summary','history','spec'}:out[key],at=decoder.raw_decode(text,at)
        else:at=skip_value(text,at)
    require(set(out)=={'summary','history','spec'},'Incomplete observation record')
    return out


def old_auditor():
    path='experiments/audit_q4_opportunistic_discovery.py';data=git(path)
    namespace={'__name__':'archived_r15_prefix_audit','__file__':str(ROOT/path)}
    exec(compile(data,str(ROOT/path),'exec'),namespace)
    identity={path:sha(data)}
    for name,expected in namespace['SOURCE_CONTRACT'].items():
        actual=sha(git(name));require(actual==expected,'Archived source identity differs: '+name)
        identity[name]=actual
        # The R15 production is never imported. Every reused parent is exactly
        # the archived byte contract, as well as the blob checked above.
        if name!='src/strategies/q4_opportunistic_discovery.py':
            require(sha((ROOT/name).read_bytes())==expected,'Current parent dependency differs: '+name)
    # Explicit archival adapter: verification above replaces only local-file
    # lookup for the retired production path; no audit predicate is bypassed.
    namespace['verify_source_contract']=lambda:None
    return namespace['audit_opportunistic_discovery_prefix'],identity


def snapshot(h,end):
    known=set();cleared=set()
    for a in h[:end]:
        if a['action']=='measure' and a['result'] in {'direction','near'}:known.add(a['channel'])
        elif a['action']=='clear' and a['result']=='success':known.add(a['channel']);cleared.add(a['channel'])
    return known,cleared


def candidate_prefix(record):
    h,before=wire_prefix(record);params=record['summary']['strategy_parameters']
    started=[e for e in params['opportunistic_discovery_log'] if e['batch_started']]
    if len(started)<2:return None,'fewer_than_two_actual_batches'
    require(len(started)==2,'Unexpected third batch')
    if any(e['status']!='completed' for e in started):return None,'second_or_first_batch_not_complete'
    e=started[1];end=e['end_actual_action_count'];known,cleared=snapshot(h,end)
    if len(known)>=16:return None,'discovery_cap_already_reached'
    points=list(map(tuple,record['summary']['coverage_points']))
    require(len(points)==22,'Not original 22-station profile')
    visited=[];cover_blocks=[];j=0
    while j<end:
        if h[j]['phase']!='coverage':j+=1;continue
        p=tuple(h[j]['position']);start=j
        while j<end and h[j]['phase']=='coverage' and tuple(h[j]['position'])==p:j+=1
        require(p not in visited,'Repeated physical cover block')
        visited.append(p);cover_blocks.append(dict(position=list(p),start=start,end=j))
    require(visited==points[:len(visited)],'Actual cover blocks are not fixed chain prefix')
    require(e['coverage_points_visited']==len(visited),'Parent complete-cover count differs')
    inner=[i for i in range(len(visited),22) if abs(math.hypot(*points[i])-970.)<1e-6]
    if not inner:return None,'no_unvisited_inner_station'
    unknown=sorted(set(range(1,21))-known)
    evidence=[];pool=[];paid=0.
    for batch in started:
        a,b=batch['after_actual_action_count'],batch['end_actual_action_count']
        actual=h[a:b];p=tuple(batch['current_position']);old_known,_=snapshot(h,a)
        previous={x['channel'] for x in h[:a] if x['action']=='measure' and tuple(x['position'])==p}
        fresh=[c for c in range(1,21) if c not in old_known and c not in previous]
        tuned=before[a][1]
        if tuned in fresh:fresh.remove(tuned);fresh.insert(0,tuned)
        require([x['channel'] for x in actual]==fresh,'Incomplete frozen unknown-channel batch')
        require(all(x['action']=='measure' and x['phase']=='post_clear_discovery' and tuple(x['position'])==p for x in actual),'Nonphysical supplemental block')
        require(tuple(h[a-1]['position'])==p and h[a-1]['action']=='clear' and h[a-1]['result']=='success','Batch is not actual successful clear endpoint')
        paid+=before[b][2]-before[a][2]
    for p in visited+[tuple(x['current_position']) for x in started]:
        if p in pool:continue
        ids={}
        for c in unknown:
            found=[i for i,a in enumerate(h[:end]) if a['action']=='measure' and a['channel']==c and tuple(a['position'])==p]
            require(found and all(h[i]['result']=='no_signal' for i in found),'Unpaid or nonnegative point in unknown-channel pool')
            ids[str(c)]=found
        pool.append(p);evidence.append(dict(position=list(p),negative_action_indices=ids))
    return dict(after_actual_action_count=end,second_batch_event_id=e['id'],current_position=list(before[end][0]),
        current_channel=before[end][1],known_channels=sorted(known),cleared_channels=sorted(cleared),unknown_channels=unknown,
        visited_count=len(visited),cover_blocks=cover_blocks,batches=[dict(id=b['id'],position=b['current_position'],
        start=b['after_actual_action_count'],end=b['end_actual_action_count'],actual_cost_s=b['actual_cost_s']) for b in started],
        paid_supplemental_fee_s=paid,public_pool_evidence=evidence,remaining_inner_indices=inner,
        original_points=[list(p) for p in points]),None


def separation(points,witness):
    x=witness['source'];normal=witness['normal']
    require(math.hypot(*x)<=1800. and abs(math.hypot(*normal)-1.)<1e-10,'Invalid witness')
    rows=[]
    for p in points:
        distance=math.dist(p,x);projection=sum(normal[j]*(p[j]-x[j]) for j in (0,1))
        require(distance>1000.+1e-5 or projection < -1e-5,'Non-strict separating witness')
        rows.append(dict(point=list(p),distance_m=distance,projection_m=projection))
    return dict(passed=True,per_station=rows,witness_is_public_geometry_not_case_truth=True)


def length(start,points):return math.fsum(math.dist(a,b) for a,b in zip([start]+points,points))


def main():
    require(not (HERE/'results.json').exists(),'Preserve prior diagnostic')
    began=time.perf_counter();audit,identities=old_auditor()
    mechanism=json.loads(git('research/q4_opportunistic_discovery/development-mechanism.json'))
    files=sorted(BASE+folder+'/records/'+LABEL+'-'+str(seed)+'.json.gz'
        for folder,seeds in [('development',range(624001,624025)),('development-stress',range(624031,624045))] for seed in seeds)
    dependencies={name:sha((ROOT/name).read_bytes()) for name in ('src/planning/q4_directional_cover.py',
        'src/localization/__init__.py','src/geometry/__init__.py','experiments/audit_q4_joint_continuation.py')}
    freeze=dict(archived_commit=COMMIT,source_identity=identities,dependencies=dependencies,
        protocol_sha256=sha((HERE/'PROTOCOL.md').read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
        sampling_rule='Lexicographic full relative record path; second complete batch; at most30 eligible prefixes; every remaining inner index ascending',
        record_files=files,certificate_options=OPTIONS,python=sys.version,unknown_is_not_counterexample=True)
    if (HERE/'freeze.json').exists():
        original=json.loads((HERE/'freeze.json').read_bytes())
        require(all(original[k]==v for k,v in freeze.items() if k!='script_sha256'),'Resumption changed input/budget contract')
        freeze['original_freeze_sha256']=sha((HERE/'freeze.json').read_bytes())
        freeze['repair']='Legacy scheduling row.successful view is derived from observed summary claim; no row/evaluation decode. Existing exact-input proofs reused.'
        write(HERE/'resume-freeze.json',freeze)
    else:write(HERE/'freeze.json',freeze)
    records=[];prefixes=[];candidates=[];counts=collections.Counter();proof_dir=HERE/'proofs';proof_dir.mkdir(exist_ok=True)
    for name in files:
        data=git(name);digest=sha(data)
        require(mechanism['input_sha256'][name]==digest,'Original gzip hash differs')
        record=observation_record(data)
        require(record['spec']==dict(entrypoint='strategies.q4_opportunistic_discovery:run_q4_opportunistic_discovery',
            kwargs=dict(config='after_clear_bounded',max_expansions=200)),'Unreviewed record specification')
        # The legacy scheduling check asks only whether success is claimed. Use
        # the observed summary claim, then let that check demand 16 real clears
        # after a cap16 declaration. This is not original-row metadata auditing.
        audit_view=dict(record,row={'successful':record['summary'].get('completion_certified_under_model') is True})
        original_audit=audit(audit_view)
        require(original_audit['passed'] and original_audit['inherited_r8']['passed'] and original_audit['inherited_r12']['passed'],'R15 prefix replay did not pass')
        prefix,why=candidate_prefix(record)
        metadata=dict(record=name,input_sha256=digest,prefix_audit=original_audit,exclusion=why)
        records.append(metadata)
        if why:counts[why]+=1;continue
        if len(prefixes)>=30:metadata['exclusion']='predeclared_prefix_cap';counts['predeclared_prefix_cap']+=1;continue
        prefix.update(record=name,input_sha256=digest,prefix_id=len(prefixes));prefixes.append(prefix)
        points=list(map(tuple,prefix['original_points']));v=prefix['visited_count'];current=tuple(prefix['current_position'])
        pool=[tuple(p['position']) for p in prefix['public_pool_evidence']];tail=points[v:]
        for index in prefix['remaining_inner_indices']:
            retained=[p for j,p in enumerate(points) if j>=v and j!=index]
            geometry=pool+retained;filename=f'candidate-{len(candidates):03d}.json.gz'
            cached=(proof_dir/filename).exists()
            if cached:
                proof=json.loads(gzip.decompress((proof_dir/filename).read_bytes()))
                require(proof['stations']==[list(p) for p in sorted(set(geometry))],'Cached proof geometry changed')
                for key,value in OPTIONS.items():
                    if key!='include_leaves':require(proof[key]==value,'Cached proof budget changed')
            else:proof=certify_directional_cover(geometry,**OPTIONS)
            item=dict(candidate_id=len(candidates),prefix_id=prefix['prefix_id'],record=name,input_sha256=digest,
                removed_original_index=index,removed_point=list(points[index]),geometry=[list(p) for p in geometry],
                status=proof['status'],reason=proof['reason'],visited_cells=proof['visited_cells'],runtime_s=proof['runtime_s'],reused_first_attempt_proof=cached)
            if proof['passed']:item['independent_verification']=verify_directional_cover_certificate(geometry,proof)
            elif proof['status']=='counterexample':item['independent_verification']=separation(geometry,proof['counterexample'])
            else:item['independent_verification']=None
            saved=(length(current,tail)-length(current,retained))/5.
            item.update(fixed_chain_path_saved_s=saved,current_unknown_scan_fee_upper_s=6.*len(prefix['unknown_channels']),
                all_uncleared_station_scan_fee_upper_s=6.*(20-len(prefix['cleared_channels'])),
                paid_supplemental_fee_s=prefix['paid_supplemental_fee_s'],
                optimistic_unknown_path_net_s=saved+6.*len(prefix['unknown_channels'])-prefix['paid_supplemental_fee_s'])
            compressed=(proof_dir/filename).read_bytes() if cached else gzip.compress(canonical(proof),mtime=0)
            if not cached:
                with (proof_dir/filename).open('xb') as f:f.write(compressed)
            item.update(proof_file='proofs/'+filename,proof_sha256=sha(compressed));candidates.append(item)
        print(json.dumps(dict(prefix=len(prefixes),record=name,candidates=len(candidates),statuses=dict(collections.Counter(c['status'] for c in candidates)))),flush=True)
    certified=[c for c in candidates if c['status']=='certified']
    summary=dict(records=len(records),all_prefix_audits_passed=True,eligible_prefixes=len(prefixes),candidates=len(candidates),
        statuses=dict(collections.Counter(c['status'] for c in candidates)),exclusions=dict(counts),
        certified_distinct_prefixes=len({c['prefix_id'] for c in certified}),
        maximum_certified_path_plus_unknown_fee_upper_s=max((c['fixed_chain_path_saved_s']+c['current_unknown_scan_fee_upper_s'] for c in certified),default=None),
        maximum_certified_optimistic_net_s=max((c['optimistic_unknown_path_net_s'] for c in certified),default=None),
        maximum_any_path_plus_unknown_fee_upper_s=max((c['fixed_chain_path_saved_s']+c['current_unknown_scan_fee_upper_s'] for c in candidates),default=None),
        paid_two_batch_fee_s=[p['paid_supplemental_fee_s'] for p in prefixes],wall_s=time.perf_counter()-began,
        scope='Pure geometry with paid real negative point pool. Conditional retained future scan obligations; no strategy/scenario/evaluation/truth decode. No counterfactual T or performance gain.')
    write(HERE/'results.json',dict(summary=summary,records=records,prefixes=prefixes,candidates=candidates))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

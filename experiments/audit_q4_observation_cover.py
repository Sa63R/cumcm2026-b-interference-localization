"""Independent fixed 28/31-station identity, continuous cover and R12 safety.

No new policy/helper imports. Coordinates and route IDs are reconstructed
arithmetically; the old generic leaf certifier and its independent verifier
rebuild the new cover. The original actual history and new geometry are never
normalized to the old 22 stations. Only the inherited resolver's entry spec is
adapted after verifying the reviewed source contract.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from experiments.audit_q4_cover import audit_record
from experiments.audit_q4_joint_continuation import audit_joint_continuation_prefix
from experiments.audit_q4_clear_before_probe import audit_clear_before_probe_prefix, require
from experiments.audit_q4_range import audit_range_prefix
from experiments.audit_q4_scheduling import audit_scheduling_prefix
from planning.q4_directional_cover import certify_directional_cover, verify_directional_cover_certificate

ROOT=Path(__file__).resolve().parents[1]
ENTRY='strategies.q4_observation_cover:run_q4_observation_cover'
CONFIGS={'ring_28':(9,1920.),'ring_31':(10,1900.)}
BUDGET=dict(max_depth=16,max_cells=200000,range_margin_m=1e-5,orientation_margin_m=1e-7)
# Filled only after all 49 inherited files are byte-matched to qualified R12
# 81aa6e1a and both constructor-only additions finish independent review.
SOURCE_CONTRACT={'src/geometry/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/geometry/__init__.py': 'ab863186eed111790cf712c0cd19c741841087144a8fd9ecd37a3c10d33e46fa',
 'src/localization/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/localization/__init__.py': 'cda7e2f2a45faa60cb7dcef631e54cd384e3f367db96f23f415fc7eb666bb98c',
 'src/localization/omni.py': '35a999a1533a7c557315f719e2dcea91ffd3e1ec695636b94d374d4aaddc1e3b',
 'src/planning/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/planning/__init__.py': '070acfaf1e3aa79d2d4c9c8616f733b512e264729118f720929783a18b227027',
 'src/planning/chain_route.py': '0502c7fe525bc64578bda3ecbe5fcfe6f6408a0ad728e3e670223d6d215f29ef',
 'src/planning/coverage.py': '4c097ba0d009d0954a96089fb867db5bb4d00020d2cb0b408d8d4c297b427000',
 'src/planning/directional_probe_pair.py': '635d46a7a9571ab8f4a9f9f066b0ad5da0d5febcaf297c3520495bff55926f06',
 'src/planning/joint_visibility_region.py': 'c2df945c3a3dc295827fd7afbbb78793b43184cf61e1d1070f47020293953d7a',
 'src/planning/observation_cover_route.py': '6086c5c8ab7c4136d35b8c4560b70c92ee5ffff5018ec8bc2c7ad0490e377c75',
 'src/planning/positive_hull_probe.py': 'b03d5906660ee7565f19acaf8a7f86706957818977e526288bd845c40d9ffb5f',
 'src/planning/q4_directional_cover.py': 'e863b3f0fb83e1deba8d1a24c4b929b8fe1caca9927f8b99e514b1deea6da5c1',
 'src/planning/routing.py': 'e9cd4d242de519247a1fd7320add95c0c3d603f21a77899bb88d296d13ae4684',
 'src/practice_control/__init__.py': '45694c74e155e7267239f23656085ae352a59ce7cb5968b0dac6a36abcc9f62d',
 'src/practice_control/__main__.py': '78cb1245f7f355e1e4e7f855314211848687da00c3f4923d57ea8c1441276c46',
 'src/practice_control/branch_worker.py': '6e376f01d02606b251404038fe08c73e732297081d1e322656f24926a38c079b',
 'src/practice_control/bridge.py': 'fc28f4a2f96f335bcac64170b9c0a96cbf30f5e50d81a0bd679daacde4714779',
 'src/practice_control/runner.py': '15c28002cb80b899f7c19b6af2326fd687d0ae7131c05b910bed9ad83f574440',
 'src/practice_control/runtime.py': 'a9e62e85d19fe9081bf5085871715ef809ce9c25c79c58c35358b68b83c43865',
 'src/simulation/__init__.py': '2e01c860eb9cd4ea7fbab3cda9f13eaa4a454bef0ac17541301751dec0925cb6',
 'src/simulation/cases.py': '4d5588d9c11ccda5f251a819b292d9d69f1d5f5f9580be20534aa3bb9b6eec39',
 'src/simulation/engine.py': '3ef36f508f773564baed47569e014309cfb1cbcebb1ba1ad268a4b0f3e125622',
 'src/simulation/q3_branch.py': 'd16b7adf3c317f5c21ca3880916a1736884f396d7b7e964b71b8d8ce89ffe790',
 'src/simulator_client/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/simulator_client/__init__.py': 'aa5f551dfde62358958cc6d56c7ce986f3ce7937883c4edbb64cf8fb2c7aeee9',
 'src/simulator_client/__main__.py': 'd96fddf1ef8a4a824c8c85cba90e8e3b9ffd38b7fc97f21cdcd630875cd146f3',
 'src/simulator_client/client.py': '441230d1f2bb231a7a64beaa119306d89b4535ffaf1f4393c8be9c4d08a81577',
 'src/simulator_client/errors.py': 'c1ad261e6e6a99a691c08fec3f4b331036386542c968b45452881c522dce7723',
 'src/simulator_client/rules.py': '9bd7ca6df718aa42a414a94a08f981b519bfa70dc9ceeccb4f1a194aefcc83af',
 'src/simulator_client/state.py': '7a64a2a871532f76613000994bd850e86148a28f72f415d017fa56553815bb42',
 'src/strategies/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/strategies/__init__.py': '7eec10fe0201db03e7fc867e474dadf5a9a19cd1d3985bb2e22cdca88f9873a1',
 'src/strategies/efficient.py': '2349996a10685d62c441f2ebf8ff176d18f37ced660eb260e4daf38cf6b1f37a',
 'src/strategies/q3_belief.py': '35a5222a2c2cd89857f3d8c9d00a12217fe2667a20a2586dc4a1e28295ff465e',
 'src/strategies/q4_clear_before_probe.py': '641e8b0e4e6cce7b6445e88117d08ac23bd073487dfb46b87e903330f678ac69',
 'src/strategies/q4_cover_search.py': '2bab5ee2f935967670bb7c1107b99a7bba20208372e15aab328a67b15d651404',
 'src/strategies/q4_joint_continuation.py': '277480e9c22b0bd983a2971ad04fc97f8ff9e3232ed8198445b9d04dc95e024e',
 'src/strategies/q4_joint_visibility.py': 'eb922aae8052b48c9d54cd4b269a088498a3feeddbae254344b6ed73e31ff0e6',
 'src/strategies/q4_observation_cover.py': 'eaad648b732f3f57a38d54fbd26b9485f27b4dd507e11355fbae86d65cb3b99c',
 'src/strategies/q4_optical_cover.py': '87f8805167a91c364856d3da0f57ea032f4b3ac23b9d9c28c18c0a642136bfa6',
 'src/strategies/q4_r2_scheduling.py': '48d0f20ed0a073d26f177078f54dd1b40a532038aa74bb0de098c81f84509a44',
 'src/strategies/q4_range_pruning.py': 'c8bd010f186ccd991ffa3cb6727ff1d2cc038c2955bc29f27bb9ab851c37a1d2',
 'src/strategies/q4_range_scheduling.py': 'ad238a4e9c75537223d83ba4b180dc2d316ed71b487497f0f0e5a9bc2d72d76d',
 'src/strategies/q4_state_search.py': '8d105d02f3cb97ef99a1ccef98b63e5510da3c9c5000f77ee6401fbb39b9de59',
 'src/strategies/rollout.py': 'f2c7afa93373735a576e364515483e43ab01e8c2ea5e119ddb67f0dfda23838d',
 'src/strategies/search.py': '3c30ea448db217b2429d89c69d7db1f8e7fafc14c334043c6c1813ba94c44de5',
 'src/workflow/__init__.py': '3755f953a7f53ae76381142ab9e08347340462faf3bcb4694b4f76156c734d8c',
 'src/workflow/__main__.py': 'c353609048133fb2d11f9c774263dbb374de42910c816e062bddaa5166120ad6',
 'src/workflow/evidence.py': 'a4c5f37fe120c58be082fd3708020dd844140aceaed01835d8f1c52132ebca80'}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def verify_source_contract(root=ROOT):
    require(bool(SOURCE_CONTRACT),'Missing reviewed observation-cover source contract')
    for name,expected in SOURCE_CONTRACT.items():
        require(sha((root/name).read_bytes())==expected,'Reviewed observation-cover source differs: '+name)


def number(actual,expected,message):
    require(isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual)
            and abs(actual-expected)<=2e-7,message)


@lru_cache(maxsize=2)
def _geometry_bytes(config):
    """Immutable cache: callers receive freshly decoded reports each time."""
    require(isinstance(config,str) and config in CONFIGS,'Unreviewed geometry config')
    k,rho=CONFIGS[config]
    native=[(0.,0.)]
    for count,radius in ((k,999.),(2*k,rho)):
        native.extend((radius*math.cos(2*math.pi*i/count),radius*math.sin(2*math.pi*i/count)) for i in range(count))
    ids=[0]+list(range(1,k+1))+[k+1+((2*k-2-j)%(2*k)) for j in range(2*k)]
    require(sorted(ids)==list(range(3*k+1)),'Independent route partition invalid')
    points=[native[i] for i in ids]
    full=certify_directional_cover(points,**BUDGET)
    require(full.get('passed') is True,'Fixed new geometry has no complete leaf certificate')
    verification=verify_directional_cover_certificate(points,full)
    require(verification.get('passed') is True,'Independent full-leaf verification failed')
    length=math.fsum(math.dist(a,b) for a,b in zip(points,points[1:]))
    formula=rho+2*(k-1)*999.*math.sin(math.pi/k)+2*(2*k-1)*rho*math.sin(math.pi/(2*k))
    require(abs(length-formula)<1e-7,'Independent route formula mismatch')
    leaf_sha=sha(encoded(full['leaves']))
    certificate={key:value for key,value in full.items() if key not in {'leaves','runtime_s'}}
    proposal=dict(inner_count=k,outer_count=2*k,inner_radius_m=999.,outer_radius_m=rho,
                  inner_phase_deg=0.,outer_phase_deg=0.)
    certificate.update(profile='observation_'+config,route_length_m=length,
        route_kind='fixed_angle_zero_inner_ccw_outer_cw',full_leaf_certificate_sha256=leaf_sha,
        independent_leaf_verification=verification,proposal=proposal)
    step=math.pi/k; heading=math.acos(1800./rho)
    range_angle=math.acos((rho*rho+1800.**2-1000.**2)/(2*rho*1800.))
    alpha=min(heading,range_angle); ratio=2*alpha/step
    boundary=dict(angular_step_deg=math.degrees(step),halfwidth_deg=math.degrees(alpha),
        heading_halfwidth_deg=math.degrees(heading),range_halfwidth_deg=math.degrees(range_angle),
        receiving_arc_over_step=ratio,minimum_receiving_outer_stations=math.floor(ratio),
        angle_fraction_at_least_two=min(1.,max(0.,ratio-1.)),
        heading_two_station_threshold_m=1800./math.cos(step),
        range_at_one_step_m=math.sqrt(rho*rho+1800.**2-2*rho*1800.*math.cos(step)),
        min_source_station_distance_m=rho-1800.)
    metadata=dict(config=config,inner_count=k,outer_count=2*k,inner_radius_m=999.,outer_radius_m=rho,
        inner_phase_deg=0.,outer_phase_deg=0.,
        native_station_route_ids=ids,route_length_m=length,route_closed_form_m=formula,pure_movement_s=length/5.,
        station_sha256=full['station_sha256'],full_leaf_certificate_sha256=leaf_sha,boundary_radial_outward=boundary,
        route_scope='Fixed origin then angle-zero inner CCW and outer CW from index 2k-2; no shortest-route claim',
        boundary_scope='Only source radius 1800m, exactly radial-outward emission normal, reception radius 1000m; neither two receivers for all sources/orientations nor localization/task-time guarantee')
    return encoded(dict(points=points,certificate=certificate,metadata=metadata,verification=verification))


def public_geometry(config):
    # Hashability must not be allowed to turn True into a numeric config.
    require(isinstance(config,str) and config in CONFIGS,'Unreviewed geometry config')
    return json.loads(_geometry_bytes(config))


def audit_geometry(summary,config):
    independent=public_geometry(config)
    expected=independent['metadata'];params=summary['strategy_parameters']
    require(summary['coverage_points']==independent['points']
            and type(summary['coverage_points_total']) is int
            and summary['coverage_points_total']==1+3*CONFIGS[config][0],
            'Changed fixed coordinates, order or coverage count')
    require(params['observation_cover_config']==config
            and params['q4_compact_profile']=='observation_'+config,
            'New geometry mislabeled as old compact_22 or another layout')
    require(params['compact_schedule']=='joint','Inherited source schedule changed')
    certificate=params['directional_cover_certificate']
    require(set(certificate)==set(independent['certificate'])|{'runtime_s'},'Changed new certificate field set')
    # Newly recomputed runtime is allowed to differ; all geometry and proof
    # witnesses, including station coordinates and leaf hash, must agree.
    for key,value in independent['certificate'].items():
        if key=='route_length_m':number(certificate[key],value,'New certificate route cost mismatch')
        else:require(certificate[key]==value,'New full-cover evidence differs: '+key)
    for runtime in (certificate['runtime_s'],params['observation_cover_setup_runtime_s']):
        require(isinstance(runtime,(int,float)) and not isinstance(runtime,bool)
                and math.isfinite(runtime) and runtime>=0,'Nonfinite/negative certificate/setup runtime')
    metadata=params['observation_cover_route']
    require(set(metadata)==set(expected),'Changed route metadata field set')
    for key,value in expected.items():
        if key=='boundary_radial_outward':
            require(set(metadata[key])==set(value),'Changed boundary formula fields')
            for field,metric in value.items():
                if type(metric) is int:
                    require(type(metadata[key][field]) is int and metadata[key][field]==metric,'False minimum boundary receiver count')
                else:number(metadata[key][field],metric,'False radial-only boundary geometry: '+field)
        elif isinstance(value,float):number(metadata[key],value,'Route quantity mismatch: '+key)
        else:
            require(metadata[key]==value and (type(value) is not int or type(metadata[key]) is int),
                    'Fixed route identity mismatch: '+key)
    require(all(type(i) is int for i in metadata['native_station_route_ids']),'Noninteger route indices')
    return dict(passed=True,config=config,stations=len(independent['points']),
                fixed_route_length_m=expected['route_length_m'],leaf_verification=independent['verification'],
                station_sha256=expected['station_sha256'],leaf_sha256=expected['full_leaf_certificate_sha256'],
                boundary_scope='Radius1800, precisely radial outward emission, R1000 only; not universal double coverage')


def audit_observation_cover_prefix(record):
    verify_source_contract()
    spec=record['spec']; config=spec.get('kwargs',{}).get('config')
    require(isinstance(config,str) and config in CONFIGS and spec=={
        'entrypoint':ENTRY,'kwargs':{'config':config,'max_expansions':200}},'Unexpected observation-cover spec')
    geometry=audit_geometry(record['summary'],config)
    require(record['row']['strategy']=='compact_'+config,'Wrong compact label bypasses full-cover audit')
    # Only the entry spec changes. All actual positions, measured P/N, source
    # regions, 28/31 coverage obligations, new certificate and failures remain.
    inherited=dict(record,spec={'entrypoint':'strategies.q4_joint_continuation:run_q4_joint_continuation',
                                'kwargs':{'config':'after_active_miss_optical','max_expansions':200}})
    r12=audit_joint_continuation_prefix(inherited)
    r8=audit_clear_before_probe_prefix(inherited)
    params=record['summary']['strategy_parameters']
    rg=audit_range_prefix(record) if params.get('range_skipped_scans') else {'passed':True}
    accepted=dict(record,history=[a for a in record['history']
        if a['action'] not in {'/measure','/clear'} or a['response'].get('accepted') is True])
    scheduling=audit_scheduling_prefix(accepted)
    require(all(x.get('passed') is True for x in (r12,r8,rg,scheduling)),'Inherited audit failed')
    return dict(passed=True,geometry=geometry,r12=r12,r8=r8,range=rg,scheduling=scheduling,
                source_contract=dict(SOURCE_CONTRACT),
                adaptation='Source-bound resolver spec only; true 28/31 coordinates and physical history unchanged')


def audit_full(record):
    verify_source_contract()
    generic=audit_record(record)
    require(generic.get('passed') is True,'Physical/new full-cover/LB audit failed: '+str(generic.get('errors')))
    prefix={k:record[k] for k in ('summary','history','row','spec')}
    return dict(passed=True,generic=generic,prefix=audit_observation_cover_prefix(prefix))

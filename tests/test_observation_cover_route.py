"""Public geometry tests: no source scenario, policy or hidden evaluation."""
import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from planning import observation_cover_route as route_module
from planning.q4_directional_cover import verify_directional_cover_certificate
from simulator_client.state import Position


@pytest.fixture(autouse=True)
def clear_private_cache():
    route_module._observation_cover_route.cache_clear()
    yield
    route_module._observation_cover_route.cache_clear()


@pytest.mark.parametrize("config,k,rho,ids", [
    ("ring_28",9,1920.,[0,1,2,3,4,5,6,7,8,9,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,27]),
    ("ring_31",10,1900.,[0,1,2,3,4,5,6,7,8,9,10,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,30]),
])
def test_exact_fixed_order_matches_public_geometry_and_full_leaf_proof(config,k,rho,ids):
    points,certificate,metadata=route_module.observation_cover_route(config)
    native=[Position(0.,0.)]
    for radius,count in ((999.,k),(rho,2*k)):
        for j in range(count):
            angle=2*math.pi*j/count
            native.append(Position(radius*math.cos(angle),radius*math.sin(angle)))
    assert points==tuple(native[i] for i in ids)
    assert len(points)==len(set(points))==1+3*k
    assert points[0]==Position(0.,0.) and points[1]==Position(999.,0.)
    assert metadata['native_station_route_ids']==ids
    proof_path=Path(__file__).resolve().parents[1]/f'research/q4_observation_cover/geometry-proofs/inner{k}-outer{2*k}-r999-{int(rho)}.json.gz'
    proof=json.loads(gzip.decompress(proof_path.read_bytes()))
    assert verify_directional_cover_certificate(points,proof)['passed']
    leaf_sha=hashlib.sha256(json.dumps(proof['leaves'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert certificate['full_leaf_certificate_sha256']==metadata['full_leaf_certificate_sha256']==leaf_sha
    assert certificate['station_sha256']==metadata['station_sha256']==proof['station_sha256']
    assert certificate['station_count']==len(points) and certificate['profile']=='observation_'+config
    assert certificate['arena_radius']==1800. and certificate['reception_radius']==1000.
    assert certificate['range_margin_m']==1e-5 and certificate['orientation_margin_m']==1e-7
    assert certificate['max_depth']==16 and certificate['max_cells']==200000
    assert certificate['independent_leaf_verification']['leaf_count']==len(proof['leaves'])
    assert 'leaves' not in certificate
    length=math.fsum(math.hypot(a.x-b.x,a.y-b.y) for a,b in zip(points,points[1:]))
    assert metadata['route_length_m']==length
    assert certificate['route_length_m']==length
    assert metadata['route_closed_form_m']==pytest.approx(length,rel=0.,abs=1e-7)
    assert metadata['pure_movement_s']==length/5
    boundary=metadata['boundary_radial_outward']
    assert boundary['minimum_receiving_outer_stations']==2
    assert boundary['angle_fraction_at_least_two']==1.
    assert boundary['range_at_one_step_m']<1000.
    assert rho*math.cos(math.pi/k)>1800.


def test_constructor_replays_full_certificate_and_does_not_call_route_heuristics(monkeypatch):
    calls=[]
    original=route_module.verify_directional_cover_certificate
    def recording(points,proof):
        calls.append((len(points),len(proof['leaves'])))
        return original(points,proof)
    monkeypatch.setattr(route_module,'verify_directional_cover_certificate',recording)
    def forbidden(*args,**kwargs):raise AssertionError('NN/2-opt is outside fixed route')
    monkeypatch.setattr('planning.coverage.nearest_order',forbidden)
    monkeypatch.setattr('planning.coverage.improve_open_route',forbidden)
    route_module.observation_cover_route('ring_28')
    route_module.observation_cover_route('ring_28')
    assert calls==[(28,1708)]


def test_inconclusive_certificate_cannot_be_cached_as_passed(monkeypatch):
    monkeypatch.setattr(route_module,'certify_directional_cover',lambda *a,**kw:dict(passed=False))
    with pytest.raises(ValueError,match='full successful'):
        route_module.observation_cover_route('ring_28')
    assert route_module._observation_cover_route.cache_info().currsize==0


def test_corrupt_complete_partition_is_rejected(monkeypatch):
    original=route_module.certify_directional_cover
    def corrupt(*args,**kwargs):
        full=original(*args,**kwargs)
        full['leaves'].pop()
        return full
    monkeypatch.setattr(route_module,'certify_directional_cover',corrupt)
    with pytest.raises(ValueError,match='missing region'):
        route_module.observation_cover_route('ring_31')


def test_cached_nested_metadata_and_certificate_are_isolated():
    points,certificate,metadata=route_module.observation_cover_route('ring_31')
    certificate['stations'][0][0]=123.
    certificate['proposal']['inner_count']=-1
    metadata['native_station_route_ids'].clear()
    metadata['boundary_radial_outward']['minimum_receiving_outer_stations']=999
    again,cert,meta=route_module.observation_cover_route('ring_31')
    assert again==points and len(meta['native_station_route_ids'])==31
    assert cert['proposal']['inner_count']==10
    assert meta['boundary_radial_outward']['minimum_receiving_outer_stations']==2
    assert cert['stations'][0][0]!=123.


@pytest.mark.parametrize('config',['ring_25','ring_28_1900','ring_31_1920','sector_1',None,28,True,[],{}])
def test_only_two_predeclared_candidates(config):
    with pytest.raises(ValueError):route_module.observation_cover_route(config)

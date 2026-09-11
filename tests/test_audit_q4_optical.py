import copy
import pytest
from experiments.audit_q4_optical import audit_optical_prefix
from localization import CandidateRegion
from strategies.q4_optical_cover import optical_cover_route


def fixture():
    region = CandidateRegion().observe((0., 0.), 0.)
    route, event = optical_cover_route(region.vertices, bearing_deg=0., start=(0., 0.))
    assert event['selected'] == 'rectangular_slabs'
    event.update(channel=1, after_actual_action_count=1)
    point = [route[0].x, route[0].y]
    history = [
        {'action':'/measure','channel':1,'position':[0.,0.],
         'response':{'accepted':True,'measure_result':'direction','svd_deg':0.}},
        {'action':'/clear','channel':1,'position':point,
         'response':{'accepted':True,'clear_result':'success'}}]
    reports = [
        {'action':'measure','channel':1,'position':[0.,0.],'result':'direction','bearing_deg':0.,'phase':'coverage'},
        {'action':'clear','channel':1,'position':point,'result':'success','phase':'guaranteed_clearance'}]
    return {'history':history,'summary':{'action_history':reports,'strategy_parameters':{
        'q4_optical_mode':'rectangular','optical_cover_log':[event]}}}


def test_valid_complete_geometric_certificate_and_successful_prefix():
    record = fixture()
    assert audit_optical_prefix(record) == {'passed':True,'applicable':True,'events':1,'optical_actions':1}


@pytest.mark.parametrize('change',['gap','bbox','columns','radius','vertices','bearing','start','position','missing','wire'])
def test_corrupted_geometry_or_actual_prefix_rejected(change):
    record = fixture()
    event = record['summary']['strategy_parameters']['optical_cover_log'][0]
    slab = event['certificate']['slabs'][0]
    if change == 'gap': slab['slab'][0] += 1
    elif change == 'bbox': slab['bounding_box'][1] -= 100
    elif change == 'columns': slab['columns'] = 1
    elif change == 'radius': event['certificate']['radius_m'] = 20.1
    elif change == 'vertices': event['observed_region_vertices'][0][0] += 1
    elif change == 'bearing': event['bearing_deg'] += .1
    elif change == 'start': event['start'][0] += 1
    elif change == 'position': record['history'][1]['position'][0] += 1
    elif change == 'missing': record['summary']['strategy_parameters']['optical_cover_log'] = []
    else: record['history'][0]['response']['accepted'] = False
    with pytest.raises(ValueError): audit_optical_prefix(record)


def test_legacy_replays_old_grid_and_rejects_wrong_position():
    from planning.coverage import clearance_grid
    record = fixture()
    event = record['summary']['strategy_parameters']['optical_cover_log'][0]
    grid = clearance_grid(event['observed_region_vertices'], bearing_deg=0., start=(0,0))
    event.update(selected='legacy',certificate=None,selected_count=len(grid))
    p = [grid[0].x,grid[0].y]
    record['history'][1]['position'] = p
    record['summary']['action_history'][1]['position'] = p
    assert audit_optical_prefix(record)['passed']


def test_not_applicable_to_unchanged_baseline():
    assert audit_optical_prefix({'summary':{}})['applicable'] is False

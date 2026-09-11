"""Constructed observation/wire archives only; no simulator or strategy runs."""

import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import zipfile

import pytest

from experiments.audit_q4_state import (ROOT, COMPONENTS, audit_directory, audit_record,
                                        digest, observation_audit, triangular_certificate_stations)
from localization import CandidateRegion
from planning.directional_probe_pair import choose_directional_probe_pair


def test_wire_accepts_publicly_allowed_all_directional_q4():
    from experiments.audit_q4_state import wire_audit
    builder=RecordBuilder(count=14)
    for source in builder.sources:
        source['orientation_deg']=0.0
    builder.clear_sources()
    assert wire_audit(builder.record())['cleared_total']==14


class RecordBuilder:
    """Explicit fixture physics, deliberately independent of local simulator."""
    def __init__(self, count=16, seed=42):
        self.sources = [{'channel': i+1, 'x': 100.+30*i, 'y': 0., 'reception_radius_m': 1000.,
                         'orientation_deg': 0. if i == 0 else None} for i in range(count)]
        self.truth = {'problem': 4, 'seed': seed, 'case_id': 'audit-synthetic-'+str(seed), 'sources': self.sources}
        self.history, self.reports = [], []
        self.position, self.tuned, self.us = (0., 0.), 1, 0
        self.parts, self.cleared = dict.fromkeys(COMPONENTS, 0), set()
        self.parameters = {}
        self.add('/enter')

    def add(self, action, q=None, channel=None, phase='', forced=None):
        response = {'accepted': True}
        if action in ('/measure', '/clear'):
            move = round(math.dist(self.position, q)/5*1e6)
            self.parts['movement_s'] += move
            self.us += move
            self.position = tuple(q)
            source = next((s for s in self.sources if s['channel'] == channel and channel not in self.cleared), None)
            distance = math.dist(q, (source['x'], source['y'])) if source else math.inf
            if action == '/measure':
                switch = int(self.tuned != channel)*1_000_000
                self.parts['switching_s'] += switch
                self.parts['detection_s'] += 5_000_000
                self.us += switch+5_000_000
                self.tuned = channel
                visible = source is not None and distance <= source['reception_radius_m']
                if visible and source['orientation_deg'] is not None:
                    angle = math.radians(source['orientation_deg'])
                    visible = (math.cos(angle)*(q[0]-source['x'])+math.sin(angle)*(q[1]-source['y'])) >= -1e-9
                result = forced or ('no_signal' if not visible else 'near' if distance <= 5 else 'direction')
                response['measure_result'] = result
                if result == 'direction':
                    response['svd_deg'] = math.degrees(math.atan2(source['y']-q[1], source['x']-q[0])) % 360
            else:
                success = distance <= 20
                self.parts['optical_s'] += 3_000_000
                self.parts['removal_s'] += int(success)*2_000_000
                self.us += 3_000_000+int(success)*2_000_000
                result = 'success' if success else 'no_target_in_range'
                response['clear_result'] = result
                if success:
                    self.cleared.add(channel)
            report = {'action': action[1:], 'channel': channel, 'position': list(q), 'phase': phase,
                      'result': result, 'virtual_time_s': self.us/1e6}
            if 'svd_deg' in response:
                report['bearing_deg'] = response['svd_deg']
            self.reports.append(report)
        response['virtual_time_s'] = self.us/1e6
        self.history.append({'index': len(self.history), 'action': action, 'channel': channel,
                             'position': dict(zip(('x', 'y'), self.position)), 'response': response})

    def clear_sources(self):
        for s in self.sources:
            if s['channel'] not in self.cleared:
                q = (s['x'], s['y'])
                self.add('/measure', q, s['channel'], 'active_localization')
                self.add('/clear', q, s['channel'], 'near_clear')

    def record(self, successful=True, label='triangular'):
        if self.history[-1]['action'] != '/exit':
            self.add('/exit')
        all_cleared = len(self.cleared) == len(self.sources)
        values = {'source_total': len(self.sources), 'cleared_total': len(self.cleared),
                  'measurement_count': sum(a['action']=='/measure' for a in self.history),
                  'failed_clear_count': sum(a['response'].get('clear_result')=='no_target_in_range' for a in self.history),
                  'action_count': len(self.history), 'all_cleared': all_cleared, 'virtual_time_s': self.us/1e6}
        parts = {k:v/1e6 for k,v in self.parts.items()}
        row = dict(values, **parts, seed=self.truth['seed'], case_id=self.truth['case_id'], problem=4,
                   stage='pilot', strategy=label, case_sha256=digest(self.truth), successful=successful,
                   completion_certified=successful, accepted_exit=True,
                   penalized_time_s=self.us/1e6 if successful else 360000., errors=[])
        summary = {'action_history': self.reports, 'strategy_parameters': self.parameters,
                   'completion_certified_under_model': successful,
                   'completion_reason': 'source_count_upper_bound_reached' if len(self.cleared)==16 else 'coverage_exhausted_and_all_detected_cleared'}
        return {'row': row, 'summary': summary, 'evaluation_phase': 'after_policy_termination',
                'evaluation': dict(values, kind='local_research_only', time_breakdown_s=parts, ground_truth=self.truth),
                'history': self.history, 'spec': {'entrypoint': 'synthetic:never_run', 'kwargs': {}}}


def full_record(count=16):
    builder = RecordBuilder(count)
    builder.clear_sources()
    if count < 16:
        stations, _ = triangular_certificate_stations()
        for channel in range(count+1, 21):
            for p in stations:
                builder.add('/measure', p, channel, 'coverage')
    return builder.record()


def test_actual_sixteen_clears_certify_without_coverage():
    result = audit_record(full_record())
    assert result['audit_passed'], result['errors']
    assert result['observations']['source_cap_actual_clears']
    assert result['observations']['terminal_certified']


def test_ten_clear_certificate_uses_each_channels_real_triangle_vertices():
    record = full_record(10)
    result = audit_record(record)
    assert result['audit_passed'], result['errors']
    assert result['observations']['triangle_station_count'] == 31
    assert all(a['missing_station_count']==0 for a in result['observations']['absence'])
    # Removing one actual station from just one channel defeats that channel's
    # sufficient certificate even if all other channels are fully scanned.
    broken = copy.deepcopy(record)
    index = next(i for i,a in enumerate(broken['history']) if a['action']=='/measure' and a['channel']==11)
    broken['history'].pop(index)
    broken['summary']['action_history'].pop(index-1)
    with pytest.raises(ValueError, match='actual Q4 cover'):
        observation_audit(broken)


def test_sixteen_detections_do_not_replace_sixteen_clears():
    builder = RecordBuilder()
    for source in builder.sources:
        builder.add('/measure', (source['x'],source['y']), source['channel'], 'coverage')
    record = builder.record()
    record['summary']['completion_reason'] = 'source_count_upper_bound_reached'
    assert not audit_record(record)['audit_passed']


def test_legal_failed_optical_search_is_costed_and_success_may_still_pass():
    builder = RecordBuilder()
    builder.add('/clear', (-1500,-1500), 1, 'guaranteed_clearance')
    builder.clear_sources()
    result = audit_record(builder.record())
    assert result['audit_passed'], result['errors']
    assert result['physical']['failed_clear_count']==1
    assert result['observations']['legal_failed_optical_attempts']==1


def test_claimed_certified_failure_is_rejected_even_when_later_all_cleared():
    builder = RecordBuilder()
    builder.add('/clear', (-1500,-1500), 1, 'certified_clear')
    builder.clear_sources()
    result = audit_record(builder.record())
    assert not result['audit_passed'] and any('Claimed safe clear' in e for e in result['errors'])


def test_directional_backside_no_signal_is_legal_and_does_not_clip_omni_halfplane():
    builder = RecordBuilder()
    builder.add('/measure', (0,0), 1, 'coverage')  # Source at (100,0), emitting east.
    builder.add('/measure', (500,0), 1, 'active_localization')
    builder.clear_sources()
    result = audit_record(builder.record())
    assert result['audit_passed'], result['errors']


@pytest.mark.parametrize('mutation', ['microseconds','component','truth','bearing','summary','feedback'])
def test_wire_or_identity_corruption_is_not_silently_accepted(mutation):
    builder = RecordBuilder()
    builder.add('/measure', (500,0), 1, 'coverage')
    builder.clear_sources()
    record = builder.record()
    if mutation=='microseconds': record['history'][1]['response']['virtual_time_s'] += 0.1
    if mutation=='component': record['row']['switching_s'] += 1
    if mutation=='truth': record['evaluation']['ground_truth']['sources'][0]['x'] += 0.5
    if mutation=='bearing': record['history'][1]['response']['svd_deg'] += 2
    if mutation=='summary': record['summary']['action_history'][0]['phase'] = 'near_clear'; record['summary']['action_history'][0]['position'][0] += 1
    if mutation=='feedback': record['history'][1]['response']['measure_result'] = 'no_signal'
    assert not audit_record(record)['audit_passed']


def test_short_failed_run_is_retained_with_full_penalty_and_without_completeness_claim():
    builder = RecordBuilder()
    builder.add('/measure', (0,0), 1, 'coverage')
    record = builder.record(False)
    assert audit_record(record)['audit_passed']
    record['row']['penalized_time_s'] = record['row']['virtual_time_s']
    assert not audit_record(record)['audit_passed']


def skipped_record():
    builder=RecordBuilder()
    source=builder.sources[0]
    builder.add('/measure',(source['x'],source['y']),1,'active_localization')
    builder.parameters['skipped_certified_scans']=[{'channel':1,'position':[0.,0.],
        'after_actual_action_count':1,'reason':'near','radius_m':None}]
    builder.add('/measure',(0.,0.),20,'coverage')
    builder.clear_sources()
    return builder.record()


def test_skipped_certified_known_scan_adds_no_unknown_coverage_credit():
    result=audit_record(skipped_record())
    assert result['audit_passed'], result['errors']
    assert len(result['observations']['certified_scans_skipped'])==1
    assert result['observations']['inferred_coverage_credits']==0


@pytest.mark.parametrize('mutation',['unknown','future','false_disk'])
def test_skipped_scan_needs_its_own_real_prefix_clear_certificate(mutation):
    record=skipped_record()
    event=record['summary']['strategy_parameters']['skipped_certified_scans'][0]
    if mutation=='unknown': event['channel']=20
    if mutation=='future': event['after_actual_action_count']=0
    if mutation=='false_disk': event.update(reason='enclosing_disk',radius_m=1.)
    assert not audit_record(record)['audit_passed']


def hull_record():
    builder = RecordBuilder()
    positives = [(1000.,-100.),(1000.,100.)]
    for p in positives: builder.add('/measure',p,1,'coverage')
    candidate = {'position':[1000.,0.], 'weights':[{'position':list(p),'weight':.5} for p in positives]}
    builder.parameters['positive_hull_log']=[{'channel':1,'index':0,'after_actual_action_count':2,
        'positive_stations':[list(p) for p in positives],'candidates':[candidate],'selected':0,'position':[1000.,0.]}]
    builder.add('/measure',(1000.,0.),1,'active_localization')
    builder.clear_sources()
    return builder.record()


def test_hull_witness_and_actual_next_action_pass():
    result = audit_record(hull_record())
    assert result['audit_passed'], result['errors']
    assert result['observations']['hull_events'][0]['executed']


@pytest.mark.parametrize('mutation',['unseen_anchor','weight','future_prefix','different_next_point'])
def test_hull_prefix_forgery_fails(mutation):
    record = hull_record()
    event = record['summary']['strategy_parameters']['positive_hull_log'][0]
    if mutation=='unseen_anchor': event['candidates'][0]['weights'][0]['position']=[0,0]
    if mutation=='weight': event['candidates'][0]['weights'][0]['weight']=.7
    if mutation=='future_prefix': event['after_actual_action_count']=3
    if mutation=='different_next_point': event['position']=[1000,1]
    assert not audit_record(record)['audit_passed']


def pair_record(first_silent):
    builder = RecordBuilder()
    # p=(200,0) lies on a directional boundary for source=(100,0).
    # The nearer chosen endpoint goes below that boundary and is silent.
    builder.sources[0]['orientation_deg'] = 90. if first_silent else 0.
    anchor=(200.,0.)
    builder.add('/measure',anchor,1,'coverage')
    region=CandidateRegion().observe(anchor,180.)
    pair, log=choose_directional_probe_pair(region,anchor,[anchor])
    assert pair is not None
    event={'channel':1,'index':0,'role':'first','after_actual_action_count':1,'pair':asdict(pair),**log}
    builder.parameters['directional_pair_log']=[event]
    builder.add('/measure',(pair.first.x,pair.first.y),1,'active_localization')
    if first_silent:
        assert builder.reports[-1]['result']=='no_signal'
        builder.parameters['directional_pair_log'].append({'channel':1,'index':1,'role':'second',
            'after_actual_action_count':2,'position':[pair.second.x,pair.second.y],'first':[pair.first.x,pair.first.y]})
        builder.add('/measure',(pair.second.x,pair.second.y),1,'active_localization')
    builder.clear_sources()
    return builder.record()


@pytest.mark.parametrize('silent',[False,True])
def test_pair_prefix_formula_and_conditional_second_probe(silent):
    result=audit_record(pair_record(silent))
    assert result['audit_passed'], result['errors']
    assert len(result['observations']['pair_events'])==1+silent


@pytest.mark.parametrize('mutation',['step','anchor','budget','omit_second','wrong_second'])
def test_pair_certificate_or_execution_forgery_fails(mutation):
    record=pair_record(True)
    events=record['summary']['strategy_parameters']['directional_pair_log']
    if mutation=='step': events[0]['pair']['step_m'] *= 2
    if mutation=='anchor': events[0]['pair']['anchor_bearing_deg'] += 1
    if mutation=='budget': events[0]['index']=5
    if mutation=='omit_second': events.pop()
    if mutation=='wrong_second': events[1]['position'][1] += 1
    assert not audit_record(record)['audit_passed']


def build_batch(directory):
    directory.mkdir()
    (directory/'records').mkdir()
    record=full_record()
    specs={name:record['spec'] for name in ('triangular','state')}
    source_names=['src/geometry/__init__.py','src/localization/__init__.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    manifest={'protocol':{'problem':4,'pilot':[42,42]},'stage':'pilot','seeds':[42],
              'specs':specs,'source_sha256':hashes}
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (directory/'freeze.json').write_text(json.dumps({'manifest_sha256':digest(manifest)}),encoding='utf-8')
    with zipfile.ZipFile(directory/'source.zip','w') as archive:
        for name in source_names: archive.write(ROOT/name,name)
    rows=[]
    for label in specs:
        value=copy.deepcopy(record)
        value['row']['strategy']=label
        rows.append(value['row'])
        with gzip.open(directory/'records'/f'{label}-42.json.gz','wt',encoding='utf-8') as stream:
            json.dump(value,stream)
    rows.sort(key=lambda r:(r['seed'],r['strategy']))
    (directory/'summary.json').write_text(json.dumps({'rows':rows}),encoding='utf-8')


def test_batch_pairs_archives_and_checks_frozen_source_bytes(tmp_path):
    directory=tmp_path/'batch'
    build_batch(directory)
    result=audit_directory(directory)
    assert result['all_passed'], result['global_errors']
    assert result['audits_passed']==2


def test_historical_strategy_archive_does_not_require_current_strategy_checkout(tmp_path):
    directory=tmp_path/'batch'
    build_batch(directory)
    historical_name='src/strategies/not_currently_checked_out.py'
    old_bytes=b'Historical frozen implementation only.'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    manifest['source_sha256'][historical_name]=hashlib.sha256(old_bytes).hexdigest()
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (directory/'freeze.json').write_text(json.dumps({'manifest_sha256':digest(manifest)}),encoding='utf-8')
    with zipfile.ZipFile(directory/'source.zip','a') as archive: archive.writestr(historical_name,old_bytes)
    result=audit_directory(directory)
    assert result['all_passed'], result['global_errors']
    assert result['current_source_changes_since_freeze']==[historical_name]


@pytest.mark.parametrize('mutation',['missing','extra','spec','pairing','summary','freeze','source'])
def test_batch_rejects_missing_duplicate_identity_or_freeze_mismatch(tmp_path,mutation):
    directory=tmp_path/'batch'
    build_batch(directory)
    path=directory/'records'/'state-42.json.gz'
    if mutation=='missing': path.unlink()
    elif mutation=='extra': (directory/'records'/'unexpected.json.gz').write_bytes(path.read_bytes())
    elif mutation in ('spec','pairing'):
        with gzip.open(path,'rt',encoding='utf-8') as stream: value=json.load(stream)
        if mutation=='spec': value['spec']['kwargs']['unexpected']=True
        else: value['row']['case_sha256']='bad'
        with gzip.open(path,'wt',encoding='utf-8') as stream: json.dump(value,stream)
    elif mutation=='summary': (directory/'summary.json').write_text('{"rows":[]}',encoding='utf-8')
    elif mutation=='freeze': (directory/'freeze.json').write_text('{"manifest_sha256":"bad"}',encoding='utf-8')
    else:
        with zipfile.ZipFile(directory/'source.zip','w') as archive: archive.writestr('src/geometry/__init__.py','changed')
    assert not audit_directory(directory)['all_passed']

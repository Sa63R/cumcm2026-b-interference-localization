"""Actual-prefix ownership and independent shared-route first-hit audit for R37.

Only the final predictor integration is new. The explicit copy of the pinned R12
replay below adds an ownership callback without changing any actual history or
canonical/auxiliary/clear/grid/continuation checks. Tests guard its exact delta.
The pinned continuous model is a reusable numerical assumption, not source truth.
"""
import hashlib
import inspect
import json
from pathlib import Path
from dataclasses import asdict

from experiments.audit_q4_joint_continuation import *
from experiments import audit_q4_joint_continuation as parent_audit
from experiments.audit_q4_clear_before_probe import audit_clear_before_probe_prefix
from experiments.audit_q4_range import audit_range_prefix
from experiments.audit_q4_scheduling import audit_scheduling_prefix
from planning import clearance_grid

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'strategies.q4_anchor_first_hit:run_q4_anchor_first_hit'
LABEL = 'compact_anchor_first_hit'
CONFIG = 'anchor_first_hit'
SOURCE_CONTRACT = {'experiments/audit_q4_clear_before_probe.py': '68c7ef07dca4cc9f7a60351332911c25054c5e0392220830637bfd89518e62c4',
 'experiments/audit_q4_joint_continuation.py': '5a04f00ee3796312b0ef0862fedf1bfefe1f9d57101e43eacbf11c7708db99e8',
 'experiments/audit_q4_range.py': '8995a087ec085cbef445ba455f68dc81d426c1e51b3406e81184c8888cdd3578',
 'experiments/audit_q4_scheduling.py': '836d82d30280d24b612ca96367ca7e9b33b0b1b76f441bbfbcf2d674606ff5be',
 'src/geometry/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
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
NEW_SOURCE_CONTRACT = {
    'src/planning/q4_conditional_belief.py': '8453de4fc80b94c5428eade327a20c601825808a4facc9515ddecbd6e36ba038',
    'src/planning/q4_anchor_first_hit.py': 'fb1d6a672e68517ce1785d79219d0853b3d47cfecd1caaa46e89327ef6488d37',
    'src/strategies/q4_anchor_first_hit.py': '9894dacd466864aaee0841d07209e0f3c4946899da4892a52de40055356e425e',
}

def verify_source_contract():
    require(len(NEW_SOURCE_CONTRACT) == 3, 'Anchor production identity has not been frozen')
    for name, expected in {**SOURCE_CONTRACT, **NEW_SOURCE_CONTRACT}.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,
                'Changed source contract: '+name)
    return len(SOURCE_CONTRACT)+len(NEW_SOURCE_CONTRACT)


def verify_replay_delta():
    original = inspect.getsource(parent_audit.audit_joint_continuation_prefix).strip()
    adapted = inspect.getsource(_audit_r12_owned).strip()
    adapted = adapted.replace('def _audit_r12_owned(record, ownership):', 'def audit_joint_continuation_prefix(record):', 1)
    adapted = adapted.replace('    owned = ownership(record, states, starts, ends, by_id, h, before)\n', '', 1)
    adapted = adapted.replace("    require(not used_probe_actions.intersection(owned), 'Probe has duplicate parent/anchor owners')\n", '', 1)
    adapted = adapted.replace("expected_probes-owned, 'Unlogged auxiliary probe measurement'", "expected_probes, 'Unlogged auxiliary probe measurement'", 1)
    adapted = adapted.replace('states[i][j] is not None or j in owned:', 'states[i][j] is not None:', 1)
    require(adapted == original, 'R12 replay differs beyond the reviewed ownership hooks')


def _audit_r12_owned(record, ownership):
    summary = record['summary']
    params = summary['strategy_parameters']
    config = params['joint_visibility_config']
    require(config == 'probe_optical' and params.get('joint_visibility_continuation_config') == 'after_active_miss_optical', 'Unknown continuation configuration')
    if 'spec' in record:
        require(record['spec']['entrypoint'] == 'strategies.q4_joint_continuation:run_q4_joint_continuation'
                and record['spec'].get('kwargs', {}).get('config', 'after_active_miss_optical') == 'after_active_miss_optical', 'Continuation differs from frozen spec')
    h, before = wire_prefix(record)
    epochs = params['joint_visibility_resolver_log']
    probes, grids = params['joint_visibility_probe_log'], params['joint_visibility_grid_log']
    terminal_clears = params.get('joint_visibility_terminal_clear_log', [])
    states, starts, ends, by_id = {}, {}, {}, {}
    counters = dict(epochs=len(epochs), refined_epochs=0, fallback_epochs=0, probe_decisions=len(probes),
        executed_probes=0, auxiliary_clear_attempts=0, auxiliary_clear_successes=0,
        optical_epochs=len(grids), optical_attempts=0, terminal_clear_events=len(terminal_clears),
        canonical_fallbacks=0, decision_wall_s=0.)
    continuations = params['joint_visibility_continuation_log']
    probe_budget = record.get('spec', {}).get('kwargs', {}).get('max_active_probes', 6)
    require(type(probe_budget) is int and 0 <= probe_budget <= 30, 'Bad continuation probe budget')
    require(params['joint_visibility_continuation_limits'] == dict(per_resolver=probe_budget,
        per_session=20*probe_budget, helper_constraint_work=65536), 'Changed deterministic continuation budget')
    require([event['id'] for event in continuations] == list(range(len(continuations))), 'Continuation IDs differ')
    continuation_map = {(event['resolver_id'], event['after_actual_action_count']): event for event in continuations}
    require(len(continuation_map) == len(continuations), 'Repeated continuation prefix')
    continuation_counts = {'session_calls': 0, 'applied': 0, 'visited': set()}
    previous_end = 0
    for index, e in enumerate(epochs):
        n, end, c = e['after_actual_action_count'], e['end_actual_action_count'], e['channel']
        require(e['id'] == index and type(n) is int and type(end) is int and
                previous_end <= n <= end <= len(h) and type(c) is int and 1 <= c <= 20,
                'Invalid resolver prefix/order/channel')
        require(all(a['channel'] == c for a in h[n:end]), 'Resolver interval contains another source')
        previous_end = end
        canonical, positive, negative, near, cleared = observations(h, n, c)
        expected_skip = ('already_cleared' if cleared else 'no_canonical_positive_region'
            if not canonical.observations or not canonical.vertices else 'canonical_ready'
            if near is not None or canonical.enclosing_disk().radius <= 19.9 else
            'no_negative_evidence' if not negative else None)
        aux, evidence = None, e['helper_evidence']
        if expected_skip is not None:
            require(e['skip_reason'] == expected_skip and evidence is None and e['initial_aux_vertices'] is None,
                    'Incorrect resolver skip or evidence created without eligible prefix')
        else:
            require(evidence is not None, 'Eligible resolver lacks one helper certificate')
            require(points(evidence['canonical_vertices']) == list(map(tuple, canonical.vertices)) and
                    points(evidence['positive_positions']) == positive and points(evidence['negative_positions']) == negative,
                    'Helper inputs differ from actual entry prefix')
            checked = audit_joint_visibility_certificate(evidence)
            if checked['fallback']:
                require(e['skip_reason'] == 'helper_fallback' and e['initial_aux_vertices'] is None,
                        'Fallback helper used a reduced auxiliary region')
                counters['fallback_epochs'] += 1
            else:
                excluded = boundary_exclusions(canonical.vertices, evidence['output_vertices'])
                require(evidence['old_vertices_excluded'] == excluded,
                        'Reported boundary reduction differs from exact old-vertex inclusion')
                if not excluded:
                    require(e['skip_reason'] == 'no_boundary_reduction' and e['initial_aux_vertices'] is None,
                            'No-boundary-reduction helper must preserve canonical policy')
                else:
                    require(e['skip_reason'] is None and points(e['initial_aux_vertices']) == points(evidence['output_vertices']),
                            'Initial auxiliary region differs from audited output')
                    aux = canonical.copy()
                    aux.vertices, aux._circle = tuple(points(e['initial_aux_vertices'])), None
                    counters['refined_epochs'] += 1
        updates = []
        local_continuation = dict(calls=0, applied_id=None, applied_prefix=None)
        state = {n: aux.copy() if aux is not None else None}
        for j in range(n, end):
            a = h[j]
            if aux is not None and a['action'] == 'measure' and a['result'] == 'direction':
                aux.observe(a['position'], a['bearing_deg'])
                updates.append(dict(after_actual_action_count=j+1, position=list(point(a['position'])),
                    result='direction', bearing_deg=a['bearing_deg'], aux_vertices=[list(p) for p in aux.vertices]))
                if not aux.vertices:
                    aux = None
            event = continuation_map.get((index, j+1))
            eligible = a['action'] == 'measure' and a['phase'] == 'active_localization' and a['result'] == 'no_signal'
            require((event is not None) == eligible, 'Missing/fabricated continuation after real active miss')
            if event is not None:
                aux = replay_continuation(event, aux, h, j+1, index, n, c, updates,
                    local_continuation, continuation_counts, probe_budget)
            state[j+1] = aux.copy() if aux is not None else None
        require(e['aux_updates'] == updates, 'Auxiliary update is missing, fabricated, or uses future/nonpositive feedback')
        require(e['status'] in {'cleared', 'unresolved', 'interrupted'}, 'Unfinished resolver event')
        successes = [a for a in h[n:end] if a['action'] == 'clear' and a['result'] == 'success']
        if e['status'] == 'cleared':
            require(cleared or len(successes) == 1 and h[end-1] == successes[0], 'Resolver success lacks last real clear')
        elif e['status'] == 'unresolved':
            require(not successes, 'Unresolved event contains success')
        else:
            require(e.get('interruption_type'), 'Interruption lacks exception metadata')
        require(math.isfinite(e['decision_wall_s']) and e['decision_wall_s'] >= 0., 'Invalid resolver decision CPU')
        counters['decision_wall_s'] += e['decision_wall_s']
        states[index], starts[index], ends[index], by_id[index] = state, n, end, e

    owned = ownership(record, states, starts, ends, by_id, h, before)
    used_probe_actions, used_clears, used_grids, decision_keys = set(), set(), set(), set()
    next_index, previous_probe_end = {}, {}
    probe_budget = record.get('spec', {}).get('kwargs', {}).get('max_active_probes', 6)

    def context(e):
        i, c, n, end = e['resolver_id'], e['channel'], e['after_actual_action_count'], e['end_actual_action_count']
        require(i in by_id and c == by_id[i]['channel'] and type(n) is int and type(end) is int
                and starts[i] <= n <= end <= ends[i], 'Decision is outside its resolver prefix')
        aux = states[i][n]
        require(aux is not None and points(e['aux_vertices']) == list(map(tuple, aux.vertices)),
                'Decision uses stale or fabricated auxiliary region')
        require(math.isfinite(e['decision_wall_s']) and e['decision_wall_s'] >= 0., 'Invalid decision CPU')
        counters['decision_wall_s'] += e['decision_wall_s']
        canonical, _, _, near, cleared = observations(h, n, c)
        require(canonical.observations and near is None and not cleared, 'Decision made for ready-near or cleared source')
        return i, c, n, end, aux, canonical

    for e in probes:
        i, c, n, end, aux, canonical = context(e)
        require((i, n, e['index']) not in decision_keys and type(e['index']) is int and e['index'] >= 0,
                'Duplicate or invalid probe index')
        actual_index = sum(a['action'] == 'measure' and a['phase'] == 'active_localization' for a in h[starts[i]:n])
        require(e['index'] == actual_index and e['index'] < probe_budget
                and n >= previous_probe_end.get(i, starts[i]), 'Probe order or local budget differs')
        next_index[i], previous_probe_end[i] = e['index']+1, end
        decision_keys.add((i, n, e['index']))
        bearing = canonical.observations[0].bearing_deg
        original, _, selected_original = probe_formula(canonical, bearing, before[n][0], h, n, c)
        expected_original = original[selected_original] if selected_original is not None else None
        require((point(e['canonical_point']) if e['canonical_point'] is not None else None) == expected_original,
                'Logged original probe differs from real canonical heuristic')
        circle = aux.enclosing_disk()
        require(point(e['center']) == tuple(circle.center), 'Wrong auxiliary circle center')
        close(e['radius_m'], circle.radius, 'Wrong auxiliary radius')
        actual = h[n:end]
        if e['kind'] == 'clear':
            require(circle.radius <= 19.9 and max(math.dist(circle.center, p) for p in aux.vertices) <= 19.9+1e-9
                    and point(e['point']) == tuple(circle.center) and not e['candidates'] and not e['fresh_indices']
                    and e['selected'] is None and not e['executed_measure'] and not e['r8_clear_before_measure'],
                    'Auxiliary clear lacks full-region certificate')
            if actual:
                require(len(actual) == 1 and actual[0]['action'] == 'clear' and actual[0]['channel'] == c and
                        point(actual[0]['position']) == point(e['point']) and actual[0]['phase'] == 'joint_visibility_clear',
                        'Auxiliary clear action differs')
                success = actual[0]['result'] == 'success'
                require(e['status'] == ('cleared' if success else 'certified_clear_failed'), 'Wrong auxiliary clear outcome')
                counters['auxiliary_clear_attempts'] += 1
                counters['auxiliary_clear_successes'] += success
                used_clears.add(n)
            else:
                require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'clear', e['point']),
                        'Unexecuted auxiliary clear lacks actual service/terminal gate')
            continue
        require(e['kind'] == 'probe' and circle.radius > 19.9, 'Probe used when auxiliary region certifies clear')
        candidates, fresh, selected = probe_formula(aux, bearing, before[n][0], h, n, c)
        require(points(e['candidates']) == candidates and e['fresh_indices'] == fresh and e['selected'] == selected,
                'Auxiliary probe candidate/freshness/choice differs')
        target = candidates[selected] if selected is not None else None
        require((point(e['point']) if e['point'] is not None else None) == target, 'Auxiliary selected point differs')
        if target is None:
            require(not actual and e['status'] == 'no_fresh_probe' and not e['executed_measure']
                    and not e['r8_clear_before_measure'], 'Empty probe set performed an action')
            continue
        require(len(actual) <= 2 and all(a['channel'] == c and point(a['position']) == target for a in actual),
                'Probe physical range differs from selected point')
        measures = [j for j in range(n, end) if h[j]['action'] == 'measure']
        speculative = [j for j in range(n, end) if h[j]['phase'] == 'speculative_clear_before_probe']
        require(bool(measures) == e['executed_measure'] and bool(speculative) == e['r8_clear_before_measure'],
                'Wrong real probe/R8 execution flags')
        require(len(measures) <= 1 and all(h[j]['phase'] == 'active_localization' for j in measures)
                and len(speculative) <= 1 and len(measures)+len(speculative) == len(actual),
                'Probe interval contains unaccounted action')
        if measures:
            require(measures == [end-1] and e['status'] == 'measured', 'Probe was not last real measurement')
            counters['executed_probes'] += 1
            used_probe_actions.update(measures)
        elif actual and actual[-1]['result'] == 'success':
            require(e['status'] == 'cleared_by_r8', 'R8 success was reported as a fictitious probe')
        else:
            require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'measure', target),
                    'Unexecuted probe lacks actual service/terminal gate')

    seen_terminal_epochs = set()
    for e in terminal_clears:
        i, c, n, end, aux, canonical = context(e)
        require(i not in seen_terminal_epochs, 'Repeated terminal auxiliary clear')
        seen_terminal_epochs.add(i)
        bearing = canonical.observations[0].bearing_deg
        require(point(e['canonical_first_point']) == canonical_first_grid_point(
            canonical.vertices, bearing, before[n][0]), 'Terminal clear original canonical point differs')
        circle = aux.enclosing_disk()
        require(point(e['center']) == tuple(circle.center) and circle.radius <= 19.9 and
                max(math.dist(circle.center, p) for p in aux.vertices) <= 19.9+1e-9,
                'Terminal auxiliary clear lacks full-region certificate')
        close(e['radius_m'], circle.radius, 'Wrong terminal auxiliary radius')
        actual = h[n:end]
        if actual:
            require(len(actual) == 1 and n not in used_clears and actual[0]['action'] == 'clear'
                    and actual[0]['channel'] == c and point(actual[0]['position']) == tuple(circle.center)
                    and actual[0]['phase'] == 'joint_visibility_clear', 'Terminal actual clear differs')
            success = actual[0]['result'] == 'success'
            require(e['status'] == ('cleared' if success else 'certified_clear_failed'), 'Wrong terminal clear outcome')
            used_clears.add(n)
            counters['auxiliary_clear_attempts'] += 1
            counters['auxiliary_clear_successes'] += success
        else:
            require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'clear', circle.center),
                    'Unexecuted terminal clear lacks actual service/terminal gate')

    seen_grid_epochs = set()
    for e in grids:
        i, c, n, end, aux, canonical = context(e)
        require(config == 'probe_optical' and i not in seen_grid_epochs, 'Repeated/disabled auxiliary optical grid')
        seen_grid_epochs.add(i)
        bearing = canonical.observations[0].bearing_deg
        require(e['bearing_deg'] == bearing and e['spacing_m'] == 28. and point(e['start']) == before[n][0],
                'Grid rotation/start/spacing differs from actual prefix')
        grid = points(e['grid'])
        verify_grid(aux.vertices, bearing, before[n][0], grid)
        require(point(e['canonical_first_point']) == canonical_first_grid_point(
            canonical.vertices, bearing, before[n][0]), 'Original canonical grid first point differs')
        require(end-n == e['actual_grid_actions'] and end-n <= len(grid), 'Wrong accepted optical count')
        actual = h[n:end]
        require(all(a['action'] == 'clear' and a['channel'] == c and a['phase'] == 'joint_visibility_optical'
                    and point(a['position']) == grid[j] for j, a in enumerate(actual)), 'Optical action does not follow common grid prefix')
        require(all(a['result'] == 'no_target_in_range' for a in actual[:-1]), 'Optical search continued after success')
        if actual and actual[-1]['result'] == 'success':
            require(e['status'] == 'cleared', 'Optical success status differs')
        elif len(actual) == len(grid):
            require(e['status'] == 'exhausted_without_success', 'Full failed grid lost contradiction marker')
            counters['canonical_fallbacks'] += 1
            p = point(e['canonical_first_point'])
            require((end < len(h) and h[end]['action'] == 'clear' and h[end]['channel'] == c
                     and h[end]['phase'] == 'guaranteed_clearance' and point(h[end]['position']) == p)
                    or stopped(summary, h, before, end, c, 'clear', p), 'Exhausted auxiliary grid omitted canonical fallback')
        else:
            require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'clear', grid[len(actual)]),
                    'Partial optical grid lacks actual service/terminal gate')
        counters['optical_attempts'] += len(actual)
        used_grids.update(range(n, end))
    require(used_clears == {j for j, a in enumerate(h) if a['phase'] == 'joint_visibility_clear'},
            'Unlogged auxiliary certified clear')
    require(used_grids == {j for j, a in enumerate(h) if a['phase'] == 'joint_visibility_optical'},
            'Unlogged auxiliary optical action')
    expected_probes = {j for i in states for j in range(starts[i], ends[i])
        if states[i][j] is not None and h[j]['action'] == 'measure' and h[j]['phase'] == 'active_localization'}
    require(not used_probe_actions.intersection(owned), 'Probe has duplicate parent/anchor owners')
    require(used_probe_actions == expected_probes-owned, 'Unlogged auxiliary probe measurement')
    all_active = {j for j, a in enumerate(h) if a['action'] == 'measure'
                  and a['phase'] == 'active_localization'}
    covered_active = {j for i in states for j in range(starts[i], ends[i]) if j in all_active}
    require(all_active == covered_active, 'Active measurement outside every resolver')
    for i in states:
        for j in range(starts[i], ends[i]):
            if j not in all_active or states[i][j] is not None or j in owned:
                continue
            c = by_id[i]['channel']
            canonical, _, _, near, cleared = observations(h, j, c)
            require(canonical.observations and near is None and not cleared,
                    'Canonical probe was made for an already finished source')
            candidates, _, selected = probe_formula(canonical,
                canonical.observations[0].bearing_deg, before[j][0], h, j, c)
            require(selected is not None and point(h[j]['position']) == candidates[selected],
                    'Unrefined canonical probe differs from the inherited heuristic')
    for c, estimate in summary.get('source_estimates', {}).items():
        canonical = observations(h, len(h), int(c))[0]
        if 'vertices' in estimate:
            require(points(estimate['vertices']) == list(map(tuple, canonical.vertices)), 'Canonical final region changed by auxiliary model')
    require(not params['joint_visibility_model_contradictions'], 'Auxiliary model contradiction; cannot qualify')
    require(continuation_counts['visited'] == set(range(len(continuations))), 'Continuation outside all resolver intervals')
    counters.update(continuation_events=len(continuations), continuation_calls=continuation_counts['session_calls'], continuation_applied=continuation_counts['applied'],
                    continuation_wall_s=sum(e['decision_wall_s'] for e in continuations))
    return dict(passed=True, **counters, boundary='Actual-prefix and full optical strip coverage; generic physical, coverage/scheduling and inherited R8 audits remain separate requirements')


def same(actual, expected, message='Anchor evidence differs'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual)==set(expected), message+' keys')
        for key in expected: same(actual[key], expected[key], message+'/'+str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual)==len(expected), message+' length')
        for x,y in zip(actual,expected): same(x,y,message)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected and type(actual)==type(expected), message)
    elif isinstance(expected, int):
        require(type(actual) is int and actual==expected, message)
    else:
        require(not isinstance(actual,bool) and isinstance(actual,(int,float)),message)
        close(actual,expected,message)


def common_route_first_hit(route, observer, support, weights):
    """Enumerate each latent's FIRST hit along ONE shared route, never its own route.

    A physical step costs movement/5+3; only the first success adds 2. The caller
    later averages these costs. Any positive-weight uncovered node rejects the
    entire model, including weights arbitrarily close to zero.
    """
    route=points(route); support=points(support); observer=point(observer)
    require(len(route)<=256 and bool(route),'Invalid or over-budget common route')
    require(len(support)==len(weights) and bool(support),'Invalid spatial model')
    require(all(isinstance(w,(float,int)) and not isinstance(w,bool) and math.isfinite(w) and w>=0 for w in weights),
            'Negative/nonfinite model weight')
    close(math.fsum(weights),1.,'Branch weights are not normalized')
    reached=[]; previous=observer; total=0.
    for q in route:
        total+=math.dist(previous,q)/5.+3.
        reached.append(total+2.)
        previous=q
    first=[]; terms=[]
    for s,w in zip(support,weights):
        hit=next((j for j,q in enumerate(route) if math.dist(s,q)<=20.),None) if w>0 else None
        require(w==0 or hit is not None,'Positive conditional mass has no physical first hit')
        first.append(hit)
        terms.append(w*reached[hit] if w>0 else 0.)
    return math.fsum(terms),first



from geometry import bearing_halfplanes, disk_halfplanes, clip_polygon
from planning.coverage import _clip_horizontal
from planning.q4_conditional_belief import build_belief, predict_branches, ModelUnavailable

LIMITS = dict(spatial_nodes=24, prior_omni=.5, max_history=64, work_limit=262144,
    candidates=3, bin_deg=1., error_deg=1.005, max_grid_cells=256,
    anchor_forward_m=100., anchor_lateral_m=100., improvement_tolerance_s=1e-9)


class ModelWork:
    def __init__(self): self.used=0
    def add(self, amount=1):
        self.used+=amount
        if self.used>262144: raise ModelUnavailable('scoring_work_budget')


def radio_history(h,n,c):
    """Exact coordinates for fixed feedback; six decimals are only action freshness."""
    observed={}; answer=[]
    for a in h[:n]:
        if a['action']!='measure' or a['channel']!=c: continue
        p=point(a['position'])
        item=dict(position=p,result=a['result'],bearing_deg=a.get('bearing_deg'))
        if p in observed:
            if observed[p]!=item: raise ModelUnavailable('conflicting_fixed_position_feedback')
        else: observed[p]=item; answer.append(item)
    return tuple(answer)


def reference_candidates(h,n,c,current,baseline):
    anchors=[(math.dist(current,point(a['position'])),j,a) for j,a in enumerate(h[:n])
             if a['action']=='measure' and a['channel']==c and a['result']=='direction']
    if not anchors: raise ModelUnavailable('no_real_direction_anchor')
    _,j,a=min(anchors,key=lambda row:(row[0],row[1]))
    seen={tuple(round(v,6) for v in point(x['position'])) for x in h[:n] if x['action']=='measure' and x['channel']==c}
    p=point(a['position']); angle=math.radians(a['bearing_deg']); co=math.cos(angle); si=math.sin(angle)
    result=[point(baseline)]
    for sign in (1.,-1.):
        q=(p[0]+100.*co-sign*100.*si,p[1]+100.*si+sign*100.*co)
        key=tuple(round(v,6) for v in q)
        if key not in seen and key not in {tuple(round(v,6) for v in x) for x in result}: result.append(q)
    return result,dict(action_index=j,position=list(p),bearing_deg=a['bearing_deg'])


def reference_branch(region,q,outcome,work):
    child=region.copy()
    if tuple(outcome)==('no_signal',): return child
    require(outcome[0]=='bearing' and len(outcome)==2 and type(outcome[1]) is int and 0<=outcome[1]<360,
            'Invalid one degree feedback bin')
    constraints=bearing_halfplanes(q,outcome[1]+.5,1.505)+disk_halfplanes(q,1500.,32,outer=True)
    vertices=list(child.vertices)
    for hp in constraints:
        work.add(len(vertices)+1)
        vertices=clip_polygon(vertices,hp)
        if not vertices: raise ModelUnavailable('empty_geometric_branch')
    child.vertices=vertices; child._circle=None
    return child


def reference_grid(child,q,bearing,work):
    circle=child.enclosing_disk()
    if circle.radius<=19.9: return [point(circle.center)],'certified_center'
    angle=math.radians(bearing); co=math.cos(angle); si=math.sin(angle)
    transformed=[(x*co+y*si,-x*si+y*co) for x,y in child.vertices]
    rows=range(math.floor(min(y for x,y in transformed)/28.), math.floor(max(y for x,y in transformed)/28.)+1)
    count=0
    for row in rows:
        work.add(2*len(transformed)+1)
        strip=_clip_horizontal(_clip_horizontal(transformed,28.*row,True),28.*(row+1),False)
        if strip: count+=math.floor(max(x for x,y in strip)/28.)-math.floor(min(x for x,y in strip)/28.)+1
        if count>256: raise ModelUnavailable('common_optical_grid_budget')
    work.add(count*count)
    route=[(p.x,p.y) for p in clearance_grid(child.vertices,bearing_deg=bearing,spacing=28.,start=q)]
    require(len(route)==count,'Independent complete grid count differs')
    verify_grid(child.vertices,bearing,q,route)
    return route,'common_grid'


def reference_first_hit(route,q,nodes,weights,work):
    # Independently integrate costs per node, then charge the documented number
    # of comparisons for the common scan until all positive nodes are absorbed.
    try: cost,hits=common_route_first_hit(route,q,nodes,weights)
    except ValueError as exc:
        if 'no physical first hit' in str(exc): raise ModelUnavailable('uncovered_positive_mass') from exc
        raise
    for step in range(1+max(i for i in hits if i is not None)):
        work.add(sum(w>0 and hit>=step for hit,w in zip(hits,weights)))
    return cost,hits


def reference_score(region,prefix,candidates,current,bearing,c,tuned):
    if not 2<=len(candidates)<=3 or len({tuple(round(v,6) for v in p) for p in candidates})!=len(candidates):
        raise ModelUnavailable('need_two_or_three_distinct_candidates')
    work=ModelWork()
    belief=build_belief(region.copy(),prefix,node_count=24,prior_omni=.5,max_history=64,work_limit=262144)
    work.add(belief.work_used); nodes=[n.position for n in belief.nodes]; scores=[]
    for index,q in enumerate(candidates):
        branches=predict_branches(belief,q,bin_deg=1.,error_deg=1.005)
        require(branches and abs(math.fsum(b.mass for b in branches)-1.)<=1e-8,'Predicted branch masses differ')
        work.add(max(b.work_used for b in branches)); logs=[]
        for b in branches:
            require(math.isfinite(b.mass) and b.mass>0,'Invalid conditional branch mass')
            if b.outcome==('near',): route,kind=[q],'near_clear'
            else: route,kind=reference_grid(reference_branch(region,q,b.outcome,work),q,bearing,work)
            value,hits=reference_first_hit(route,q,nodes,b.spatial_weights,work)
            logs.append(dict(outcome=list(b.outcome),mass=b.mass,spatial_weights=list(b.spatial_weights),
                             first_hit_cost_s=value,kind=kind,route=[list(p) for p in route],first_hit_indices=hits))
        immediate=math.dist(current,q)/5.+5.+int(c!=tuned)
        total=immediate+math.fsum(b['mass']*b['first_hit_cost_s'] for b in logs)
        require(math.isfinite(total) and total>=immediate,'Invalid expected model cost')
        scores.append(dict(index=index,position=list(q),immediate_s=immediate,expected_cost_s=total,branches=logs))
    best=min(range(len(scores)),key=lambda j:(scores[j]['expected_cost_s'],j))
    selected=best if scores[best]['expected_cost_s']<scores[0]['expected_cost_s']-1e-9 else 0
    return selected,dict(nodes=[list(p) for p in nodes],spatial_weights=list(belief.spatial_weights),
        candidate_scores=scores,selected_index=selected,work_used=work.used,limits=dict(LIMITS),
        original_expected_cost_s=scores[0]['expected_cost_s'],selected_expected_cost_s=scores[selected]['expected_cost_s'])



def audit_ownership(record,states,starts,ends,epochs,h,before,metrics):
    params=record['summary']['strategy_parameters']; events=params['anchor_first_hit_log']
    same(params['anchor_first_hit_limits'],dict(LIMITS,actual_new_probe_limit=40),'Changed anchor limits')
    budget=record['spec'].get('kwargs',{}).get('max_active_probes',6)
    parent_probes=params['joint_visibility_probe_log']
    keys=set(); owned=set(); actual_count=0; modeled=0; fallback=0; prior_n=-1
    def preview(domain,bearing,current,n,c):
        values,_,selected=probe_formula(domain,bearing,current,h,n,c)
        return values[selected] if selected is not None else None
    for index,e in enumerate(events):
        i=e['resolver_id']; c=e['channel']; n=e['after_actual_action_count']; end=e['end_actual_action_count']
        require(e['id']==index and type(i) is int and i in epochs and epochs[i]['channel']==c,
                'Anchor has missing/wrong resolver identity')
        require(type(n) is int and type(end) is int and starts[i]<=n<=end<=ends[i] and n>=prior_n,
                'Anchor interval lies outside real resolver')
        prior_n=n
        actual_index=sum(a['action']=='measure' and a['phase']=='active_localization' for a in h[starts[i]:n])
        require(type(e['index']) is int and e['index']==actual_index and 0<=actual_index<budget,
                'Anchor index/budget differs from actual probes')
        key=(i,n,actual_index)
        require(key not in keys,'Duplicate anchor decision'); keys.add(key)
        current,tuned,_=before[n]
        same(e['current_position'],current,'Anchor current point'); require(e['current_channel']==tuned,'Anchor tuned channel')
        require(e['actual_new_probes_before']==actual_count and type(e['actual_new_probes_before']) is int,
                'Anchor count is not actual prefix count')
        canonical,positive,negative,near,cleared=observations(h,n,c); aux=states[i][n]
        expected=dict(canonical_radius_m=None,model_region_kind=None,model_vertices=None,baseline=None,
                      anchor=None,candidates=[],model=None)
        reason=None; choice=0; choices=[]
        if actual_index!=0: reason='not_first_probe'
        elif cleared or near is not None or not canonical.vertices or not canonical.observations:
            reason='no_live_positive_region'
        else:
            radius=canonical.enclosing_disk().radius; expected['canonical_radius_m']=radius
            if not math.isfinite(radius) or radius<=40.: reason='canonical_not_wide'
            elif aux is not None and aux.enclosing_disk().radius<=19.9: reason='auxiliary_ready'
            elif actual_count>=40: reason='actual_new_probe_limit'
            else:
                domain=aux if aux is not None else canonical; bearing=canonical.observations[0].bearing_deg
                expected.update(model_region_kind='auxiliary' if aux is not None else 'canonical',
                                model_vertices=[list(p) for p in domain.vertices])
                baseline=preview(domain,bearing,current,n,c)
                expected['baseline']=list(baseline) if baseline is not None else None
                if baseline is None: reason='no_parent_probe'
                else:
                    try:
                        choices,anchor=reference_candidates(h,n,c,current,baseline)
                        expected.update(candidates=[list(p) for p in choices],anchor=anchor)
                        choice,model=reference_score(domain,radio_history(h,n,c),choices,current,bearing,c,tuned)
                        expected['model']=model; modeled+=1
                    except ModelUnavailable as error:
                        reason='model_unavailable'
                        require(e.get('unavailable_reason')==str(error),'Whole-decision fallback reason differs')
                    if reason is None and choice==0: reason='original_best'
        for k,v in expected.items(): same(e[k],v,'Anchor prefix/model/'+k)
        if reason!='model_unavailable': require('unavailable_reason' not in e,'Fabricated model fallback')
        changed=reason is None
        require(type(e['changed']) is bool and e['changed']==changed and type(e['parent_fallback']) is bool
                and e['parent_fallback']==(not changed),'Partial-score or wrong ownership selection')
        actual=h[n:end]
        if changed:
            target=choices[choice]
            same(e['selected'],target,'Anchor replacement point')
            require(actual_count<40 and len(actual)<=1,'Extra owned action or actual replacement cap')
            require(not any(p['resolver_id']==i and p['after_actual_action_count']==n for p in parent_probes),
                    'New anchor was forged as an original parent probe')
            if actual:
                a=actual[0]
                require(a['action']=='measure' and a['phase']=='active_localization' and a['channel']==c
                        and point(a['position'])==target,'Owned radio differs from selected prefix point')
                require(e['status']=='measured' and e['executed_measure'] is True and e['actual_result']==a['result'],
                        'Anchor real feedback or execution flag differs')
                require(not e['r8_clear_before_measure'],'Wide anchor falsely ran small-region R8')
                close(e['actual_cost_s'],before[end][2]-before[n][2],'Anchor fee differs from real wire')
                owned.add(n); actual_count+=1
            else:
                require(e['status']=='interrupted' and e['executed_measure'] is False and e['actual_result'] is None
                        and not e['r8_clear_before_measure'] and e['actual_cost_s']==0.
                        and stopped(record['summary'],h,before,n,c,'measure',target),
                        'Unexecuted owned radio lacks actual service/terminal gate')
        else:
            fallback+=1
            require(e['executed_measure'] is False and e['r8_clear_before_measure'] is False
                    and e['actual_result'] is None and e['actual_cost_s']==0. and e['action_wall_s']==0.,
                    'Fallback claimed ownership of a parent action')
            if aux is not None and aux.enclosing_disk().radius<=19.9:
                # The parent may perform its own auxiliary clear INSIDE next_probe.
                matches=[p for p in parent_probes if p['resolver_id']==i and p['after_actual_action_count']==n
                         and p['index']==actual_index and p['kind']=='clear']
                require(len(matches)==1 and matches[0]['end_actual_action_count']==end,
                        'Parent auxiliary clear boundary differs')
                require(e['status'] in {'parent_clear','interrupted'} and e['selected'] is None,
                        'Parent clear was turned into a probe')
            else:
                require(not actual,'Fallback next_probe falsely owns later physical actions')
                if e['status']=='interrupted':
                    require(terminal(record['summary'],n,h),'Unexplained parent-preview interruption')
                else:
                    require(e['status']==reason,'Incorrect fallback status')
                    target=preview(aux if aux is not None else canonical,canonical.observations[0].bearing_deg,current,n,c)
                    same(e['selected'],list(target) if target is not None else None,'Fallback point differs from unchanged parent')
        require(type(e['actual_new_probes_after']) is int and e['actual_new_probes_after']==actual_count,
                'Counter includes attempted/rejected or fallback probes')
        require(all(isinstance(e[k],(int,float)) and not isinstance(e[k],bool) and math.isfinite(e[k]) and e[k]>=0
                    for k in ('decision_wall_s','action_wall_s')),'Invalid anchor CPU metadata')
    # Every actual active probe, R8 interception, and auxiliary next_probe clear
    # has exactly one invocation record. No model log may replace a real radio.
    required=set()
    for i in epochs:
        c=epochs[i]['channel']
        for j in range(starts[i],ends[i]):
            a=h[j]
            if a['phase']=='speculative_clear_before_probe' or a['action']=='measure' and a['phase']=='active_localization':
                n=j-1 if a['action']=='measure' and j>starts[i] and h[j-1]['phase']=='speculative_clear_before_probe' else j
                k=sum(x['action']=='measure' and x['phase']=='active_localization' for x in h[starts[i]:n])
                required.add((i,n,k))
    required.update((p['resolver_id'],p['after_actual_action_count'],p['index']) for p in parent_probes)
    require(required<=keys,'Missing anchor invocation log for actual/parent-owned probe')
    metrics.update(anchor_probe_actions=actual_count,anchor_decisions=len(events),scored_decisions=modeled,
                   parent_fallback_decisions=fallback)
    return owned


def audit_anchor_first_hit_prefix(record):
    count=verify_source_contract(); verify_replay_delta()
    spec=record['spec']; kwargs=spec.get('kwargs',{})
    require(spec['entrypoint']==ENTRY and record['row']['strategy']==LABEL,'Unreviewed anchor entry/label')
    require(set(kwargs)<={'config','max_expansions','max_actions','max_active_probes','problem'}
            and kwargs.get('config')==CONFIG and type(kwargs.get('problem',4)) is int and kwargs.get('problem',4)==4
            and type(kwargs.get('max_expansions',200)) is int and kwargs.get('max_expansions',200)==200,'Unreviewed anchor spec')
    require(record['summary']['strategy_parameters']['anchor_first_hit_config']==CONFIG,'Wrong anchor configuration')
    view=dict(record,spec=dict(spec,entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                              kwargs=dict(kwargs,config='after_active_miss_optical')))
    result={}
    r12=_audit_r12_owned(view,lambda *args:audit_ownership(*args,result))
    r8=audit_clear_before_probe_prefix(view)
    accepted=dict(view,history=[w for w in view['history'] if w['action'] not in {'/measure','/clear'} or w['response'].get('accepted') is True])
    range_result=audit_range_prefix(accepted); scheduling=audit_scheduling_prefix(accepted)
    for child in (r12,r8,range_result,scheduling): require(child.get('passed') is True,'Inherited anchor audit did not pass')
    result.update(passed=True,r12=r12,r8=r8,range=range_result,scheduling=scheduling,source_contract_files=count,
        boundary='Real wire and complete inherited safety audit, with explicit new-probe ownership. Pinned continuous radio integration is a model assumption; candidates, shared routes, first-hit costs and all actual choices independently replayed. No source truth is used for ranking; no actual actions are removed.')
    return result


def audit_full(record):
    from experiments.audit_q4_cover import audit_record
    generic=audit_record(record)
    require(generic.get('passed') is True,'Generic physical/coverage audit failed: '+str(generic.get('errors')))
    return dict(passed=True,generic=generic,prefix=audit_anchor_first_hit_prefix(record))

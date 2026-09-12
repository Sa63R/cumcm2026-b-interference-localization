"""Centered one-cover feedback surrogate. All hypothetical state is private.

Four deterministic volume-stratified worlds approximate a research posterior;
they are neither calibrated probabilities nor physical certificates. The only
route targets are real or feedback-updated region centers / near observations.
"""
from dataclasses import asdict
import math
import time

from planning.chain_route import ChainSource, solve_chain_route
from planning.release_chain_route import solve_release_chain_route
from planning.q4_conditional_belief import build_belief, ModelUnavailable
from simulator_client.state import Position
from strategies.q4_range_pruning import region_distance_lower

WORLDS = 4
SOLVE_EXPANSIONS = 200
PREDICTION_LIMIT = 32
VETO_LIMIT = 8
EXPANSION_LIMIT = 11 * SOLVE_EXPANSIONS * PREDICTION_LIMIT
QUANTILE_MULTIPLIERS = (3, 5, 7, 11, 13)


def xy(p):
    p = Position.coerce(p)
    return [p.x, p.y]


def quantile(world, channel, dimension):
    """Four equal strata, channel/dimension phases; no random-state input."""
    return ((world + .5) / WORLDS +
            ((channel * QUANTILE_MULTIPLIERS[dimension]) % 23 + .5) / 23) % 1.


def _choose(weights, u):
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ModelUnavailable('empty_sampling_volume')
    threshold, accumulated = u * total, 0.
    positive = [i for i, w in enumerate(weights) if w > 0]
    for i in positive:
        accumulated += weights[i]
        if threshold < accumulated:
            return i
    return positive[-1]


def _quantized_bearing(bearing, error):
    value = round((bearing + error) % 360., 2) % 360.
    delta = (value - bearing + 180.) % 360. - 180.
    if delta > 1. + 1e-12:
        value = round(value - .01, 2) % 360.
    elif delta < -1. - 1e-12:
        value = round(value + .01, 2) % 360.
    return value


def sample_feedback(belief, observer, channel, world):
    q = tuple(xy(observer))
    for old in belief.prefix:
        if q == old.position:
            return dict(channel=channel, result=old.result, bearing_deg=old.bearing_deg,
                        origin='locked_actual_feedback', latent=None)
    uniforms = [quantile(world, channel, d) for d in range(5)]
    ni = _choose(belief.spatial_weights, uniforms[0])
    node = belief.nodes[ni]
    components, masses = [], []
    if node.omni_interval:
        lo, hi = node.omni_interval
        components.append(('omni', lo, hi, ()))
        masses.append(belief.prior_omni * (hi-lo) / 500.)
    for segment in node.directional_segments:
        components.append(('directional', segment.lower, segment.upper, segment.angles))
        masses.append((1.-belief.prior_omni) * (segment.upper-segment.lower) *
                      math.fsum(b-a for a, b in segment.angles) / (500. * math.tau))
    ci = _choose(masses, uniforms[1])
    kind, lo, hi, angles = components[ci]
    radius = lo + (hi-lo) * uniforms[2]
    theta = None
    if kind == 'directional':
        lengths = [b-a for a, b in angles]
        target = uniforms[3] * math.fsum(lengths)
        for (a, b), length in zip(angles, lengths):
            if target < length:
                theta = a + target
                break
            target -= length
        if theta is None:
            theta = math.nextafter(angles[-1][1], angles[-1][0])
    px, py = node.position
    dx, dy = q[0]-px, q[1]-py
    distance = math.hypot(dx, dy)
    visible = distance <= radius and (theta is None or
        math.cos(theta)*dx + math.sin(theta)*dy >= -1e-12*max(1., distance))
    result, bearing = 'no_signal', None
    error = 2.*uniforms[4]-1.
    if visible:
        result = 'near' if distance <= 5. else 'direction'
        if result == 'direction':
            bearing = _quantized_bearing(math.degrees(math.atan2(-dy, -dx)) % 360., error)
    return dict(channel=channel, result=result, bearing_deg=bearing, origin='conditional_volume',
        latent=dict(node_index=ni, position=list(node.position), component_index=ci,
                    source_type=kind, radius_m=radius, orientation_rad=theta,
                    bearing_error_deg=error, quantiles=uniforms))


def describe_source(item):
    region, near = item['region'], item['near']
    if near is not None:
        target, radius, ready = xy(near), None, True
    else:
        if region is None or not region.vertices:
            raise ModelUnavailable('missing_or_empty_known_region')
        circle = region.copy().enclosing_disk()
        target, radius = xy(circle.center), circle.radius
        if not math.isfinite(radius) or any(not math.isfinite(v) for v in target):
            raise ModelUnavailable('nonfinite_known_target')
        ready = radius <= 19.9
    return dict(channel=item['channel'], target=target, radius_m=radius, ready=ready,
        near_point=None if near is None else xy(near),
        vertices=[] if region is None else [list(v) for v in region.vertices],
        error_deg=None if region is None else region.error_deg,
        observations=[] if region is None else [asdict(o) for o in region.observations])


def apply_feedback(item, observer, feedback):
    """Identical observation updates to canonical _Search; never use latent."""
    child = dict(item, region=None if item['region'] is None else item['region'].copy())
    if feedback['result'] == 'direction':
        if child['region'] is None:
            raise ModelUnavailable('direction_without_known_region')
        child['region'].observe(observer, feedback['bearing_deg'])
        if not child['region'].vertices:
            raise ModelUnavailable('predicted_empty_canonical')
    elif feedback['result'] == 'near':
        child['near'] = Position.coerce(observer)
    elif feedback['result'] != 'no_signal':
        raise ModelUnavailable('invalid_predicted_feedback')
    describe_source(child)
    return child


def evaluate_cover_feedback(*, current, covers, ready_channels, ready_positions,
                            sources, cleared_count, selected_channel, incumbent_cost_s,
                            expansion_allowance=EXPANSION_LIMIT):
    began = time.perf_counter()
    log = dict(status='started', fallback_reason=None, recommend_veto=False,
               information_changed=False, input_sources=[], beliefs=[], worlds=[], solve_log=[],
               expanded=0, max_expansions_per_solve=SOLVE_EXPANSIONS)

    def record_solve(role, inputs, function, **kwargs):
        if log['expanded'] + SOLVE_EXPANSIONS > expansion_allowance:
            raise ModelUnavailable('prediction_expansion_budget')
        result = function(max_expansions=SOLVE_EXPANSIONS, **kwargs)
        log['expanded'] += result.expanded
        log['solve_log'].append(dict(role=role, **inputs, result=asdict(result),
                                    expanded_charged=result.expanded))
        return result

    try:
        if not covers or selected_channel not in ready_channels or len(ready_channels) != len(ready_positions):
            raise ModelUnavailable('invalid_current_ready_task')
        q, tail = Position.coerce(covers[0]), list(covers[1:])
        descriptions = [describe_source(s) for s in sources]
        log['input_sources'] = [dict(**d, prefix=list(s['prefix'])) for s, d in zip(sources, descriptions)]
        if len({s['channel'] for s in sources}) != len(sources) or len(sources)+cleared_count > 16:
            raise ModelUnavailable('invalid_known_channel_set')
        background = 6.*max(0, 20-cleared_count-len(ready_channels))
        forced = record_solve('forced_q', dict(current=xy(q), covers=[xy(p) for p in tail],
            channels=list(ready_channels), positions=[xy(p) for p in ready_positions],
            background_scan_s=background, scan_source_s=0., service_s=5.), solve_chain_route,
            cover_points=tail, sources=[ChainSource(p, 5.) for p in ready_positions], start=q,
            background_scan_s=background, scan_source_s=0.)
        full_cost = Position.coerce(current).distance_to(q)/5. + background + forced.cost_s
        log['forced_cover'] = dict(cost_s=full_cost, initial_cost_s=full_cost-forced.cost_s,
            order=[['cover', 0]] + [[kind, index+1 if kind == 'cover' else index] for kind,index in forced.order])
        log['D0_s'] = max(0., full_cost-incumbent_cost_s)
        beliefs = {}
        for s, d in zip(sources, descriptions):
            if d['ready']:
                continue
            if s['region'].observations and region_distance_lower(q, s['region'].vertices) > 1500.+1e-5:
                beliefs[s['channel']] = 'range_skip'
                continue
            belief = build_belief(s['region'], s['prefix'], node_count=24, prior_omni=.5,
                                  max_history=64, work_limit=262144)
            beliefs[s['channel']] = belief
            log['beliefs'].append(dict(channel=s['channel'], **asdict(belief)))
        world_sources = []
        for h in range(WORLDS):
            updates, feedbacks, changed = [], [], []
            for s, d in zip(sources, descriptions):
                if d['ready'] or beliefs.get(s['channel']) == 'range_skip':
                    feedbacks.append(dict(channel=s['channel'], result=None, bearing_deg=None,
                        origin='ready_skip' if d['ready'] else 'range_skip', latent=None))
                    updates.append(s)
                    continue
                feedback = sample_feedback(beliefs[s['channel']], q, s['channel'], h)
                # The fixed-point reply is already in this canonical prefix.
                # Re-clipping it could create a purely numerical "change".
                updated = s if feedback['origin'] == 'locked_actual_feedback' else apply_feedback(s, q, feedback)
                after = describe_source(updated)
                # Appending a redundant observation alone is not information.
                if any(after[key] != d[key] for key in ('vertices','near_point','ready','target')):
                    changed.append(s['channel'])
                updates.append(updated); feedbacks.append(feedback)
            world_sources.append(updates)
            log['worlds'].append(dict(index=h, feedbacks=feedbacks,
                updated_sources=[describe_source(s) for s in updates], changed_channels=changed))
        log['information_changed'] = any(w['changed_channels'] for w in log['worlds'])
        if not log['information_changed']:
            raise ModelUnavailable('no_information_change')

        def value(role, items, omit):
            kept = [s for s in items if not (omit and s['channel'] == selected_channel)]
            ds = [describe_source(s) for s in kept]
            # c is already ready, so +c/-c has the same unknown background and
            # every other source's scan column. Hypothetical updates earn no
            # REAL scan credits; this is a value-only future matrix.
            bg = [6.*max(0,20-cleared_count-int(omit)-len(kept)) for _ in tail]
            weights = [[0. if d['ready'] or (s['region'] is not None and s['region'].observations and
                region_distance_lower(p,s['region'].vertices)>1500.+1e-5) else 6.
                for s,d in zip(kept,ds)] for p in tail]
            releases = [0 if d['ready'] else len(tail) for d in ds]
            result = record_solve(role, dict(current=xy(q), covers=[xy(p) for p in tail],
                channels=[s['channel'] for s in kept], positions=[d['target'] for d in ds],
                ready=[d['ready'] for d in ds], release_indices=releases,
                background_scan_s=bg, source_scan_s=weights, service_s=5.), solve_release_chain_route,
                cover_points=tail, sources=[ChainSource(d['target'],5.) for d in ds], start=q,
                background_scan_s=bg, source_scan_s=weights, release_indices=releases)
            return result.cost_s

        v0p = value('V0_plus', sources, False)
        v0m = value('V0_minus', sources, True)
        log['base_marginal_s'] = v0p-v0m
        marginal = []
        for h, items in enumerate(world_sources):
            plus = value(f'world{h}_plus', items, False)
            minus = value(f'world{h}_minus', items, True)
            marginal.append(plus-minus)
            log['worlds'][h].update(value_plus_s=plus,value_minus_s=minus)
        log['world_marginals_s'] = marginal
        log['D_s'] = log['D0_s'] + math.fsum(marginal)/WORLDS - log['base_marginal_s']
        if not math.isfinite(log['D_s']):
            raise ModelUnavailable('nonfinite_centered_score')
        log.update(status='scored', recommend_veto=log['D_s'] < -1e-9)
    except ModelUnavailable as error:
        log.update(status='fallback', fallback_reason=str(error), recommend_veto=False)
    except (ValueError, ArithmeticError) as error:
        log.update(status='fallback', fallback_reason=type(error).__name__+': '+str(error), recommend_veto=False)
    finally:
        log['runtime_s'] = time.perf_counter()-began
    return log

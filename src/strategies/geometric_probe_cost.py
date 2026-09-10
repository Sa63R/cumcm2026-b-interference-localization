"""Q3 first-probe ablations with explicit local costs and replacement credit.

Finite polygon hypotheses and synthetic bounded error sequences are planning
proxies, not a calibrated posterior or an expected-optimality certificate.
Only actual observations update the executing policy's regions.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import math
import statistics
import time

from simulator_client.state import ClientState, Position, TimeBreakdown

from .efficient import EfficientSearch
from .geometric_joint import (GeometricJointSearch, JointObservationResult,
                              polygon_quadrature, shared_observation_value)
from .search import _StopSearch


class _ProxyTimeout(Exception):
    pass


def observation_worlds(region, limit=9):
    """Filter area quadrature by the original fixed-radius observation model."""
    result = []
    for source in polygon_quadrature(region.vertices, limit):
        if math.hypot(*source) > 1800.+1e-8:
            continue
        positive = [math.dist(source, obs.position) for obs in region.observations]
        if not positive or any(distance <= 5 for distance in positive):
            continue
        low = max(1000., max(positive))
        negatives = getattr(region, 'no_signal_positions', ())
        high = min([1500.] + [math.dist(source, point)-1e-7 for point in negatives])
        if low > high:
            continue
        for phase in range(3):
            result.append((source, (low+high)/2, phase))
    return tuple(result)


def radius_width_weights(region, worlds):
    """Unnormalized area-hypothesis masses after integrating a uniform R prior.

    Keep precisely the existing worlds and their equal-mass area quadrature.
    This only adds the compatible radius interval's relative width. It does
    not add the quantized bearing likelihood, refine the outer polygon, or
    integrate future errors/R inside each world; it is not an exact posterior.
    The 1e-7 negative-response margin matches observation_worlds exactly.
    """
    values = []
    for source, _radius, _phase in worlds:
        positive = [math.dist(source, obs.position) for obs in region.observations]
        if (not positive or math.hypot(*source) > 1800.+1e-8
                or any(distance <= 5. for distance in positive)):
            values.append(0.)
            continue
        low = max(1000., max(positive))
        high = min([1500.] + [math.dist(source, point)-1e-7
                   for point in getattr(region, 'no_signal_positions', ())])
        values.append(max(0., high-low)/500.)
    return tuple(values)


class _HypothesisClient:
    """Cost-only generative client for an explicit caller-supplied hypothesis."""
    def __init__(self, region, world, start, tuned_channel, deadline):
        self.state = ClientState(session='active', position=Position.coerce(start),
                                 current_channel=tuned_channel,
                                 max_virtual_duration_s=360000., time_breakdown=TimeBreakdown())
        self.source, self.radius, self.error_phase = world
        self.deadline = deadline
        self._anchors = {tuple(obs.position): ('direction', obs.bearing_deg) for obs in region.observations}
        self._anchors.update({tuple(point): ('no_signal', None)
                              for point in getattr(region, 'no_signal_positions', ())})

    def _check(self):
        if time.perf_counter() >= self.deadline:
            raise _ProxyTimeout()

    def retune_without_measurement(self, channel):
        # These retunes precede primary measurements whose other costs were
        # already charged in the primary continuation. No cost is duplicated.
        if self.state.current_channel != channel:
            self.state.time_breakdown.switching_s += 1.
            self.state.virtual_time_s += 1.
            self.state.current_channel = channel

    def _move(self, point):
        self._check()
        point = Position.coerce(point)
        movement = self.state.position.distance_to(point)/5.
        self.state.time_breakdown.movement_s += movement
        self.state.virtual_time_s += movement
        self.state.position = point
        return point

    def measure(self, point, channel):
        point = self._move(point)
        self.retune_without_measurement(channel)
        self.state.time_breakdown.detection_s += 5.
        self.state.virtual_time_s += 5.
        key = (point.x, point.y)
        if key not in self._anchors:
            distance = math.dist(key, self.source)
            if distance <= 5:
                response = ('near', None)
            elif distance > self.radius:
                response = ('no_signal', None)
            else:
                # Three synthetic bounded coordinate fields, unrelated to a
                # scenario seed. Plain/shared branches agree at the same new
                # coordinate even when their measurement counts differ.
                tag = f'probe-proxy-v1:{self.error_phase}:{point.x:.6f}:{point.y:.6f}'.encode('ascii')
                number = int.from_bytes(hashlib.blake2s(tag, digest_size=4).digest(), 'little')
                error = 2.*number/(2**32-1)-1.
                bearing = (math.degrees(math.atan2(self.source[1]-point.y, self.source[0]-point.x))+error) % 360.
                response = ('direction', round(bearing, 2) % 360.)
            self._anchors[key] = response
        kind, bearing = self._anchors[key]
        result = dict(accepted=True, measure_result=kind)
        if bearing is not None:
            result['svd_deg'] = bearing
        return result

    def clear(self, point, channel):
        point = self._move(point)
        success = math.dist((point.x, point.y), self.source) <= 20.
        self.state.time_breakdown.optical_s += 3.
        self.state.virtual_time_s += 3.
        if success:
            self.state.time_breakdown.removal_s += 2.
            self.state.virtual_time_s += 2.
        return dict(accepted=True, clear_result='success' if success else 'no_target_in_range')


def _continuation(region, world, start, tuned, channel, observed, max_probes, deadline):
    client = _HypothesisClient(region, world, start, tuned, deadline)
    policy = EfficientSearch(client, 2000, max_probes, None)
    policy.regions = {channel: region.copy()}
    policy.first_bearings = {channel: region.observations[0].bearing_deg}
    policy.detected = {channel}
    policy.observed_positions = {channel: set(observed)}
    return policy


def primary_cost_samples(region, worlds, current, tuned, channel, probe, observed,
                         max_probes=6, deadline=math.inf):
    """Replay the frozen single-source tail; incomplete traces never win."""
    values = []
    for world in worlds:
        policy = _continuation(region, world, current, tuned, channel, observed,
                               max(0, max_probes-1), deadline)
        try:
            policy._perform('measure', probe, channel, 'active_localization')
            if not policy._resolve(channel):
                return None
        except _StopSearch:
            return None
        values.append(dict(cost_s=policy.client.state.virtual_time_s,
                           components_s=asdict(policy.client.state.time_breakdown),
                           actions=policy.report.action_history,
                           clear_position=policy.client.state.position))
    return values


def secondary_credit(region, worlds, primary_trace, primary_channel, channel, observed,
                     joint_config, max_probes=6, deadline=math.inf, value_cache=None):
    """Net information credit, including replacement of later shared measures.

    Compare a full sequential sharing schedule along this SAME primary trace
    against no sharing on it, followed by the same efficient secondary tail.
    Primary motion/actions are already charged elsewhere; only extra sharing,
    induced retunes, and subsequent secondary resolution are charged here.
    """
    endpoint = Position.coerce(primary_trace[-1]['position'])
    values, sharing_counts = [], []
    value_cache = {} if value_cache is None else value_cache
    for world in worlds:
        plain = _continuation(region, world, endpoint, primary_channel, channel,
                              observed, max_probes, deadline)
        try:
            if not plain._resolve(channel):
                return None
            shared = _continuation(region, world, primary_trace[0]['position'], primary_channel,
                                   channel, observed, max_probes, deadline)
            shared_count = 0
            for primary_action in primary_trace:
                shared.client._check()
                # Jump only in this accounting client: primary travel was paid
                # in primary_cost_samples, including travel to its clear point.
                point = Position.coerce(primary_action['position'])
                shared.client.state.position = point
                if primary_action['action'] == 'measure':
                    shared.client.retune_without_measurement(primary_channel)
                    enabled = joint_config.active_points
                    resume = primary_channel
                else:
                    enabled = (joint_config.after_clear
                               and primary_action.get('result') == 'success')
                    resume = shared.client.state.current_channel
                key = (round(point.x, 6), round(point.y, 6))
                if (not enabled or not joint_config.max_shared_per_stop
                        or channel in shared.near_points
                        or key in shared.observed_positions.get(channel, set())):
                    continue
                current_region = shared.regions[channel]
                cache_key = (tuple(current_region.vertices), current_region.error_deg,
                             current_region.reception_radius, point.x, point.y)
                if cache_key not in value_cache:
                    value_cache[cache_key] = shared_observation_value(current_region, point)
                value = value_cache[cache_key]
                if value is None:
                    continue
                overhead = 5+int(channel != shared.client.state.current_channel)+int(channel != resume)
                if value['travel_saving_proxy_m']/5-overhead < joint_config.minimum_net_gain_s:
                    continue
                shared._perform('measure', point, channel, 'proxy_shared')
                shared_count += 1
            if not shared._resolve(channel):
                return None
        except _StopSearch:
            return None
        values.append(plain.client.state.virtual_time_s-shared.client.state.virtual_time_s)
        sharing_counts.append(shared_count)
    return dict(mean_credit_s=statistics.fmean(values), credits_s=values,
                mean_shared_count=statistics.fmean(sharing_counts))


@dataclass
class ProbeCostResult(JointObservationResult):
    first_probe_planning: dict = field(default_factory=dict)


class GeometricProbeCostSearch(GeometricJointSearch):
    def __init__(self, client, max_actions, max_active_probes, joint_config, mode, max_planning_s,
                 hypothesis_weighting='equal'):
        if mode not in ('disabled', 'single', 'cross'):
            raise ValueError('probe_mode must be disabled, single, or cross')
        if hypothesis_weighting not in ('equal', 'radius_width'):
            raise ValueError('hypothesis_weighting must be equal or radius_width')
        if hypothesis_weighting != 'equal' and mode != 'single':
            raise ValueError('radius_width weighting is an isolated single-mode ablation')
        if (isinstance(max_planning_s, bool) or not isinstance(max_planning_s, (int, float))
                or not math.isfinite(max_planning_s) or not 0 <= max_planning_s <= 600):
            raise ValueError('max_planning_s must be finite in 0..600')
        super().__init__(client, max_actions, max_active_probes, joint_config)
        self.mode, self.max_planning_s = mode, max_planning_s
        self.hypothesis_weighting = hypothesis_weighting
        self.variant = 'geometric_probe_'+mode
        self.report = ProbeCostResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.strategy_parameters.update(probe_mode=mode, max_planning_s=max_planning_s)
        self.report.strategy_parameters['hypothesis_weighting'] = hypothesis_weighting
        self.report.first_probe_planning = dict(planning_s=0., changed=0, decisions=[], fallbacks=[])
        self._probe_considered = set()

    def _next_probe(self, channel, index):
        baseline = super()._next_probe(channel, index)
        if (baseline is None or self.mode == 'disabled' or index != 0
                or channel in self._probe_considered):
            return baseline
        self._probe_considered.add(channel)
        stats = self.report.first_probe_planning
        remaining = self.max_planning_s-stats['planning_s']
        real_left = getattr(self.client, 'remaining_real_time_s', None)
        if real_left is not None:
            remaining = min(remaining, real_left-5.)
        if remaining <= 0:
            stats['fallbacks'].append('compute_budget')
            return baseline
        started, deadline = time.perf_counter(), time.perf_counter()+remaining
        region = self.regions[channel]
        circle = region.enclosing_disk()
        angle = math.radians(self.first_bearings[channel])
        transverse = (-math.sin(angle), math.cos(angle))
        step = min(150., max(25., circle.radius/4))
        candidates = [baseline]
        for multiple in (-2., -1., 1., 2.):
            candidate = Position(circle.center[0]+multiple*step*transverse[0],
                                 circle.center[1]+multiple*step*transverse[1])
            key = (round(candidate.x, 6), round(candidate.y, 6))
            if (key not in self.observed_positions.get(channel, set())
                    and all(math.dist((candidate.x, candidate.y), v) <= 1000.-1e-6 for v in region.vertices)
                    and candidate not in candidates):
                candidates.append(candidate)
        worlds = observation_worlds(region)
        if not worlds or len(candidates) < 2:
            stats['fallbacks'].append('no_worlds_or_candidates')
            return baseline
        weights = radius_width_weights(region, worlds) if self.hypothesis_weighting == 'radius_width' else None
        weight_sum = math.fsum(weights) if weights is not None else None
        if weights is not None and weight_sum <= 0.:
            stats['fallbacks'].append('zero_radius_mass')
            stats['planning_s'] += time.perf_counter()-started
            return baseline
        decision = dict(channel=channel, current=[self.client.state.position.x, self.client.state.position.y],
                        baseline=[baseline.x, baseline.y], worlds=len(worlds), candidates=[], status='baseline')
        if weights is not None:
            decision.update(hypothesis_weighting='radius_width', radius_weights=list(weights),
                            effective_world_count=weight_sum**2/math.fsum(w*w for w in weights))
        stats['decisions'].append(decision)
        try:
            values = []
            traces = []
            for candidate in candidates:
                samples = primary_cost_samples(region, worlds, self.client.state.position,
                    self.client.state.current_channel, channel, candidate,
                    self.observed_positions.get(channel, set()), self.max_active_probes, deadline)
                if samples is None:
                    stats['fallbacks'].append('incomplete_primary_proxy')
                    decision['status'] = 'incomplete_primary_proxy'
                    return baseline
                cost = (statistics.fmean(sample['cost_s'] for sample in samples)
                        if weights is None else math.fsum(sample['cost_s']*w
                            for sample, w in zip(samples, weights))/weight_sum)
                # Same quadrature index/error phase across candidate positions;
                # this is one representative trace, not hidden actual truth.
                representative = samples[len(samples)//2]['actions']
                traces.append(representative)
                values.append(cost)
                decision['candidates'].append(dict(position=[candidate.x, candidate.y],
                    primary_mean_cost_s=cost, primary_costs_s=[sample['cost_s'] for sample in samples],
                    representative_clear_position=representative[-1]['position']))
            credits = [0.]*len(candidates)
            if self.mode == 'cross':
                primary_endpoint = Position.coerce(traces[0][-1]['position'])
                others = [(c, self._target(c)) for c in sorted(self.detected-self.cleared-{channel})
                          if c not in self.near_points and c not in self.blocked]
                others = [(c, point) for c, point in others if point is not None]
                other = min(others, key=lambda item: (primary_endpoint.distance_to(item[1]), item[0]))[0] if others else None
                decision['secondary_channel'] = other
                if other is not None:
                    secondary_worlds = observation_worlds(self.regions[other])
                    if secondary_worlds:
                        value_cache = {}
                        for i, trace in enumerate(traces):
                            credit = secondary_credit(self.regions[other], secondary_worlds, trace,
                                channel, other, self.observed_positions.get(other, set()),
                                self.joint_config, self.max_active_probes, deadline, value_cache)
                            if credit is None:
                                stats['fallbacks'].append('incomplete_secondary_proxy')
                                decision['status'] = 'incomplete_secondary_proxy'
                                return baseline
                            credits[i] = credit['mean_credit_s']
                            decision['candidates'][i]['secondary_credit'] = credit
            scores = [cost-(credit-credits[0]) for cost, credit in zip(values, credits)]
            for i, score in enumerate(scores):
                decision['candidates'][i].update(score_s=score, incremental_credit_s=credits[i]-credits[0])
            best = min(range(len(scores)), key=lambda i: (scores[i], i))
            # A fixed five-second margin avoids tiny proxy gains; no threshold
            # sweep or uncertainty guarantee is claimed for these finite worlds.
            if best and scores[0]-scores[best] >= 5.:
                stats['changed'] += 1
                decision.update(status='changed', selected=decision['candidates'][best]['position'],
                                predicted_gain_s=scores[0]-scores[best])
                return candidates[best]
            decision['selected'] = [baseline.x, baseline.y]
            return baseline
        except _ProxyTimeout:
            stats['fallbacks'].append('compute_budget')
            decision['status'] = 'compute_budget'
            return baseline
        finally:
            elapsed = time.perf_counter()-started
            stats['planning_s'] += elapsed
            decision['planning_s'] = elapsed


def run_probe_cost_search(client, *, problem=3, max_actions=20000, max_active_probes=6,
                          joint_config=None, probe_mode='single', max_planning_s=120.,
                          hypothesis_weighting='equal'):
    if problem not in (3, 'q3'):
        raise ValueError('Explicit first-probe planning is Q3-only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricProbeCostSearch(client, max_actions, max_active_probes, joint_config,
                                    probe_mode, max_planning_s, hypothesis_weighting).run()

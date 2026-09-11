"""Round-three guarded rollout; frozen round-two modules remain unchanged.

Screen on four worlds, accept using ONLY eight confirmation worlds. G2 makes
the coverage shift an explicit proposal from an untouched B action boundary.
Hypothetical continuations always run the frozen B to certified exit.
"""
from dataclasses import asdict, dataclass, is_dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from geometry import distance
from simulator_client.errors import SimulatorError
from strategies.q3_fresh import COVER, open_route
from strategies.q3_fresh_belief import sample_worlds, make_branch, BeliefSamplingError
from strategies.q3_fresh_rollout import FreshRollout
from strategies.q3_fresh_stepper import StepperQ3, Action
from strategies.q3_movable_tail import optimize_tail, certifies_cell


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return sorted(plain(v) for v in value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Proposal:
    label: str
    actions: tuple
    kind: str = 'override'
    original_position: tuple | None = None
    anchor: int | None = None


class Round3Stepper(StepperQ3):
    """B execution layer with a separately evaluated current-cover proposal."""
    def __init__(self, client):
        super().__init__(client, movable_tail=False)

    def cover_proposal(self):
        a = self.pending
        if (a is None or a.kind != 'measure' or a.phase != 'cover' or self.active()
                or not self.unknown() or distance(self.position, a.position) < 1e-6):
            return None
        anchors = self.needed_anchors()
        route = open_route(anchors, self.position)
        if not route or route[0][2] != a.position:
            return None
        # Only the first action of a fresh fixed-point scan is eligible.
        frames = [f for f in self.stack if f[0] == 'scan_unknown' and f[1] == a.position]
        if len(frames) != 1 or frames[0][-1] != 1:
            return None
        optimized = optimize_tail(tuple(item[2] for item in route), self.position)
        q, anchor = tuple(optimized[0]), route[0][1]
        if distance(q, a.position) < 1e-5 or not certifies_cell(q, COVER[anchor]):
            return None
        return Proposal('cover_shift', (Action('measure', q, a.channel, 'cover'),),
                        'cover_shift', a.position, anchor)

    def apply_proposal(self, proposal):
        if proposal.kind == 'baseline':
            return
        if proposal.kind != 'cover_shift':
            self.override(proposal.actions)
            return
        old, q = proposal.original_position, proposal.actions[0].position
        if self.pending is None or self.pending.position != old:
            raise ValueError('Coverage proposal belongs to a different B state')
        if not certifies_cell(q, COVER[proposal.anchor]):
            raise ValueError('Coverage responsibility certificate failed')
        # Retain exactly the scan channel list, visited marker and B main frame.
        # Only this scan's position changes; future cover stops stay fixed B.
        changed = []
        for frame in self.stack:
            if frame[0] in {'scan_unknown', 'scan_known_start', 'scan_known'} and frame[1] == old:
                frame = (frame[0], q, *frame[2:])
            changed.append(frame)
        self.stack = changed
        self.pending = proposal.actions[0]
        self.external_actions += 1


def apply_proposal(policy, proposal):
    if proposal.kind == 'baseline':
        return
    if isinstance(policy, Round3Stepper):
        policy.apply_proposal(proposal)
    else:
        policy.override(proposal.actions)


def controller_summary(policy):
    return dict(pending=plain(policy.pending), stack=plain(policy.stack),
                visited=sorted(policy.visited), movable_tail=policy.movable_tail,
                tail_plan=plain(policy.tail_plan), position=policy.position,
                history_length=len(policy.history), history_sha256=digest(policy.history),
                channels={str(i): dict(status=c.status, local_probes=c.local_probes,
                    negative_count=len(c.negatives), measurement_count=len(c.measurements),
                    region_sha256=digest(c.region.vertices)) for i,c in policy.channels.items()})


def summary_delta(values):
    return dict(mean_s=statistics.mean(values), standard_error_s=statistics.stdev(values)/math.sqrt(len(values)))


class Round3Rollout(FreshRollout):
    def __init__(self, *, evaluate_cover=False, events=False, directions=False,
                 witness_mode='off', log_dir=None, **kwargs):
        super().__init__(allow_attempts=False, **kwargs)
        self.evaluate_cover, self.events, self.directions = evaluate_cover, events, directions
        self.witness_mode = witness_mode
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.stats.update(cover_choices=0, risk_vetoes=0, event_calls=0)
        self.branch_records = []

    def proposals(self, policy):
        old = super().candidates(policy)
        result = [Proposal('baseline', (policy.pending,), 'baseline')]
        if self.evaluate_cover:
            candidate = policy.cover_proposal()
            if candidate is not None:
                result.append(candidate)
        if self.directions:
            result.extend(self.direction_proposals(policy))
        # Reserve room for explicit cover/direction decisions before nearby probes.
        seen = {digest(p.actions) for p in result}
        for index,(label,actions) in enumerate(old[1:],1):
            key = digest(actions)
            if key not in seen:
                result.append(Proposal(f'{label}_{index}', actions))
                seen.add(key)
        return result[:6 if self.evaluate_cover or self.directions else 5]

    def direction_proposals(self, policy):
        return []  # Phase two is introduced only after the G1/G2 ablation.

    def event_reason(self, policy):
        return None

    def risk_check(self, policy, proposal, worlds, deadline, record):
        return True

    def _evaluate_proposal(self, policy, proposal, world, deadline, world_index, candidate_index, stage):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Planning deadline')
        client = make_branch(world, policy.client.state, policy.history)
        fork = policy.clone(client)
        fork.movable_tail, fork.tail_plan = False, []
        before = client.state.snapshot()
        start = client.state.virtual_time_s
        history_size = len(fork.history)
        item = dict(world_index=world_index, candidate_index=candidate_index, stage=stage,
                    world_sha256=digest(world), proposal_sha256=digest(proposal),
                    complete=False, remaining_s=None, error=None)
        self.branch_records.append(item)
        try:
            apply_proposal(fork, proposal)
            result = fork.run(deadline=deadline)
            if not result['completion_certified'] or client.state.session != 'exited':
                raise RuntimeError('Uncertified hypothetical exit')
            item.update(complete=True, remaining_s=client.state.virtual_time_s-start,
                        channels=result['channels'], completion_certified=True)
            self.stats['completed_rollouts'] += 1
            return item['remaining_s']
        except Exception as exc:
            item['error'] = type(exc).__name__
            raise
        finally:
            item.update(before_public_state=before, after_public_state=client.state.snapshot(),
                        future_action_history=plain(fork.history[history_size:]),
                        terminal_session=client.state.session)

    def maybe_choose(self, policy):
        if not self.enabled or len(policy.history) < 20:
            return
        event = self.event_reason(policy) if self.events else None
        if len(policy.history)-self.last_planned_action < 8 and event is None:
            return
        if self.stats['planning_calls'] >= self.max_decisions:
            return
        remaining = self.budget_s-self.stats['planning_wall_s']
        if policy.client.remaining_real_time_s is not None:
            remaining = min(remaining, policy.client.remaining_real_time_s-30)
        if remaining <= .15:
            self.stats['budget_fallbacks'] += 1
            return
        started = time.perf_counter()
        options = self.proposals(policy)
        if len(options) < 2:
            self.stats['planning_wall_s'] += time.perf_counter()-started
            return
        if (event is None and policy.pending.phase in {'cover','opportunistic'} and policy.active()
                and policy.history[-1]['result'] != 'direction'):
            self.stats['planning_wall_s'] += time.perf_counter()-started
            return
        self.last_planned_action = len(policy.history)
        self.stats['planning_calls'] += 1
        if event:
            self.stats['event_calls'] += 1
        deadline = started+min(self.decision_budget_s,remaining)-.1
        record = dict(action_index=len(policy.history), candidates=[p.label for p in options],
                      selected='baseline', selected_index=0, completed=False,
                      event=event, acceptance_world_indices=list(range(4,12)),
                      screening_world_indices=list(range(4)), baseline_is_frozen_B=not policy.movable_tail)
        evidence = dict(schema=3, public_state=policy.client.state.snapshot(),
                        controller=controller_summary(policy), public_history=plain(policy.history),
                        proposals=plain(options), worlds=[], decision=record)
        worlds, best = [], None
        self.branch_records = []
        try:
            seed = self.seed+104729*self.stats['planning_calls']
            worlds = sample_worlds(policy.history, count=12, seed=seed, deadline=deadline)
            evidence.update(sampling_seed=seed, worlds=plain(worlds))
            costs = [[] for _ in options]
            for wi,world in enumerate(worlds[:4]):
                for ci,proposal in enumerate(options):
                    costs[ci].append(self._evaluate_proposal(policy,proposal,world,deadline,wi,ci,'screen'))
            best = min(range(1,len(options)),key=lambda i: statistics.mean(costs[i]))
            for wi,world in enumerate(worlds[4:],4):
                for ci in (0,best):
                    costs[ci].append(self._evaluate_proposal(policy,options[ci],world,deadline,wi,ci,'confirm'))
            delta = [a-b for a,b in zip(costs[best],costs[0])]
            screen, confirm = summary_delta(delta[:4]), summary_delta(delta[4:])
            record.update(completed=True, best_index=best, best_candidate=options[best].label,
                          paired_delta_s=delta, mean_delta_s=statistics.mean(delta),
                          baseline_mean_remaining_s=statistics.mean(costs[0]),
                          candidate_mean_remaining_s=statistics.mean(costs[best]),
                          screening=screen, confirmation=confirm, confirmation_delta_s=delta[4:])
            nominal_accept = confirm['mean_s']+confirm['standard_error_s'] < -3.0
            record['nominal_accept'] = nominal_accept
            risk_accept = self.risk_check(policy,options[best],worlds,deadline,record) if nominal_accept else True
            if nominal_accept and risk_accept:
                apply_proposal(policy,options[best])
                record.update(selected=options[best].label,selected_index=best)
                self.stats['overrides'] += 1
                self.stats['cover_choices'] += options[best].kind == 'cover_shift'
            else:
                self.stats['fallbacks'] += 1
        except BeliefSamplingError as exc:
            record['fallback_reason'] = type(exc).__name__
            self.stats['sampling_failures'] += 1
            self.stats['fallbacks'] += 1
        except TimeoutError:
            record['fallback_reason'] = 'planning_deadline'
            self.stats['budget_fallbacks'] += 1
            self.stats['fallbacks'] += 1
        except (ValueError, RuntimeError, SimulatorError) as exc:
            record['fallback_reason'] = type(exc).__name__
            record['failure_detail'] = str(exc)
            self.stats['continuation_failures'] += 1
            self.stats['fallbacks'] += 1
        finally:
            # Every branch has costs/status; baseline and winning candidate also
            # keep their complete future action histories, including failures.
            evidence['branches'] = [{k:v for k,v in item.items()
                if k != 'future_action_history' or item['candidate_index'] in (0,best)}
                for item in self.branch_records]
            record['wall_s_before_log'] = time.perf_counter()-started
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True,exist_ok=True)
                path = self.log_dir/f"decision-{self.stats['planning_calls']:03d}.json.gz"
                raw = json.dumps(evidence,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                with gzip.open(path,'wb',compresslevel=1) as f:
                    f.write(raw)
                record.update(evidence_file=path.name,evidence_sha256=hashlib.sha256(raw).hexdigest())
            record['wall_s'] = time.perf_counter()-started
            self.stats['planning_wall_s'] += record['wall_s']
            self.stats['decisions'].append(record)
            self.branch_records = []


class OldLoggedRollout(FreshRollout):
    """Frozen CR selection implementation with an observational branch journal.

The inherited candidate generation and mixed-12-world acceptance are intact.
Logging consumes real time, so deadline-sensitive reruns may differ by timing.
"""
    def __init__(self, *, log_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.branch_records, self.worlds, self.options = [], [], []

    def candidates(self, policy):
        options = super().candidates(policy)
        self.options = options
        self.evidence = dict(schema=3, baseline_is_frozen_B=False,
            policy='frozen_CR_with_observation_only_logging',
            public_state=policy.client.state.snapshot(), controller=controller_summary(policy),
            public_history=plain(policy.history),
            proposals=[plain(Proposal(label, actions, 'baseline' if i==0 else 'override'))
                       for i,(label,actions) in enumerate(options)])
        return options

    def _evaluate(self, policy, proposal, world, deadline):
        if world not in self.worlds:
            self.worlds.append(world)
        wi = self.worlds.index(world)
        ci = next(i for i,(_,actions) in enumerate(self.options) if actions==proposal)
        candidate = Proposal(self.options[ci][0],proposal,'baseline' if ci==0 else 'override')
        return Round3Rollout._evaluate_proposal(self,policy,candidate,world,deadline,wi,ci,
                                               'screen' if wi<4 else 'mixed_acceptance')

    def maybe_choose(self, policy):
        before = self.stats['planning_calls']
        self.branch_records, self.worlds = [], []
        super().maybe_choose(policy)
        if self.stats['planning_calls']==before:
            return
        record=self.stats['decisions'][-1]
        started=time.perf_counter()
        counts={i:sum(b['candidate_index']==i for b in self.branch_records) for i in range(len(self.options))}
        best=max(range(1,len(self.options)),key=lambda i:counts[i]) if self.options else None
        self.evidence.update(worlds=plain(self.worlds),decision=record,
            branches=[{k:v for k,v in item.items() if k!='future_action_history' or item['candidate_index'] in (0,best)}
                      for item in self.branch_records])
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            raw=json.dumps(self.evidence,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            path=self.log_dir/f"decision-{self.stats['planning_calls']:03d}.json.gz"
            with gzip.open(path,'wb',compresslevel=1) as f:
                f.write(raw)
            record.update(evidence_file=path.name,evidence_sha256=hashlib.sha256(raw).hexdigest())
        elapsed=time.perf_counter()-started
        self.stats['planning_wall_s']+=elapsed
        record['wall_s']+=elapsed
        self.branch_records,self.worlds=[],[]

"""Bounded one-layer, full-completion rollout over a frozen V1b continuation.

Six proposals, four common worlds, then two survivors get eight more common
worlds. All hypothetical continuations end at their own certified /exit.
Sampling/truncated-continuation failures never become cheap samples.
"""

from dataclasses import asdict
import math
import statistics
import time

from geometry import distance
from simulator_client.errors import SimulatorError
from strategies.q3_fresh import MARGIN, nearest_safe_clear
from strategies.q3_fresh_stepper import Action
from strategies.q3_fresh_belief import sample_worlds, make_branch, BeliefSamplingError


class FreshRollout:
    def __init__(self, *, allow_attempts=False, enabled=True, budget_s=60.0,
                 max_decisions=12, decision_budget_s=6.0, seed=962000):
        self.allow_attempts, self.enabled = allow_attempts, enabled
        self.budget_s, self.max_decisions = budget_s, max_decisions
        self.decision_budget_s, self.seed = decision_budget_s, seed
        self.last_planned_action = -100
        self.stats = dict(planning_calls=0, planning_wall_s=0.0, completed_rollouts=0,
                          overrides=0, fallbacks=0, budget_fallbacks=0,
                          sampling_failures=0, continuation_failures=0,
                          speculative_choices=0, decisions=[])

    def candidates(self, policy):
        base = (policy.pending,)
        candidates = [("baseline", base)]
        seen = {tuple((a.kind,a.position,a.channel,a.speculative) for a in base)}

        def add(label, actions):
            actions = tuple(a for a in actions if policy.channels[a.channel].status != "cleared"
                            and (a.kind != "measure" or a.position not in policy.channels[a.channel].measurements))
            key = tuple((a.kind,a.position,a.channel,a.speculative) for a in actions)
            if actions and key not in seen:
                candidates.append((label,actions))
                seen.add(key)

        active = sorted(policy.active(), key=lambda i: distance(policy.position,
                        policy.channels[i].region.enclosing_disk().center))
        # Reserve one of the six slots for the independent speculative module.
        limit = 5  # RA is R plus one extra candidate, not a changed shortlist.
        for i in active[:2]:
            region = policy.channels[i].region
            circle = region.enclosing_disk()
            if circle.radius <= 20-MARGIN:
                q = nearest_safe_clear(region.vertices, policy.position)
                add("clear_other_source", (Action("clear",q,i,"rollout_clear"),))
            else:
                add("probe_other_source", (Action("measure",circle.center,i,"rollout_probe"),))
        # A bundle shares only travel; each measurement/switch remains charged.
        uncertain = [i for i in active if policy.channels[i].region.enclosing_disk().radius > 20-MARGIN]
        if uncertain:
            add("shared_current", [Action("measure",policy.position,i,"rollout_shared") for i in uncertain[:3]])
        anchors = sorted(policy.needed_anchors(), key=lambda item: distance(policy.position,item[2]))
        if anchors:
            p = anchors[0][2]
            add("unknown_cover_bundle", [Action("measure",p,i,"rollout_cover") for i in policy.unknown()])
        if uncertain:
            i = uncertain[0]
            region = policy.channels[i].region
            center = region.enclosing_disk().center
            obs = region.observations[-1] if region.observations else None
            if obs is not None:
                theta = math.radians(obs.bearing_deg)
                q = (center[0]-75*math.sin(theta),center[1]+75*math.cos(theta))
                if max(distance(q,v) for v in region.vertices) < 1000-MARGIN:
                    add("transverse_probe", (Action("measure",q,i,"rollout_probe"),))
        candidates = candidates[:limit]
        if self.allow_attempts:
            for i in active:
                circle = policy.channels[i].region.enclosing_disk()
                if (20-MARGIN < circle.radius <= 80 and policy.attempts_since_observation[i] < 2
                        and all(distance(circle.center,q)>1e-6 for q in policy.failed_disks[i])):
                    add("attempt_clear", (Action("clear",circle.center,i,"rollout_attempt",True),))
                    break
        return candidates[:6]

    def _evaluate(self, policy, proposal, world, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Planner budget exhausted")
        client = make_branch(world,policy.client.state,policy.history)
        fork = policy.clone(client)
        # The continuation policy is the frozen B (V1b), also in CR/CRA. The
        # currently proposed C action is retained for paired evaluation.
        fork.movable_tail = False
        fork.tail_plan = []
        if proposal != (policy.pending,):
            fork.override(proposal)
        start = client.state.virtual_time_s
        result = fork.run(deadline=deadline)
        if not result["completion_certified"] or client.state.session != "exited":
            raise RuntimeError("Incomplete rollout cannot be scored")
        self.stats["completed_rollouts"] += 1
        return client.state.virtual_time_s-start

    def maybe_choose(self, policy):
        if not self.enabled or len(policy.history)<20:
            return
        # Each API boundary remains interruptible. Expensive replanning is
        # spaced out; ordinary boundaries continue the exact frozen schedule.
        if len(policy.history)-self.last_planned_action < 8:
            return
        if self.stats["planning_calls"] >= self.max_decisions:
            return
        remaining = self.budget_s-self.stats["planning_wall_s"]
        real_remaining = policy.client.remaining_real_time_s
        if real_remaining is not None:
            remaining = min(remaining,real_remaining-30.0)
        if remaining <= 0.02:
            self.stats["budget_fallbacks"] += 1
            return
        started = time.perf_counter()
        options = self.candidates(policy)
        if len(options)<2:
            self.stats["planning_wall_s"] += time.perf_counter()-started
            return
        # Avoid spending all slots halfway through a routine spectrum sweep.
        if policy.pending.phase in {"cover","opportunistic"} and policy.active():
            if policy.history[-1]["result"] != "direction":
                self.stats["planning_wall_s"] += time.perf_counter()-started
                return
        self.last_planned_action = len(policy.history)
        self.stats["planning_calls"] += 1
        deadline = started+min(self.decision_budget_s,remaining)
        record = dict(action_index=len(policy.history), candidates=[label for label,_ in options],
                      selected="baseline", completed=False)
        try:
            # One common pool, same fixed errors across every candidate.
            worlds = sample_worlds(policy.history,count=12,
                seed=self.seed+104729*self.stats["planning_calls"],deadline=deadline)
            costs = [[] for _ in options]
            for world in worlds[:4]:
                for index, (_, proposal) in enumerate(options):
                    costs[index].append(self._evaluate(policy,proposal,world,deadline))
            # Baseline is always included in stage two, so a final choice has
            # twelve paired completed worlds, even when baseline ranks third.
            best = min(range(1,len(options)),key=lambda i: statistics.mean(costs[i]))
            for world in worlds[4:]:
                for index in (0,best):
                    costs[index].append(self._evaluate(policy,options[index][1],world,deadline))
            delta=[a-b for a,b in zip(costs[best],costs[0])]
            mean=statistics.mean(delta)
            se=statistics.stdev(delta)/math.sqrt(len(delta)) if len(delta)>1 else math.inf
            record.update(completed=True, best_candidate=options[best][0], paired_delta_s=delta,
                          mean_delta_s=mean, standard_error_s=se,
                          baseline_mean_remaining_s=statistics.mean(costs[0]),
                          candidate_mean_remaining_s=statistics.mean(costs[best]))
            # Modest engineering guard, not a distributional guarantee.
            if mean+se < -3.0:
                policy.override(options[best][1])
                record["selected"] = options[best][0]
                self.stats["overrides"] += 1
                if policy.pending.speculative:
                    self.stats["speculative_choices"] += 1
            else:
                self.stats["fallbacks"] += 1
        except BeliefSamplingError as exc:
            record["fallback_reason"] = type(exc).__name__
            self.stats["sampling_failures"] += 1
            self.stats["fallbacks"] += 1
        except TimeoutError:
            record["fallback_reason"] = "planning_deadline"
            self.stats["budget_fallbacks"] += 1
            self.stats["fallbacks"] += 1
        except (ValueError, RuntimeError, SimulatorError) as exc:
            # An inconsistent hypothetical branch is a modeling failure, never
            # evidence about whether the real source exists.
            record["fallback_reason"] = type(exc).__name__
            record["failure_detail"] = str(exc)
            self.stats["continuation_failures"] += 1
            self.stats["fallbacks"] += 1
        finally:
            elapsed=time.perf_counter()-started
            self.stats["planning_wall_s"] += elapsed
            record["wall_s"] = elapsed
            self.stats["decisions"].append(record)

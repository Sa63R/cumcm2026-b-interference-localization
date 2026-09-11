"""Serializable action-boundary continuation of the frozen V1b controller.

The stack contains data, not generators/callbacks: a hypothetical client can
resume a deep copy without sharing the real client or replaying charged work.
"""

import copy
from dataclasses import dataclass
import math

from geometry import clip_polygon, disk_halfplanes, distance, point
from strategies.q3_fresh import COVER, MARGIN, FreshQ3, covered, nearest_safe_clear, open_route


@dataclass(frozen=True)
class Action:
    kind: str
    position: tuple
    channel: int
    phase: str
    speculative: bool = False


class StepperQ3(FreshQ3):
    def __init__(self, client, *, movable_tail=False):
        super().__init__(client, "v1", selective=True)
        self.stack = [("main", 0), ("visited", 0), ("scan", COVER[0], "cover", False)]
        self.pending = None
        self.movable_tail = movable_tail
        self.tail_plan = []
        self.tail_optimizations = []
        self.failed_disks = {c: [] for c in self.channels}
        self.attempts_since_observation = {c: 0 for c in self.channels}
        self.external_actions = 0

    def clone(self, client):
        other = object.__new__(type(self))
        other.__dict__ = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "client"}
        other.client = client
        return other

    def _scan_unknown(self, p, opportunistic):
        todo = list(self.unknown())
        if opportunistic:
            todo = [i for i in todo if not self.channels[i].negatives or
                    min(distance(p, n) for n in self.channels[i].negatives) >= 350]
            todo = [i for i in todo if any(
                not covered(tuple(self.channels[i].negatives), a, True) and
                covered(tuple(self.channels[i].negatives) + (p,), a, True) for a in COVER)]
        return todo

    def _shared_known_worthwhile(self, p, i):
        c = self.channels[i]
        if p in c.measurements:
            return False
        circle = c.region.enclosing_disk()
        if circle.radius <= 20 - MARGIN:
            return False
        if c.region.observations and min(distance(p, o.position) for o in c.region.observations) < 25:
            return False
        d = distance(p, circle.center)
        if d > 1500 + circle.radius:
            return False
        if d > 5:
            bearing = math.degrees(math.atan2(circle.center[1]-p[1], circle.center[0]-p[0])) % 360
            predicted = c.region.copy().observe(p, bearing)
            if predicted.vertices:
                r = predicted.enclosing_disk().radius
                if r > 20 - MARGIN and circle.radius - r < 30:
                    return False
        return True

    def _set_pending(self, action):
        if action.kind == "measure" and action.position in self.channels[action.channel].measurements:
            return None
        if self.channels[action.channel].status == "cleared":
            return None
        self.pending = action
        return action

    def next_action(self):
        if self.pending is not None:
            return self.pending
        while self.stack:
            task = self.stack.pop()
            op = task[0]
            if op == "visited":
                self.visited.add(task[1])
            elif op == "scan":
                _, p, phase, opportunistic = task
                self.stack.append(("scan_known_start", p, phase))
                self.stack.append(("scan_unknown", p, phase, self._scan_unknown(p, opportunistic), 0))
            elif op == "scan_unknown":
                _, p, phase, todo, index = task
                if index < len(todo):
                    i = todo[index]
                    self.stack.append(("scan_unknown", p, phase, todo, index+1))
                    self.stack.append(("count_bound",))
                    if self.channels[i].status == "unknown":
                        a = self._set_pending(Action("measure", p, i, phase))
                        if a is not None:
                            return a
            elif op == "count_bound":
                self.unknown()
            elif op == "scan_known_start":
                self.stack.append(("scan_known", task[1], task[2], self.active(), 0))
            elif op == "scan_known":
                _, p, phase, todo, index = task
                if index < len(todo):
                    i = todo[index]
                    self.stack.append(("scan_known", p, phase, todo, index+1))
                    if self.channels[i].status == "detected" and self._shared_known_worthwhile(p, i):
                        return self._set_pending(Action("measure", p, i, phase))
            elif op == "after_local":
                _, i, contraction, p = task
                c = self.channels[i]
                c.local_probes += 1
                if contraction is not None:
                    for hp in disk_halfplanes(p, contraction, 64, outer=True):
                        c.region.vertices = clip_polygon(c.region.vertices, hp)
                    c.region._circle = None
                    if not c.region.vertices:
                        raise RuntimeError("Inconsistent halving bound")
            elif op == "locate":
                _, i, iteration = task
                if self.channels[i].status == "cleared":
                    continue
                if iteration >= 18:
                    raise RuntimeError("Localizer exceeded its safety budget")
                c = self.channels[i]
                circle = c.region.enclosing_disk()
                if circle.radius <= 20 - MARGIN:
                    q = nearest_safe_clear(c.region.vertices, self.position)
                    self.stack.append(("scan", q, "opportunistic", True))
                    return self._set_pending(Action("clear", q, i, "clear"))
                q, contraction = circle.center, None
                if c.local_probes >= 2 or q in c.measurements:
                    obs = c.region.observations[-1]
                    radius = min(1500.0, max(distance(obs.position, v) for v in c.region.vertices))
                    step = radius / (2 * math.cos(math.radians(c.region.error_deg)))
                    theta = math.radians(obs.bearing_deg)
                    q = (obs.position[0] + step * math.cos(theta), obs.position[1] + step * math.sin(theta))
                    contraction = step + MARGIN
                    if contraction <= 20 - MARGIN:
                        self.stack.append(("scan", q, "opportunistic", True))
                        return self._set_pending(Action("clear", q, i, "halving_clear"))
                self.stack += [("locate", i, iteration+1), ("scan", q, "opportunistic", True),
                               ("after_local", i, contraction, q)]
                action = self._set_pending(Action("measure", q, i, "localize"))
                if action is not None:
                    return action
            elif op == "injected":
                action = self._set_pending(task[1])
                if action is not None:
                    return action
            elif op == "main":
                iteration = task[1]
                if iteration >= 200:
                    raise RuntimeError("Mixed route exceeded its safety budget")
                anchors = self.needed_anchors()
                active = self.active()
                items = anchors + [("source", i, self.channels[i].region.enclosing_disk().center) for i in active]
                if not items:
                    continue
                route = open_route(items, self.position)
                kind, index, p = route[0]
                if active:
                    self.tail_plan = []
                elif self.movable_tail and self.unknown():
                    if not self.tail_plan:
                        from strategies.q3_movable_tail import optimize_tail
                        original = tuple(item[2] for item in route)
                        optimized = optimize_tail(original, self.position)
                        self.tail_plan = [(item[1], q) for item, q in zip(route, optimized)]
                        self.tail_optimizations.append(dict(start=self.position, original=original,
                                                           optimized=optimized, order=[r[1] for r in route]))
                    index, p = self.tail_plan.pop(0)
                    kind = "cover"
                self.route_decisions.append(dict(position=self.position, route=route, executed=(kind,index,p)))
                self.stack.append(("main", iteration+1))
                if kind == "cover":
                    self.stack += [("visited", index), ("scan", p, "cover", False)]
                else:
                    self.stack.append(("locate", index, 0))
            else:
                raise RuntimeError("Unknown controller frame")
        self.certified = not self.unknown() and not self.active()
        if not self.certified:
            raise RuntimeError("Missing discovery/clearance certificate")
        return None

    def override(self, actions):
        """Replace a proposal, preserving observations but releasing task lock.

        A channel bundle is interruptible between each actual API action.
        After the bundle, the observation-only baseline replans its route.
        """
        if not actions:
            raise ValueError("An action or bundle is required")
        self.external_actions += 1
        self.pending = actions[0]
        self.stack = [("main", 0)] + [("injected", a) for a in reversed(actions[1:])]
        self.tail_plan = []

    def execute_pending(self):
        a = self.pending
        if a is None:
            raise RuntimeError("No pending action")
        if a.kind == "measure":
            FreshQ3.measure(self, a.position, a.channel, a.phase)
            self.attempts_since_observation[a.channel] = 0
        elif not a.speculative:
            FreshQ3.clear(self, a.position, a.channel, a.phase)
        else:
            c = self.channels[a.channel]
            if c.status != "detected" or self.attempts_since_observation[a.channel] >= 2:
                raise RuntimeError("Speculative clear guard violated")
            response = self.client.clear(a.position, a.channel)
            result = self._record("clear", a.position, a.channel, response, a.phase)
            self.attempts_since_observation[a.channel] += 1
            if result == "success":
                c.status = "cleared"
                self.last_clear_time = self.client.state.virtual_time_s
            else:
                self.failed_disks[a.channel].append(a.position)
        self.pending = None

    def result(self):
        return dict(version="v1b_stepper", completion_certified=self.certified,
                    channels={i:c.status for i,c in self.channels.items()},
                    action_history=self.history, route_decisions=self.route_decisions,
                    visited_cover_indices=sorted(self.visited),
                    confirmation_tail_s=self.client.state.virtual_time_s-self.last_clear_time,
                    tail_optimizations=self.tail_optimizations, external_actions=self.external_actions)

    def run(self, planner=None, *, deadline=None):
        import time
        if self.client.state.session == "new":
            self.client.enter()
        for _ in range(1600):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Continuation deadline exceeded")
            if self.next_action() is None:
                self.client.exit()
                return self.result()
            if planner is not None:
                planner.maybe_choose(self)
            self.execute_pending()
        raise RuntimeError("Action budget exceeded")

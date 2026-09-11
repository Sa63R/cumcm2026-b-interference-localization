"""Composable Q4 whole-station proposals from public service positions only.

This module neither executes actions nor predicts measurement responses. Its
public plan-cost proxy is not a value function or a performance guarantee.
Prospective certificates never grant actual observation/absence credit.
"""
from dataclasses import dataclass
from itertools import combinations
import math
import time

from simulator_client.state import Position
from localization import CandidateRegion
from .adaptive_cover import (VERSION as EXACT_VERSION, certify, position_key,
                             verify_exact_certificate, replay_adaptive_artifact, _digest)


VERSION = "q4-shared-service-station-v1"


@dataclass(frozen=True)
class ServicePoint:
    position: Position
    source_channel: int
    origin: str

    def __post_init__(self):
        if not isinstance(self.position,Position) or type(self.source_channel) is not int or not 1<=self.source_channel<=20:
            raise ValueError("Invalid service point")
        if self.origin not in ("actual_near","positive_outer_center","positive_transverse_probe"):
            raise ValueError("Service point needs a public construction")


@dataclass(frozen=True)
class PlannedAction:
    kind: str
    position: Position
    channel: int
    purpose: str = "service"

    def __post_init__(self):
        if self.kind not in ("measure", "clear") or type(self.channel) is not int or not 1<=self.channel<=20:
            raise ValueError("Invalid public planned action")
        if not isinstance(self.position,Position):
            raise ValueError("Planned coordinates must be validated Position objects")
        if self.purpose not in ("service", "coverage", "shared_scan"):
            raise ValueError("Unknown planned action purpose")


@dataclass(frozen=True)
class SharedPlan:
    station: Position
    service_points: tuple
    channels: tuple
    actions: tuple
    before_cost: dict
    after_cost: dict
    estimated_saved_s: float
    certificate: dict
    proof_root: str


def public_service_points(regions, near_points, current, *, cleared=(), maximum=4):
    """Derive <=4 finite positions from public positive regions/near receipts.

    ``regions`` and ``near_points`` are the controller's observation-derived
    maps, not a simulator or scenario. A future center need not receive signal;
    it is only a legal position at which an actual request could be submitted.
    """
    if type(maximum) is not int or not 1<=maximum<=4:
        raise ValueError("At most four public service positions")
    current = Position.coerce(current)
    candidates = []
    for channel in sorted(set(regions)|set(near_points)):
        if channel in cleared:
            continue
        if type(channel) is not int or not 1<=channel<=20:
            raise ValueError("Invalid observed source channel")
        if channel in near_points:
            candidates.append(ServicePoint(Position.coerce(near_points[channel]),channel,"actual_near"))
            continue
        region = regions[channel]
        if not region.vertices or not region.observations:
            continue
        center = Position.coerce(region.enclosing_disk().center)
        candidates.append(ServicePoint(center,channel,"positive_outer_center"))
        angle = math.radians(region.observations[0].bearing_deg)
        step = min(180.,max(25.,region.enclosing_disk().radius*.5))
        for sign in (1.,-1.):
            candidates.append(ServicePoint(Position(center.x-sign*step*math.sin(angle),
                                                     center.y+sign*step*math.cos(angle)),
                                           channel,"positive_transverse_probe"))
    # Preserve source diversity before filling spare slots with probe choices.
    candidates.sort(key=lambda p:(current.distance_to(p.position),p.source_channel,p.origin))
    selected,channels,points = [],set(),set()
    for source_diversity in (True,False):
        for candidate in candidates:
            if candidate.position in points or (source_diversity and candidate.source_channel in channels):
                continue
            selected.append(candidate)
            channels.add(candidate.source_channel)
            points.add(candidate.position)
            if len(selected)==maximum:
                return tuple(selected)
    return tuple(selected)


def public_plan_cost(actions, start, current_channel):
    """Count every specified move, detection, switch and optical upper cost.

    Clear success costs five seconds, failure three; using five is a public
    upper bound for each retained clear. No unknown future response is read.
    """
    start = Position.coerce(start)
    if type(current_channel) is not int or not 1<=current_channel<=20:
        raise ValueError("Invalid current channel")
    result = dict(movement_s=0.,detection_s=0.,switching_s=0.,clear_upper_s=0.)
    current,tuned = start,current_channel
    for action in actions:
        if not isinstance(action,PlannedAction):
            raise ValueError("Explicit public action plan required")
        result["movement_s"] += math.ceil(current.distance_to(action.position)/5.*1e6)/1e6
        if action.kind=="measure":
            result["detection_s"] += 5.
            result["switching_s"] += int(tuned!=action.channel)
            tuned = action.channel
        else:
            result["clear_upper_s"] += 5.
        current = action.position
    result["total_upper_s"] = sum(result.values())
    result["action_count"] = len(actions)
    return result


class SharedCoverLedger:
    def __init__(self, fixed_points, *, max_certificate_queries=8, query_wall_s=.5, max_cells=20000):
        if type(max_certificate_queries) is not int or not 1<=max_certificate_queries<=8:
            raise ValueError("At most eight expensive certificate queries per episode")
        if not math.isfinite(query_wall_s) or not 0<query_wall_s<=2.:
            raise ValueError("Invalid query time cap")
        if type(max_cells) is not int or not 1<=max_cells<=200000:
            raise ValueError("Invalid cell budget")
        self.fixed = frozenset(position_key(p) for p in fixed_points)
        self.max_queries,self.query_wall_s,self.max_cells = max_certificate_queries,query_wall_s,max_cells
        self.certificate_queries,self.certificate_wall_s = 1,0.
        before = time.perf_counter()
        base,diagnostic = certify(self.fixed,max_cells=max_cells)
        self.certificate_wall_s += time.perf_counter()-before
        if base is None:
            raise ValueError("No exact initial cover: "+diagnostic["status"])
        self.base = base
        root = _digest(base)
        self.proofs,self.roots = {root:base},{c:root for c in range(1,21)}
        self.pending = {c:set(self.fixed) for c in range(1,21)}
        self.negatives = {c:set() for c in range(1,21)}
        self.known,self.cleared = set(),set()
        self.regions,self.near_points = {},{}
        self.history,self.events,self.transactions,self.attempts = [],[],[],[]
        self.cancelled_stations = set()

    def unknown(self):
        return set(range(1,21))-self.known-self.cleared

    def observe(self, action_index, action, point, channel, result, *, accepted, bearing_deg=None):
        if accepted is not True:
            return False
        if type(action_index) is not int or action_index!=len(self.history):
            raise ValueError("Consume every accepted physical action in exact order")
        if type(channel) is not int or not 1<=channel<=20:
            raise ValueError("Invalid observed channel")
        legal = {"measure":{"near","direction","no_signal"},"clear":{"success","no_target_in_range"}}
        if action not in legal or result not in legal[action]:
            raise ValueError("Invalid accepted public response")
        if result=="direction" and (bearing_deg is None or not math.isfinite(bearing_deg) or not 0<=bearing_deg<360):
            raise ValueError("A direction response needs its actual public bearing")
        point = position_key(point)
        self.history.append(dict(action=action,position=list(point),channel=channel,result=result))
        if action=="clear":
            if result=="success":
                self.cleared.add(channel)
        elif result in ("near","direction"):
            self.known.add(channel)
            if result=="near":
                self.near_points[channel] = Position(*point)
            else:
                self.history[-1]["bearing_deg"] = bearing_deg
                self.regions.setdefault(channel,CandidateRegion()).observe(point,bearing_deg)
        elif channel in self.unknown():
            self.negatives[channel].add(point)
            self.pending[channel].discard(point)
        return True

    def _replacement_actions(self, station, points, channels, baseline, current):
        station_key = position_key(station)
        kept = tuple(a for a in baseline if not (a.kind=="measure" and a.purpose=="coverage"
                    and position_key(a.position)==station_key and a.channel in channels))
        # Known-source/localization requirements at the old station prevent
        # calling this a cancelled station even if unknown pairs can be removed.
        if any(a.position==station for a in kept):
            return None
        scheduled = {(position_key(a.position),a.channel) for a in kept if a.kind=="measure"}
        by_position = {}
        for point in points:
            key = position_key(point.position)
            if point.position!=current and not any(a.position==point.position and a.purpose=="service" for a in kept):
                return None  # No free invented service visit in the cost proxy.
            by_position[key] = [c for c in channels if key not in self.negatives[c] and (key,c) not in scheduled]
        output = []
        tuned = self._proxy_channel

        def insert(position):
            nonlocal tuned
            key = position_key(position)
            if key not in by_position:
                return
            remaining = by_position.pop(key)
            for channel in sorted(remaining,key=lambda c:(c!=tuned,c)):
                output.append(PlannedAction("measure",position,channel,"shared_scan"))
                tuned = channel

        insert(current)
        for action in kept:
            output.append(action)
            if action.kind=="measure":
                tuned = action.channel
            insert(action.position)
        return tuple(output)

    def service_points(self,current):
        return public_service_points(self.regions,self.near_points,current,cleared=self.cleared)

    def proposals(self, service_points, baseline_actions, current, current_channel, *, target_station=None):
        """Return cost-positive, exactly certified whole-inner-station plans.

        All retained unknown-channel obligations must occur in the supplied
        public plan; otherwise a short-horizon proxy could hide future costs.
        New service scans are charged at real planned visits, never teleporting.
        """
        if len(service_points)>4 or any(not isinstance(p,ServicePoint) for p in service_points):
            raise ValueError("Use at most four derived public ServicePoint values")
        if any(p.source_channel not in self.known-self.cleared for p in service_points):
            raise ValueError("Future positions must belong to actually known live channels")
        if len({p.position for p in service_points})!=len(service_points):
            raise ValueError("Duplicate public service positions")
        baseline = tuple(baseline_actions)
        current = Position.coerce(current)
        if not set(service_points)<=set(self.service_points(current)):
            raise ValueError("Service coordinates do not match the accepted public history")
        before = public_plan_cost(baseline,current,current_channel)
        self._proxy_channel = current_channel
        unknown = self.unknown()
        for c in unknown:
            available = {position_key(a.position) for a in baseline if a.kind=="measure" and a.channel==c}
            if not self.pending[c]<=available:
                raise ValueError("Proxy plan omits a retained discovery obligation")
        visited = {tuple(a["position"]) for a in self.history}|{position_key(current)}
        cheap = []
        inner = [Position(*p) for p in self.fixed if abs(math.hypot(*p)-970.)<=1e-6 and p not in visited]
        if target_station is not None:
            target_station = Position.coerce(target_station)
            if target_station not in inner:
                return ()
            inner = [target_station]
        for station in sorted(inner,key=lambda p:(p.x,p.y)):
            owed = tuple(sorted(c for c in unknown if position_key(station) in self.pending[c]))
            if not owed:
                continue
            for count in (1,2):
                for points in combinations(service_points,count):
                    if any(p.position==station for p in points):
                        continue
                    actions = self._replacement_actions(station,points,owed,baseline,current)
                    if actions is None:
                        continue
                    after = public_plan_cost(actions,current,current_channel)
                    saving = before["total_upper_s"]-after["total_upper_s"]
                    if saving>1e-6:
                        cheap.append((saving,station,points,owed,actions,after))
        cheap.sort(key=lambda x:(-x[0],x[1].x,x[1].y,tuple(position_key(p.position) for p in x[2])))
        result = []
        for saving,station,points,owed,actions,after in cheap:
            if self.certificate_queries>=self.max_queries-1:
                break
            common = set.intersection(*(self.negatives[c]|(self.pending[c]-{position_key(station)}) for c in owed))
            prospective = common|{position_key(p.position) for p in points}
            self.certificate_queries += 1
            started = time.perf_counter()
            proof,diagnostic = certify(prospective,max_cells=self.max_cells,
                deadline=started+self.query_wall_s,template=self.base)
            self.certificate_wall_s += time.perf_counter()-started
            self.attempts.append(dict(station=list(position_key(station)),
                service_points=[list(position_key(p.position)) for p in points],
                estimated_saved_s=saving,diagnostic=diagnostic,actual_credit=False))
            if proof is not None:
                result.append(SharedPlan(station,tuple(points),owed,actions,before,after,saving,proof,_digest(proof)))
        return tuple(result)

    def cancel_station(self, plan):
        """Atomic station cancellation only after each needed real channel scan.

        A prospective positive response is not assumed: only channels already
        known/cleared in the accepted history may leave the unknown packet.
        """
        if not isinstance(plan,SharedPlan) or _digest(plan.certificate)!=plan.proof_root:
            raise ValueError("Damaged shared plan certificate")
        station = position_key(plan.station)
        if station not in self.fixed or abs(math.hypot(*station)-970.)>1e-6 or not 1<=len(plan.service_points)<=2:
            raise ValueError("Only one inner station and one or two public service points")
        if station in {tuple(a["position"]) for a in self.history}:
            return False  # This prototype only cancels not-yet-visited stations.
        unknown = self.unknown()
        owed = sorted(c for c in unknown if station in self.pending[c])
        if not owed or not set(owed)<=set(plan.channels):
            return False
        support = {position_key(p) for p in plan.certificate["points"]}
        for c in owed:
            if any(position_key(p.position) not in self.negatives[c] for p in plan.service_points):
                return False
            if not support<=self.negatives[c]|(self.pending[c]-{station}):
                return False
        if self.certificate_queries>=self.max_queries:
            return False
        self.certificate_queries += 1
        started = time.perf_counter()
        try:
            verify_exact_certificate(plan.certificate)
        finally:
            self.certificate_wall_s += time.perf_counter()-started
        # No mutation before all channels have satisfied the same transaction.
        self.proofs[plan.proof_root] = plan.certificate
        event_start = len(self.events)
        for c in owed:
            previous = sorted(self.pending[c])
            self.pending[c].remove(station)
            self.roots[c] = plan.proof_root
            self.events.append(dict(after_action_index=len(self.history)-1,channel=c,
                removed=list(station),previous_pending=previous,pending=sorted(self.pending[c]),
                proof_root=plan.proof_root,proof_kind="residual_plan"))
        self.cancelled_stations.add(station)
        self.transactions.append(dict(after_action_index=len(self.history)-1,
            station=list(station),channels=owed,proof_root=plan.proof_root,
            service_points=[list(position_key(p.position)) for p in plan.service_points],
            pair_event_start=event_start,pair_event_end=len(self.events),
            estimated_saved_s=plan.estimated_saved_s,actual_credit_only=True))
        return True

    def artifact(self):
        absent = {c for c in self.unknown() if not self.pending[c]}
        visited = {tuple(a["position"]) for a in self.history}
        return dict(version=EXACT_VERSION,shared_version=VERSION,fixed_points=sorted(self.fixed),
            proofs=self.proofs,roots={str(c):r for c,r in self.roots.items()},events=self.events,
            pending={str(c):sorted(p) for c,p in self.pending.items()},certified_absent=sorted(absent),
            station_transactions=self.transactions,cancelled_station_count=len(self.cancelled_stations),
            cancelled_stations_not_actually_visited=sorted(self.cancelled_stations-visited),
            cost=dict(certificate_queries_including_initial=self.certificate_queries,
                      max_certificate_queries=self.max_queries,certificate_wall_s=self.certificate_wall_s),
            proposal_attempts=self.attempts)


def replay_shared_cover(action_history, artifact):
    """Reuse unchanged exact proof/provenance replay, then check station groups."""
    if artifact.get("shared_version")!=VERSION:
        raise ValueError("Unknown shared ledger version")
    result = replay_adaptive_artifact(action_history,artifact)
    fixed = {position_key(p) for p in artifact["fixed_points"]}
    pending = {c:set(fixed) for c in range(1,21)}
    known,visited,cancelled = set(),set(),set()
    action_cursor,event_cursor = 0,0
    for transaction in artifact["station_transactions"]:
        prefix = transaction["after_action_index"]
        if type(prefix) is not int or not action_cursor-1<=prefix<len(action_history):
            raise ValueError("Invalid or unordered station transaction prefix")
        for item in action_history[action_cursor:prefix+1]:
            c,point = item["channel"],position_key(item["position"])
            visited.add(point)
            if (item["action"]=="measure" and item["result"] in ("near","direction")) or (item["action"]=="clear" and item["result"]=="success"):
                known.add(c)
            elif item["action"]=="measure" and item["result"]=="no_signal" and c not in known:
                pending[c].discard(point)
        action_cursor = prefix+1
        preceding = action_history[:prefix+1]
        station = tuple(transaction["station"])
        if station not in fixed or abs(math.hypot(*station)-970.)>1e-6 or station in cancelled:
            raise ValueError("Invalid or duplicate whole inner station")
        if station in visited:
            raise ValueError("Cancelled station was already visited")
        channels = transaction["channels"]
        owed = {c for c in range(1,21) if c not in known and station in pending[c]}
        if not owed or set(channels)!=owed or len(channels)!=len(owed):
            raise ValueError("Partial channel deletion is not whole-station cancellation")
        if not 1<=len(transaction["service_points"])<=2 or transaction.get("actual_credit_only") is not True:
            raise ValueError("Invalid service scan transaction")
        if transaction["pair_event_start"]!=event_cursor or transaction["pair_event_end"]!=event_cursor+len(owed):
            raise ValueError("Station groups must partition pair events in order")
        events = artifact["events"][transaction["pair_event_start"]:transaction["pair_event_end"]]
        if {e["channel"] for e in events}!=set(transaction["channels"]):
            raise ValueError("Station/pair transaction differs")
        for c in transaction["channels"]:
            if c in known:
                raise ValueError("Deletion transaction cites a non-unknown channel")
            negatives = {tuple(a["position"]) for a in preceding if a["action"]=="measure"
                         and a["channel"]==c and a["result"]=="no_signal"}
            if not {tuple(p) for p in transaction["service_points"]}<=negatives:
                raise ValueError("Future or wrong-channel service scan claimed as actual")
        for e in events:
            if e["removed"]!=transaction["station"] or e["proof_root"]!=transaction["proof_root"] or e["after_action_index"]!=prefix:
                raise ValueError("Station cancellation is not atomic")
            pending[e["channel"]].remove(station)
        cancelled.add(station)
        event_cursor += len(events)
    all_visited = {position_key(a["position"]) for a in action_history}
    if event_cursor!=len(artifact["events"]) or artifact["cancelled_station_count"]!=len(cancelled):
        raise ValueError("Unaccounted pair deletions or inflated station count")
    if {position_key(p) for p in artifact["cancelled_stations_not_actually_visited"]}!=cancelled-all_visited:
        raise ValueError("Claimed skipped stations differ from actual route")
    result.update(cancelled_station_count=len(cancelled),
                  shared_transactions_verified=len(artifact["station_transactions"]))
    return result

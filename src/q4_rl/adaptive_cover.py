"""Replayable Q4 cover obligations; planned points never certify execution.

Floating geometry searches for a dyadic witness. Every accepted witness is
independently checked with exact rational arithmetic on submitted float values.
Search failure or budget exhaustion preserves the previous obligations.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import time

from planning.q4_directional_cover import (certify_directional_cover, _children, _outside,
                                           _support, _separating_witness)
from simulator_client.state import Position


VERSION = "q4-adaptive-exact-cover-v1"
RANGE_MARGIN = 1e-5
ORIENTATION_MARGIN = 1e-7
_CERT_CACHE = {}


class CoverBudgetExpired(Exception):
    pass


def position_key(value):
    point = Position.coerce(value)
    return float(point.x), float(point.y)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points
    lo, hi = [], []
    for point in points:
        while len(lo) >= 2 and _cross(lo[-2], lo[-1], point) <= 0:
            lo.pop()
        lo.append(point)
    for point in reversed(points):
        while len(hi) >= 2 and _cross(hi[-2], hi[-1], point) <= 0:
            hi.pop()
        hi.append(point)
    return lo[:-1]+hi[:-1]


@lru_cache(maxsize=32768)
def _box(path):
    if not isinstance(path, str) or len(path) > 20 or any(c not in "0123" for c in path):
        raise ValueError("Invalid dyadic path")
    x0, x1, y0, y1 = map(Fraction, (-1800, 1800, -1800, 1800))
    for code in path:
        x, y = (x0+x1)/2, (y0+y1)/2
        x0, x1, y0, y1 = ((x0, x, y0, y), (x, x1, y0, y),
                          (x0, x, y, y1), (x, x1, y, y1))[int(code)]
    return x0, x1, y0, y1


@lru_cache(maxsize=65536)
def _verify_leaf(path, support):
    """Cache geometry only; channel-specific observation provenance is separate."""
    x0, x1, y0, y1 = _box(path)
    if support is None:
        x, y = max(x0, min(Fraction(0), x1)), max(y0, min(Fraction(0), y1))
        if x*x+y*y <= Fraction(1800)**2:
            raise ValueError("Outside leaf intersects target disk")
        return True
    points = [tuple(Fraction(v) for v in point) for point in support]
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    limit_squared = (Fraction(1000)-Fraction(RANGE_MARGIN))**2
    if any(sum((a-b)**2 for a, b in zip(p, q)) > limit_squared
           for p in points for q in corners):
        raise ValueError("Support point outside guaranteed reception radius")
    hull = _hull(points)
    if len(hull) < 3:
        raise ValueError("Degenerate support hull")
    margin_squared = Fraction(ORIENTATION_MARGIN)**2
    for a, b in zip(hull, hull[1:]+hull[:1]):
        edge_squared = sum((v-w)**2 for v, w in zip(a, b))
        for q in corners:
            cross = _cross(a, b, q)
            if cross <= 0 or cross*cross < margin_squared*edge_squared:
                raise ValueError("Support hull does not conservatively contain box")
    return True


def verify_exact_certificate(certificate, deadline=None):
    """Independently validate partition and geometry; does not infer provenance."""
    if certificate.get("version") != VERSION:
        raise ValueError("Unknown certificate version")
    points = tuple(position_key(point) for point in certificate["points"])
    if points != tuple(sorted(set(points))):
        raise ValueError("Certificate points must be distinct and sorted")
    tree = {}
    for leaf in certificate["leaves"]:
        if deadline is not None and time.perf_counter() >= deadline:
            raise CoverBudgetExpired()
        path = leaf["path"]
        _box(path)
        node = tree
        for code in path:
            if "leaf" in node:
                raise ValueError("Overlapping coverage leaves")
            node = node.setdefault(code, {})
        if node:
            raise ValueError("Duplicate or overlapping coverage leaf")
        node["leaf"] = True
        if leaf["kind"] == "outside":
            _verify_leaf(path, None)
        elif leaf["kind"] == "covered":
            indices = leaf["stations"]
            if not indices or any(type(i) is not int or not 0 <= i < len(points) for i in indices):
                raise ValueError("Invalid support point ID")
            _verify_leaf(path, tuple(sorted(points[i] for i in indices)))
        else:
            raise ValueError("Unknown coverage leaf type")
    def complete(node):
        if "leaf" in node:
            return len(node) == 1
        return set(node) == set("0123") and all(complete(node[c]) for c in "0123")
    if not complete(tree):
        raise ValueError("Incomplete dyadic partition")
    return True


def _search_from_template(points, template, max_cells, deadline):
    """Reuse still-valid support leaves; refine only affected dyadic boxes."""
    old = [position_key(p) for p in template["points"]]
    indices = {p: i for i, p in enumerate(points)}
    leaves, pending = [], []
    examined, reused = 0, 0
    for leaf in template["leaves"]:
        if deadline is not None and time.perf_counter() >= deadline:
            raise CoverBudgetExpired()
        if leaf["kind"] == "outside":
            leaves.append(dict(leaf))
            reused += 1
            continue
        support = [old[i] for i in leaf["stations"]]
        if all(p in indices for p in support):
            leaves.append({"path": leaf["path"], "kind": "covered",
                           "stations": [indices[p] for p in support]})
            reused += 1
        else:
            box = tuple(float(v) for v in _box(leaf["path"]))
            pending.append((leaf["path"], box))
    while pending:
        if (deadline is not None and time.perf_counter() >= deadline) or examined >= max_cells:
            raise CoverBudgetExpired()
        path, box = pending.pop()
        examined += 1
        if _outside(box, 1800., RANGE_MARGIN):
            leaves.append({"path": path, "kind": "outside"})
            continue
        support = _support(points, box, 1000., RANGE_MARGIN, ORIENTATION_MARGIN)
        if support is not None:
            leaves.append({"path": path, "kind": "covered", "stations": [indices[p] for p in support]})
            continue
        center = ((box[0]+box[1])/2., (box[2]+box[3])/2.)
        nearest = (max(box[0], min(0., box[1])), max(box[2], min(0., box[3])))
        for probe in (center, nearest):
            if math.hypot(*probe) <= 1800.-RANGE_MARGIN:
                witness = _separating_witness(points, probe, 1000., RANGE_MARGIN)
                if witness is not None:
                    return {"passed": False, "status": "counterexample", "reason": "strict_separating_counterexample",
                            "visited_cells": examined, "reused_leaves": reused, "counterexample": witness}
        if len(path) >= 16:
            return {"passed": False, "status": "inconclusive", "reason": "depth_budget",
                    "visited_cells": examined, "reused_leaves": reused}
        pending.extend((path+str(i), child) for i, child in enumerate(_children(box)))
    return {"passed": True, "status": "certified", "reason": None, "visited_cells": examined,
            "reused_leaves": reused, "leaves": sorted(leaves, key=lambda leaf: leaf["path"])}


def _certify(points, max_cells, deadline=None, cache_only=False, template=None):
    started = time.perf_counter()
    key = points, max_cells
    if key in _CERT_CACHE:
        certificate, diagnostic = _CERT_CACHE[key]
        return certificate, dict(diagnostic, cache_hit=True, wall_time_s=time.perf_counter()-started)
    if cache_only or (deadline is not None and started >= deadline):
        return None, {"status": "budget_exhausted", "cache_hit": False, "wall_time_s": 0.}
    if len(points) < 3:
        return None, {"status": "insufficient_points", "wall_time_s": time.perf_counter()-started}
    try:
        floating = (_search_from_template(points, template, max_cells, deadline) if template is not None else
            certify_directional_cover(points, max_cells=max_cells,
                max_depth=16, range_margin_m=RANGE_MARGIN, orientation_margin_m=ORIENTATION_MARGIN))
    except CoverBudgetExpired:
        return None, {"status": "budget_exhausted", "cache_hit": False,
                      "visited_cells": max_cells, "wall_time_s": time.perf_counter()-started}
    diagnostic = {"status": floating["status"], "reason": floating["reason"],
                  "visited_cells": floating["visited_cells"],
                  "reused_leaves": floating.get("reused_leaves", 0),
                  "counterexample": floating.get("counterexample"), "cache_hit": False}
    if deadline is not None and time.perf_counter() >= deadline:
        diagnostic.update(status="budget_exhausted", wall_time_s=time.perf_counter()-started)
        return None, diagnostic
    if not floating["passed"]:
        diagnostic["wall_time_s"] = time.perf_counter()-started
        if len(_CERT_CACHE) >= 256:
            _CERT_CACHE.pop(next(iter(_CERT_CACHE)))
        _CERT_CACHE[key] = None, diagnostic
        return None, diagnostic
    certificate = {"version": VERSION, "points": [list(p) for p in points],
                   "leaves": floating["leaves"]}
    try:
        verify_exact_certificate(certificate, deadline=deadline)
    except CoverBudgetExpired:
        diagnostic.update(status="budget_exhausted", wall_time_s=time.perf_counter()-started)
        return None, diagnostic
    except (ValueError, ArithmeticError) as error:
        diagnostic.update(status="exact_check_rejected", reason=str(error),
                          wall_time_s=time.perf_counter()-started)
        return None, diagnostic
    diagnostic.update(status="exact_certified", wall_time_s=time.perf_counter()-started)
    if len(_CERT_CACHE) >= 256:
        _CERT_CACHE.pop(next(iter(_CERT_CACHE)))
    _CERT_CACHE[key] = certificate, diagnostic
    return certificate, diagnostic


def certify(points, max_cells=20000, deadline=None, cache_only=False, template=None):
    return _certify(tuple(sorted(set(position_key(p) for p in points))), max_cells, deadline, cache_only, template)


class AdaptiveCoverLedger:
    def __init__(self, fixed_points, *, max_cells=20000, episode_wall_s=10., decision_wall_s=.5,
                 max_episode_searches=64):
        if type(max_cells) is not int or not 1 <= max_cells <= 200000:
            raise ValueError("Invalid coverage work budget")
        self.fixed = frozenset(position_key(p) for p in fixed_points)
        self.max_cells = max_cells
        self.episode_wall_s, self.decision_wall_s = episode_wall_s, decision_wall_s
        if not all(math.isfinite(v) and v > 0 for v in (episode_wall_s, decision_wall_s)):
            raise ValueError("Positive certificate time budgets required")
        if type(max_episode_searches) is not int or not 1 <= max_episode_searches <= 1000:
            raise ValueError("Invalid expensive certificate search budget")
        self.max_episode_searches = max_episode_searches
        self.expensive_searches = 0
        self.search_wall_s = 0.
        self.decision_search_wall_s = 0.
        self.decision_queries = 0
        started = time.perf_counter()
        base, diagnostic = certify(self.fixed, max_cells)
        self.initial_certificate_wall_s = time.perf_counter()-started
        if base is None:
            raise ValueError("Initial fixed cover has no exact certificate: "+diagnostic["status"])
        self.proofs = {_digest(base): base}
        root = _digest(base)
        self.pending = {c: set(self.fixed) for c in range(1, 21)}
        self.negatives = {c: set() for c in range(1, 21)}
        self.known, self.cleared = set(), set()
        self.roots = {c: root for c in range(1, 21)}
        self.events = []
        self.attempts = []
        self.last_action_index = -1

    def begin_decision(self):
        self.decision_search_wall_s = 0.
        self.decision_queries = 0

    def observe(self, action_index, action, point, channel, result):
        self.last_action_index = action_index
        if action == "clear":
            if result == "success":
                self.cleared.add(channel)
            return
        if action != "measure":
            return
        point = position_key(point)
        if result in ("near", "direction"):
            self.known.add(channel)
        elif result == "no_signal" and channel not in self.known | self.cleared:
            self.negatives[channel].add(point)
            self.pending[channel].discard(point)

    def unknown(self):
        return set(range(1, 21))-self.known-self.cleared

    def proposal(self, channel, removed, added=()):
        """Prospective feasibility only: this never changes actual or pending."""
        removed = position_key(removed)
        if channel not in self.unknown() or removed not in self.pending[channel]:
            return None
        self.decision_queries += 1
        remaining = min(self.episode_wall_s-self.search_wall_s,
                        self.decision_wall_s-self.decision_search_wall_s)
        if remaining <= 0 or self.decision_queries > 128:
            self.attempts.append({"after_action_index": self.last_action_index,
                "channel": channel, "removed": list(removed), "status": "budget_exhausted", "wall_time_s": 0.})
            return None
        points = self.negatives[channel] | (self.pending[channel]-{removed}) | {position_key(p) for p in added}
        began = time.perf_counter()
        certificate, diagnostic = certify(points, self.max_cells, deadline=began+remaining,
            cache_only=self.expensive_searches >= self.max_episode_searches,
            template=self.proofs[self.roots[channel]])
        elapsed = time.perf_counter()-began
        self.search_wall_s += elapsed
        self.decision_search_wall_s += elapsed
        self.expensive_searches += not diagnostic.get("cache_hit", False) and "visited_cells" in diagnostic
        if elapsed > remaining:
            certificate = None
            diagnostic["status"] = "budget_exhausted"
        self.attempts.append({"after_action_index": self.last_action_index,
            "channel": channel, "removed": list(removed),
            "prospective_points": [list(position_key(p)) for p in added], **diagnostic})
        return certificate

    def remove(self, channel, point):
        """Commit only a certificate using already actual + retained planned points."""
        point = position_key(point)
        certificate = self.proposal(channel, point)
        if certificate is None:
            return False
        before = sorted(self.pending[channel])
        self.pending[channel].remove(point)
        root = _digest(certificate)
        self.proofs[root] = certificate
        self.roots[channel] = root
        self.events.append({"after_action_index": self.last_action_index,
            "channel": channel, "removed": list(point), "previous_pending": before,
            "pending": sorted(self.pending[channel]), "proof_root": root,
            "proof_kind": "residual_plan"})
        return True

    def absent(self):
        # Empty pending is safe because each committed residual plan contains
        # only actual negatives plus remaining pending; source provenance is
        # independently replayed from the action history before any use outside.
        return {c for c in self.unknown() if not self.pending[c]}

    def artifact(self):
        return {"version": VERSION, "fixed_points": sorted(self.fixed),
            "proofs": self.proofs, "events": self.events, "attempts": self.attempts,
            "pending": {str(c): sorted(v) for c, v in self.pending.items()},
            "roots": {str(c): root for c, root in self.roots.items()},
            "certified_absent": sorted(self.absent()),
            "budget": {"episode_search_wall_s": self.episode_wall_s,
                "decision_search_wall_s": self.decision_wall_s, "max_cells_per_search": self.max_cells,
                "max_expensive_searches": self.max_episode_searches, "queries_per_decision": 128},
            "cost": {"initial_certificate_wall_s": self.initial_certificate_wall_s,
                "search_wall_s": self.search_wall_s, "expensive_searches": self.expensive_searches}}


def replay_adaptive_artifact(action_history, artifact):
    """Verify deletion history and final actual certificates from real records.

    Input is SearchResult.action_history (accepted actions), not fabricated
    ledger credits. Physical wire/protocol acceptance is independently audited.
    """
    if artifact.get("version") != VERSION:
        raise ValueError("Unknown adaptive ledger version")
    proofs = artifact["proofs"]
    for root, certificate in proofs.items():
        if _digest(certificate) != root:
            raise ValueError("Certificate hash mismatch")
        verify_exact_certificate(certificate)
    fixed = {position_key(p) for p in artifact["fixed_points"]}
    pending = {c: set(fixed) for c in range(1, 21)}
    negatives = {c: set() for c in range(1, 21)}
    known, cleared = set(), set()
    by_prefix = {}
    for event in artifact["events"]:
        prefix = event["after_action_index"]
        if type(prefix) is not int or not 0 <= prefix < len(action_history):
            raise ValueError("Deletion has no accepted triggering action")
        by_prefix.setdefault(prefix, []).append(event)
    for index, item in enumerate(action_history):
        channel, result = item["channel"], item["result"]
        point = position_key(item["position"])
        if item["action"] == "measure":
            if result in ("near", "direction"):
                known.add(channel)
            elif result == "no_signal" and channel not in known | cleared:
                negatives[channel].add(point)
                pending[channel].discard(point)
        elif item["action"] == "clear" and result == "success":
            cleared.add(channel)
        for event in by_prefix.get(index, []):
            c = event["channel"]
            removed = position_key(event["removed"])
            previous = {position_key(p) for p in event["previous_pending"]}
            after = {position_key(p) for p in event["pending"]}
            if c in known | cleared or previous != pending[c] or removed not in previous:
                raise ValueError("Invalid deletion source state or previous obligations")
            if after != previous-{removed} or event["proof_kind"] != "residual_plan":
                raise ValueError("Nontransactional deletion")
            certificate = proofs[event["proof_root"]]
            points = {position_key(p) for p in certificate["points"]}
            if not points <= negatives[c] | after:
                raise ValueError("Planned proof cites unmeasured and unplanned points")
            pending[c] = after
    for c in range(1, 21):
        if pending[c] != {position_key(p) for p in artifact["pending"][str(c)]}:
            raise ValueError("Final obligations differ from actual history replay")
    absent = set(artifact["certified_absent"])
    expected = {c for c in range(1, 21) if c not in known | cleared and not pending[c]}
    if absent != expected:
        raise ValueError("False final absence claim")
    for c in absent:
        points = {position_key(p) for p in proofs[artifact["roots"][str(c)]]["points"]}
        if not points <= negatives[c]:
            raise ValueError("Final absence relies on future or wrong-channel measurements")
    return {"passed": True, "certified_absent": sorted(absent),
            "actual_cleared": sorted(cleared), "deleted_obligations": len(artifact["events"]),
            "complete": len(cleared) == 16 or (10 <= len(cleared) <= 16 and known <= cleared
                        and cleared | absent == set(range(1, 21)))}

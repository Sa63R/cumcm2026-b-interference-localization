"""Filter unknown-channel cover choices at the public Q3 source-count cap.

This optional v3 variant retains every original source action and its order.
It neither credits a physical scan nor declares an uncleared source removed.
The decision-limit safety fallback is intentionally the unchanged base routine;
it can still execute scans omitted from the actor's candidate list.

Known-channel covers remain available even when a safe clear is offered.
Safe clearance proves feasibility, not zero value of another measurement:
tighter localization may enlarge the guaranteed-clear robot-position set
and reduce subsequent travel. That tradeoff remains the policy's choice.

Geometric and relative-negative silence certificates are NOT implemented here.
They need a separately reviewed geometry/provenance contract. A small region,
no_signal count, or inferred absence alone is never a filtering certificate.
"""
from simulator_client.rules import CHANNELS, MAX_SOURCES
from strategies.efficient import EfficientSearch

from .action_sets import action_schema
from .joint_scan import JointScanRLSearch


def filter_certified_cover(candidates, known_channels):
    """Filter an ORIGINAL JointScanRLSearch candidate list, never arbitrary actions.

    Count-cap evidence is the set of distinct accepted known channels, not an
    estimate of how many unknown sources might remain. Clear availability,
    region radius, and blocked status never filter known-channel covers.
    """
    known = set(known_channels)
    removed = {"known_count_cap": 0}
    if not known <= set(CHANNELS) or len(known) > MAX_SOURCES:
        return list(candidates), removed  # Inconsistent public state: fail closed.
    count_certified = len(known) == MAX_SOURCES
    retained = []
    for candidate in candidates:
        reason = None
        if candidate.kind == "cover":
            if count_certified and candidate.channel not in known:
                reason = "known_count_cap"
        if reason is None:
            retained.append(candidate)
        else:
            removed[reason] += 1
    return retained, removed


class CertifiedCoverRLSearch(JointScanRLSearch):
    def __init__(self, client, policy, *, feature_version="v3", **kwargs):
        if feature_version != "v3":
            raise ValueError("certified cover requires v3 feature semantics")
        schema = action_schema("certified_cover")
        if hasattr(policy, "action_schema") and policy.action_schema != schema:
            raise ValueError("certified cover controller requires a matching policy action schema")
        super().__init__(client, policy, feature_version=feature_version, **kwargs)
        self.report.learning.update(action_schema=schema,
            certified_cover_filter_calls=0,
            certified_cover_removed_sum={"known_count_cap": 0},
            certified_cover_count_semantics="candidate exclusions across decision builds; not skipped physical actions",
            certified_cover_scope="actor cover candidates only; physical ledger and safety fallback unchanged")

    def _candidates(self, remaining):
        original = super()._candidates(remaining)
        retained, removed = filter_certified_cover(original, self.detected | self.cleared)
        metrics = self.report.learning
        metrics["certified_cover_filter_calls"] += 1
        for reason, count in removed.items():
            metrics["certified_cover_removed_sum"][reason] += count
        return retained

    def _teacher(self, candidates, remaining):
        """Preserve the public heuristic where available, never select a removed row.

        The inherited scan heuristic consults the physical pending ledger,
        which deliberately still contains certified-but-unmeasured pairs.
        Construct its cover choices from the offered rows without mutating that
        ledger, focus, client state, or any measured/visited count.
        """
        if not candidates:
            raise ValueError("No legal candidate for the teacher")
        sources = {c.channel for c in candidates if c.kind != "cover"}
        if self.focus in sources:
            return next(i for i, c in enumerate(candidates)
                        if c.kind != "cover" and c.channel == self.focus)
        covers = {}
        for index, candidate in enumerate(candidates):
            if candidate.kind == "cover":
                covers.setdefault(candidate.point, []).append(index)

        def scan_index(point):
            indices = covers[point]
            return next((i for i in indices
                         if candidates[i].channel == self.client.state.current_channel), indices[0])

        if self.scan_focus in covers:
            return scan_index(self.scan_focus)
        if not any(self.scan_ledger.values()) and self.points[0] in covers:
            return scan_index(self.points[0])
        offered_sites = [p for p in remaining if p in covers]
        task = EfficientSearch._next_task(self, offered_sites)
        if task is not None:
            kind, channel, point = task
            if kind == "cover" and point in covers:
                return scan_index(point)
            if kind == "source":
                return next((i for i, c in enumerate(candidates)
                             if c.kind != "cover" and c.channel == channel), 0)
        return 0

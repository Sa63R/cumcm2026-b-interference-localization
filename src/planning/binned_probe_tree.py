"""Finite observation bins with exact continuous bounded-noise probabilities."""

import math

from planning.probe_tree import ProbeTree, WeightedSource


class BinnedProbeTree(ProbeTree):
    def __init__(self, *, bin_width_deg=0.5, **kwargs):
        super().__init__(**kwargs)
        if bin_width_deg not in (0.5, 1.0):
            raise ValueError("bin_width_deg must be 0.5 or 1.0")
        self.bin_width_deg = bin_width_deg

    def _branches(self, region, current, support):
        width = self.bin_width_deg
        bins = round(360 / width)
        branches = {}
        for source in support:
            if current.distance_to(source.position) <= 5:
                outcomes = [(None, 1.0)]
            else:
                angle = math.degrees(math.atan2(source.position.y - current.y,
                                                source.position.x - current.x)) % 360
                # The local model is uniform continuous angular error [-1,1].
                # Integrate its interval exactly over each circular bin.
                left, right = angle - 1, angle + 1
                outcomes = []
                for index in range(math.floor(left / width), math.floor(right / width) + 1):
                    overlap = min(right, (index + 1) * width) - max(left, index * width)
                    if overlap > 1e-14:
                        outcomes.append((index % bins, overlap / 2))
            for outcome, chance in outcomes:
                weights = branches.setdefault(outcome, {})
                weights[source.position] = weights.get(source.position, 0) + source.weight * chance
        result = []
        for outcome, weights in branches.items():
            self.branch_count += 1
            self.singleton_branches += len(weights) == 1
            probability = sum(weights.values())
            posterior = tuple(WeightedSource(point, weight / probability)
                              for point, weight in weights.items())
            if outcome is None:
                updated = None
            else:
                updated = region.copy()
                # Preserve old constraints; the *new* interval observation has
                # an extra half-bin uncertainty. This never touches live state.
                updated.error_deg = 1.005 + width / 2
                updated.observe(current, (outcome + .5) * width)
                if not updated.vertices:
                    raise ValueError("Binned finite observation has an empty geometric region")
            result.append((probability, updated, posterior))
        return result

    def choose(self, region, current, observed):
        point, log = super().choose(region, current, observed)
        log.update(model="finite_source_continuous_uniform_error_binned_observation",
                   observation_bin_deg=self.bin_width_deg,
                   noise_nodes=None, noise_law="uniform_continuous_minus1_plus1")
        return point, log

"""G3 memory features over unchanged G1 candidates, execution and certificates."""
from . import micro_controller as g1
from .negative_memory import (NegativeObservationMemory, FEATURE_NAMES,
                              VERSION as MEMORY_VERSION, BANK_SIZE)


FEATURE_SCHEMA_VERSION = "q4-micro-g3-negative-v1"
GLOBAL_FEATURE_NAMES = g1.GLOBAL_FEATURE_NAMES
CANDIDATE_FEATURE_NAMES = g1.CANDIDATE_FEATURE_NAMES + FEATURE_NAMES
GLOBAL_DIM, CANDIDATE_DIM = len(GLOBAL_FEATURE_NAMES), len(CANDIDATE_FEATURE_NAMES)


def feature_schema():
    return dict(version=FEATURE_SCHEMA_VERSION, global_features=list(GLOBAL_FEATURE_NAMES),
        candidate_features=list(CANDIDATE_FEATURE_NAMES), negative_memory_version=MEMORY_VERSION,
        negative_memory_bank_size=BANK_SIZE)


class Q4MemorySearch(g1.Q4MicroSearch):
    def __init__(self, client, policy=None, **kwargs):
        self.negative_memory = NegativeObservationMemory()
        self._memory_history_index = 0
        kwargs.setdefault("max_decisions", 512)
        super().__init__(client, policy=policy, **kwargs)
        self.report.learning.update(algorithm=FEATURE_SCHEMA_VERSION, feature_schema=feature_schema())
        self.report.strategy_parameters.update(q4_rl_feature_schema=FEATURE_SCHEMA_VERSION,
            negative_memory_scope="Eight public negative-only compatibility features; scores never mask actions or certify safety/completion",
            negative_memory_version=MEMORY_VERSION, negative_memory_bank_size=BANK_SIZE)

    def _consume_actual_history(self):
        super()._consume_actual_history()
        for item in self.report.action_history[self._memory_history_index:]:
            # _Search appends these records only after accepted is exactly True.
            # Forward only public action/position/channel/result, never identity,
            # source truth, case metadata or post-termination evaluations.
            self.negative_memory.observe(action=item["action"], position=item["position"],
                channel=item["channel"], result=item["result"], accepted=True)
        self._memory_history_index = len(self.report.action_history)

    def _features(self, candidates):
        self._consume_actual_history()
        global_features, rows = super()._features(candidates)
        extras = self.negative_memory.score_candidates(
            [(candidate.channel, (candidate.point.x, candidate.point.y)) for candidate in candidates])
        for row, extra in zip(rows, extras.tolist()):
            row.extend(extra)
        if len(global_features) != GLOBAL_DIM or any(len(row) != CANDIDATE_DIM for row in rows):
            raise ValueError("G3 memory feature schema mismatch")
        return global_features, rows

    def run(self):
        report = super().run()
        self._consume_actual_history()
        report.learning["negative_memory"] = self.negative_memory.state_summary()
        return report


def run_q4_memory(client, policy=None, *, problem=4, **kwargs):
    if type(problem) is not int or problem != 4:
        raise ValueError("Q4 memory controller supports problem=4 only")
    return Q4MemorySearch(client, policy=policy, **kwargs).run()

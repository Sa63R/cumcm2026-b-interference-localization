"""Client-side observations, with simulator truth kept separate from inference."""

from dataclasses import asdict, dataclass, field
import math

from .rules import MAX_COORDINATE_M


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def __post_init__(self):
        for name in ("x", "y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if abs(value) > MAX_COORDINATE_M or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and within +/-{MAX_COORDINATE_M:g}")

    def distance_to(self, other: "Position") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    @classmethod
    def coerce(cls, value: "Position | tuple[float, float]") -> "Position":
        if isinstance(value, cls):
            return value
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("position must be Position(x, y) or a pair of coordinates")
        return cls(*value)


@dataclass
class SourceState:
    # no_signal alone never changes a source to absent or cleared.
    status: str = "unknown"
    measurement_count: int = 0
    last_result: str | None = None
    last_bearing_deg: float | None = None
    last_measurement_position: Position | None = None
    last_measurement_virtual_time_s: float | None = None
    failed_clear_count: int = 0


@dataclass
class TimeBreakdown:
    movement_s: float = 0.0
    switching_s: float = 0.0
    detection_s: float = 0.0
    optical_s: float = 0.0
    removal_s: float = 0.0

    @property
    def total_s(self) -> float:
        return sum(asdict(self).values())


@dataclass
class ClientState:
    session: str = "new"
    position: Position = field(default_factory=lambda: Position(0.0, 0.0))
    current_channel: int = 1
    virtual_time_s: float = 0.0
    max_virtual_duration_s: float | None = None
    real_deadline: float | None = None
    accepted_actions: int = 0
    sources: dict[int, SourceState] = field(default_factory=dict)
    time_breakdown: TimeBreakdown = field(default_factory=TimeBreakdown)

    @property
    def cleared_count(self) -> int:
        return sum(source.status == "cleared" for source in self.sources.values())

    def snapshot(self) -> dict:
        result = asdict(self)
        result["cleared_count"] = self.cleared_count
        result["estimated_total_virtual_time_s"] = self.time_breakdown.total_s
        return result

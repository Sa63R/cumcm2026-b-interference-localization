"""Independent local physical engine with an observation-only client adapter.

This is a research implementation of the supplied PDF/appendix rules. It does
not implement official login, encryption, upload, or formal-case generation.
"""

from dataclasses import asdict
import hashlib
import math
import threading
import time
from types import SimpleNamespace

from simulator_client.client import SimulatorClient
from simulator_client.errors import DeadlineExceeded, SessionError
from simulator_client.state import ClientState, Position
from .cases import Scenario


class LocalResearchSimulator:
    """Owns ground truth. Policies receive only client(), never this object."""

    def __init__(self, scenario: Scenario, *, max_virtual_duration_s=360_000,
                 max_real_duration_s=1200, clock=time.monotonic):
        if max_virtual_duration_s <= 0 or max_real_duration_s <= 0:
            raise ValueError("Time budgets must be positive")
        self.scenario = scenario
        self._sources = {s.channel: s for s in scenario.sources}
        self._cleared: set[int] = set()
        self._position = Position(0, 0)
        self._channel = 1
        self._microseconds = 0
        self._components_us = dict(movement_s=0, switching_s=0, detection_s=0,
                                   optical_s=0, removal_s=0)
        self._session = "new"
        self._clock = clock
        self._started = None
        self._ended = None
        self._max_virtual = float(max_virtual_duration_s)
        self._max_real = float(max_real_duration_s)
        self._client_created = False
        self._history = []

    def client(self) -> "MemoryClient":
        if self._client_created:
            raise SessionError("Only one research client is permitted per scenario run")
        self._client_created = True
        return MemoryClient(self._execute, clock=self._clock)

    def _check_deadline(self):
        if self._started is not None and self._clock() - self._started >= self._max_real:
            self._session = "real_timeout"
            self._ended = self._clock()
            raise DeadlineExceeded("Research real-time budget exhausted")
        if self._microseconds >= self._max_virtual * 1_000_000:
            self._session = "virtual_timeout"
            self._ended = self._clock()
            raise DeadlineExceeded("Research virtual-time budget exhausted")

    def _error(self, position, channel):
        # Normalize signed zero. IEEE float coordinates are otherwise exact keys;
        # revisiting the same coordinate and channel gives exactly the same error.
        values = [0.0 if v == 0 else float(v) for v in (position.x, position.y)]
        key = f"{self.scenario.seed}|{channel}|{values[0].hex()}|{values[1].hex()}"
        bits = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big")
        mode = self.scenario.error_mode
        if mode == "zero":
            return 0.0
        if mode == "positive_extreme":
            return 1.0
        if mode == "negative_extreme":
            return -1.0
        if mode == "alternating_extreme":
            return 1.0 if bits % 2 else -1.0
        return 2 * (bits / (2**64 - 1)) - 1

    @staticmethod
    def _quantize_bearing(bearing, error):
        # The official implementation's rounding order is unspecified. This
        # explicit research convention preserves BOTH centidegrees and <=1 deg
        # final circular error. It may move one centidegree toward the truth.
        value = round((bearing + error) % 360, 2) % 360
        delta = (value - bearing + 180) % 360 - 180
        if delta > 1 + 1e-12:
            value = round(value - 0.01, 2) % 360
        elif delta < -1 - 1e-12:
            value = round(value + 0.01, 2) % 360
        return value

    def _visible(self, source, position):
        dx, dy = position.x - source.x, position.y - source.y
        distance = math.hypot(dx, dy)
        if distance > source.reception_radius_m:
            return False
        if source.orientation_deg is not None:
            theta = math.radians(source.orientation_deg)
            dot = math.cos(theta) * dx + math.sin(theta) * dy
            # Tolerance only covers trig roundoff at the explicitly included edge.
            if dot < -1e-12 * max(1.0, distance):
                return False
        return True

    def _execute(self, path, payload):
        if path == "/enter":
            if self._session != "new":
                raise SessionError("Research session already entered")
            self._session = "active"
            self._started = self._clock()
            details = {"max_virtual_duration_s": self._max_virtual,
                       "max_real_duration_s": self._max_real,
                       "remaining_real_duration_s": self._max_real}
        else:
            if self._session != "active":
                raise SessionError("Research session is not active")
            self._check_deadline()
            if path == "/exit":
                self._session = "exited"
                self._ended = self._clock()
                details = {"exit_reason": "user_exit"}
            elif path in {"/measure", "/clear"}:
                position = Position(**payload["position"])
                channel = payload["channel"]
                if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
                    raise ValueError("Channel must be an integer in 1..20")
                costs = {"movement_s": round(self._position.distance_to(position) / 5 * 1_000_000)}
                source = self._sources.get(channel) if channel not in self._cleared else None
                if path == "/measure":
                    costs.update(switching_s=int(channel != self._channel) * 1_000_000,
                                 detection_s=5_000_000)
                    self._channel = channel
                    if source is None or not self._visible(source, position):
                        details = {"measure_result": "no_signal"}
                    elif math.hypot(source.x - position.x, source.y - position.y) <= 5:
                        details = {"measure_result": "near"}
                    else:
                        bearing = math.degrees(math.atan2(source.y - position.y, source.x - position.x)) % 360
                        details = {"measure_result": "direction", "svd_deg": self._quantize_bearing(
                            bearing, self._error(position, channel))}
                else:
                    success = source is not None and math.hypot(source.x - position.x, source.y - position.y) <= 20
                    costs.update(optical_s=3_000_000, removal_s=int(success) * 2_000_000)
                    if success:
                        self._cleared.add(channel)
                    details = {"clear_result": "success" if success else "no_target_in_range"}
                for name, value in costs.items():
                    self._components_us[name] += value
                self._microseconds += sum(costs.values())
                self._position = position
            else:
                raise ValueError("Unknown research action")
        response = {"accepted": True, "real_timestamp_ms": round(time.time() * 1000),
                    "virtual_time_s": self._microseconds / 1_000_000, **details}
        self._history.append({"index": len(self._history), "action": path,
                              "position": asdict(self._position),
                              "channel": payload.get("channel"),
                              "response": response.copy()})
        return response

    def finish_for_evaluation(self, reason="harness_finished"):
        """Harness-only cleanup after an interrupted policy; not a robot action."""
        if self._session in {"new", "active"}:
            self._session = reason
            self._ended = self._clock()

    def evaluation(self) -> dict:
        """Expose truth to the evaluator ONLY after the run has terminated."""
        if self._session in {"new", "active"}:
            raise SessionError("Evaluation truth is unavailable until the run ends")
        count = len(self._sources)
        clear_count = len(self._cleared)
        virtual_time = self._microseconds / 1_000_000
        return {
            "kind": "local_research_only", "case_id": self.scenario.case_id,
            "source_total": count, "cleared_total": clear_count,
            "cleared_fraction": clear_count / count,
            "all_cleared": clear_count == count,
            "virtual_time_s": virtual_time,
            "mean_time_per_cleared_s": virtual_time / clear_count if clear_count else None,
            "wall_time_s": (self._ended - self._started) if self._started is not None else 0.0,
            "time_breakdown_s": {k: v / 1_000_000 for k, v in self._components_us.items()},
            "action_count": len(self._history),
            "measurement_count": sum(a["action"] == "/measure" for a in self._history),
            "failed_clear_count": sum(a["response"].get("clear_result") == "no_target_in_range" for a in self._history),
            "cleared_channels": sorted(self._cleared),
            "remaining_channels": sorted(set(self._sources) - self._cleared),
            "simulator_stop_reason": self._session,
            "ground_truth": self.scenario.evaluation_config(),
        }

    def observation_history(self) -> list[dict]:
        """Observer trace contains only submitted actions and legal responses."""
        return [dict(item, position=item["position"].copy(), response=item["response"].copy())
                for item in self._history]


class MemoryClient(SimulatorClient):
    """Same state/action API as SimulatorClient, without HTTP or truth access.

    The callback is private implementation detail. This API separation prevents
    accidental truth leakage; Python object introspection is not sandboxed.
    """

    def __init__(self, exchange_fn, *, clock=time.monotonic):
        self.robot_id = "LOCAL-RESEARCH-ONLY"
        self.base_url = "memory://local-research"
        self.state = ClientState()
        self.timeout_s = 3.0
        self._clock = clock
        self._lock = threading.Lock()
        self._pending = None
        self._closed = False
        self._sequence = 0
        self._exchange_fn = exchange_fn

    def _new_action(self, path, fields):
        self._sequence += 1
        action = SimpleNamespace(path=path, payload=fields, first_sent_at=self._clock())
        response = self._exchange_fn(path, fields)
        self._validate_common(response)
        self._validate_accepted(action, response)
        self._apply_accepted(action, response)
        return response

    def _log(self, event, **fields):
        pass

    def close(self):
        self._closed = True

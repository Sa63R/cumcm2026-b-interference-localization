"""Reliable, serial HTTP client for the official CUMCM 2026 B simulator."""

from .client import SimulatorClient
from .errors import (
    DeadlineExceeded, OutcomeUnknown, PendingActionError,
    RequestRejected, SessionError, SimulatorError,
)
from .state import ClientState, Position

__all__ = [
    "SimulatorClient", "Position", "ClientState", "SimulatorError",
    "SessionError", "DeadlineExceeded", "OutcomeUnknown",
    "PendingActionError", "RequestRejected",
]

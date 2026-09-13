"""Errors distinguish definite rejection from an unknown execution outcome."""


class SimulatorError(Exception):
    """Base client error."""


class SessionError(SimulatorError):
    """An action is invalid in the client's current session state."""


class DeadlineExceeded(SessionError):
    """The conservative local runtime budget is exhausted."""


class PendingActionError(SessionError):
    """Resolve the previous uncertain action before sending a new action."""


class RequestRejected(SimulatorError):
    def __init__(self, status_code: int, response: dict, request_id: str):
        self.status_code = status_code
        self.response = response
        self.request_id = request_id
        super().__init__(f"Simulator rejected {request_id}: HTTP {status_code}, {response}")


class OutcomeUnknown(SimulatorError):
    def __init__(self, request_id: str, detail: str):
        self.request_id = request_id
        super().__init__(
            f"Outcome of {request_id} is unknown: {detail}. "
            "Use retry_pending() with this client; do not send the action with a new ID."
        )

"""One-shot GUI replies with request identity and cooperative cancellation."""

from copy import deepcopy
from threading import Condition
from uuid import uuid4


class GuiToolMailbox:
    """Synchronize one worker's GUI requests without accepting stale replies."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._active: tuple[str, str] | None = None
        self._result: dict | None = None
        self._cancelled = False

    def begin(self, name: str) -> str | None:
        """Return a fresh transport token, or None after worker cancellation."""
        with self._condition:
            if self._cancelled:
                return None
            if self._active is not None:
                raise RuntimeError("A GUI request is already pending")
            token = uuid4().hex
            self._active = (token, name)
            self._result = None
            return token

    def submit(self, result: dict) -> bool:
        """Copy and accept the first reply matching the live token and tool."""
        with self._condition:
            if (
                self._cancelled
                or self._active is None
                or self._result is not None
                or self._active != (result.get("tool_call_id"), result.get("name"))
            ):
                return False
            self._result = deepcopy(result)
            self._condition.notify_all()
            return True

    def wait(self, timeout: float) -> dict | None:
        """Consume the reply, closing admission on timeout or cancellation."""
        with self._condition:
            if self._active is None:
                return None
            self._condition.wait_for(
                lambda: self._cancelled or self._result is not None, timeout
            )
            result = None if self._cancelled else self._result
            self._active = None
            self._result = None
            return result

    def cancel(self) -> None:
        """Permanently close this worker's mailbox and wake its waiter."""
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

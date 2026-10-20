from __future__ import annotations

from collections import deque
from typing import Protocol

from ..samples import ImuSample


class ImuSource(Protocol):
    def now(self) -> float:
        """Current time on the same clock as the sample timestamps."""

    def poll(self) -> list[ImuSample]:
        """Everything that arrived since the last call, oldest first. Never blocks."""

    def connected(self) -> bool:
        """True while samples are arriving."""

    def last_receive(self) -> float | None:
        """`now()` when the last sample arrived, or None if none yet."""

    def rate_hz(self) -> float:
        """Recent gyro sample rate."""

    def reset(self) -> None:
        """Drop the current connection and look for the stream again."""

    def close(self) -> None: ...


class RateMeter:
    """Samples per second over a sliding window."""

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self._times: deque[float] = deque()

    def add(self, t: float) -> None:
        self._times.append(t)

    def rate(self, now: float) -> float:
        while self._times and self._times[0] < now - self.window_s:
            self._times.popleft()
        return len(self._times) / self.window_s

    def clear(self) -> None:
        self._times.clear()

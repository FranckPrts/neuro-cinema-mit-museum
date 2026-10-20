"""A real-time source backed by the virtual head (no hardware needed)."""

from __future__ import annotations

import time

from ..samples import ImuSample
from ..sim.head import RATE_HZ, VirtualHead
from .base import RateMeter


class SimSource:
    def __init__(self, head: VirtualHead | None = None):
        self.head = head or VirtualHead(seed=int(time.time()))
        self.streaming = True
        self._t_next: float | None = None
        self._last_receive: float | None = None
        self._rate = RateMeter()

    def now(self) -> float:
        return time.monotonic()

    def poll(self) -> list[ImuSample]:
        now = self.now()
        if not self.streaming:
            self._t_next = None
            return []
        if self._t_next is None or now - self._t_next > 1.0:
            self._t_next = now  # first call, or we fell far behind: do not catch up
        out: list[ImuSample] = []
        while self._t_next <= now:
            t = self._t_next
            acc, gyro = self.head.step(t)
            out.append(ImuSample(t, "acc", *acc))
            out.append(ImuSample(t, "gyro", *gyro))
            self._rate.add(now)
            self._t_next += 1.0 / RATE_HZ
        if out:
            self._last_receive = now
        return out

    def connected(self) -> bool:
        return self.streaming and self._last_receive is not None and self.now() - self._last_receive < 1.0

    def last_receive(self) -> float | None:
        return self._last_receive

    def rate_hz(self) -> float:
        return self._rate.rate(self.now())

    def reset(self) -> None:
        self._t_next = None

    def close(self) -> None:
        pass

"""No headset (`source.kind = "none"`): the engine runs only as the camera's hub.

The headset side stays `disconnected`; heartbeat, status and commands work as usual.
"""

from __future__ import annotations

import time

from ..samples import ImuSample


class NoSource:
    def now(self) -> float:
        return time.monotonic()

    def poll(self) -> list[ImuSample]:
        return []

    def connected(self) -> bool:
        return False

    def last_receive(self) -> float | None:
        return None

    def rate_hz(self) -> float:
        return 0.0

    def headset_id(self) -> str | None:
        return None

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

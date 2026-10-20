"""Which of the six screen halves the visitor is facing.

Three screens (left / middle / right) chosen by yaw, each split horizontally
(top / bottom) chosen by pitch. Both use hysteresis so the highlight does not
flicker when the head rests near a boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import TargetsConfig

COLUMNS = ("left", "middle", "right")
ROWS = ("top", "bottom")


@dataclass(frozen=True)
class Target:
    column: str
    row: str

    def __str__(self) -> str:
        return f"{self.column}/{self.row}"


class TargetTracker:
    def __init__(self, cfg: TargetsConfig):
        self.cfg = cfg
        self._centres = {
            "left": cfg.left_yaw_deg,
            "middle": cfg.middle_yaw_deg,
            "right": cfg.right_yaw_deg,
        }
        self.reset()

    HISTORY_S = 10.0

    def reset(self) -> None:
        self._column: str | None = None
        self._row: str | None = None
        self.current: Target | None = None
        self.since: float | None = None
        # (since, target) for recent changes, oldest first.
        self._history: deque[tuple[float, Target | None]] = deque()

    def column_centre(self, column: str) -> float:
        return self._centres[column]

    def dwell(self, t: float) -> float:
        if self.current is None or self.since is None:
            return 0.0
        return max(0.0, t - self.since)

    def at(self, t: float) -> tuple[Target | None, float | None]:
        """The target that was current at time t, and since when."""
        for since, target in reversed(self._history):
            if since <= t:
                return target, since
        return None, None

    def update(self, t: float, yaw: float, pitch: float, hold_row: bool = False) -> bool:
        """Returns True when the current target changed.

        `hold_row` keeps top/bottom as it is (used while a nod may be under
        way, since nodding swings the pitch across the divider).
        """
        self._column = self._classify_column(yaw)
        if not hold_row or self._row is None:
            self._row = self._classify_row(pitch)
        new = Target(self._column, self._row) if self._column and self._row else None
        if new == self.current:
            return False
        self.current = new
        self.since = t
        self._history.append((t, new))
        while self._history and self._history[0][0] < t - self.HISTORY_S and len(self._history) > 1:
            self._history.popleft()
        return True

    def _classify_column(self, yaw: float) -> str | None:
        cfg = self.cfg
        h = cfg.yaw_hysteresis_deg
        margin = cfg.outer_margin_deg + (h / 2 if self._column else -h / 2)
        if yaw < cfg.left_yaw_deg - margin or yaw > cfg.right_yaw_deg + margin:
            return None
        nearest = min(COLUMNS, key=lambda c: abs(yaw - self._centres[c]))
        current = self._column
        if current is None or nearest == current:
            return nearest
        # Switch only once clearly past the boundary (by h/2 on each side).
        if abs(yaw - self._centres[nearest]) + h < abs(yaw - self._centres[current]):
            return nearest
        return current

    def _classify_row(self, pitch: float) -> str | None:
        cfg = self.cfg
        h = cfg.pitch_hysteresis_deg
        d = pitch - cfg.split_pitch_deg
        limit = cfg.max_pitch_deg + (h / 2 if self._row else -h / 2)
        if abs(d) > limit:
            return None
        current = self._row
        if current is None:
            return "top" if d >= 0 else "bottom"
        if current == "top" and d < -h / 2:
            return "bottom"
        if current == "bottom" and d > h / 2:
            return "top"
        return current

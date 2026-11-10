"""Which of the six screen halves the visitor is facing.

Three columns (left / middle / right) chosen by yaw, each split horizontally
(top / bottom) chosen by pitch. The yaw range of each column comes from the
active selection setup (`setups.py`); neighbouring ranges may overlap, and the
head stays in its column across the shared zone, so the highlight does not
flicker when the head rests near a boundary. Rows use a dead band likewise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import TargetsConfig
from .setups import COLUMNS, SelectionSetup, builtin_setup

ROWS = ("top", "bottom")


@dataclass(frozen=True)
class Target:
    column: str
    row: str

    def __str__(self) -> str:
        return f"{self.column}/{self.row}"


class TargetTracker:
    """`cfg` gives the hysteresis; `setup` the columns (default: the built-in one)."""

    def __init__(self, cfg: TargetsConfig, setup: SelectionSetup | None = None):
        self.cfg = cfg
        self.setup = setup or builtin_setup(cfg)
        self.reset()

    def set_setup(self, setup: SelectionSetup) -> None:
        """Switch to another selection setup; the target starts again from none."""
        self.setup = setup
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
        return self.setup.centres[column]

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
        """Stay in the current column while inside its range (overlaps are the
        hysteresis); otherwise the column whose range holds the yaw, and in an
        overlap entered from elsewhere, the zone's midpoint decides. Outside
        every range: none. The outer ends are yaw_hysteresis_deg / 2 harder to
        enter than to stay in."""
        ranges = self.setup.ranges
        current = self._column
        h2 = self.cfg.yaw_hysteresis_deg / 2
        out = h2 if current else -h2

        def inside(col: str) -> bool:
            lo, hi = ranges[col]
            if col == COLUMNS[0]:
                lo -= out
            if col == COLUMNS[-1]:
                return lo <= yaw <= hi + out
            return lo <= yaw < hi

        if current is not None and inside(current):
            return current
        hits = [c for c in COLUMNS if inside(c)]
        if len(hits) < 2:
            return hits[0] if hits else None
        a, b = hits[0], hits[-1]  # neighbours (only they may overlap)
        return a if yaw < (ranges[b][0] + ranges[a][1]) / 2 else b

    def _classify_row(self, pitch: float) -> str | None:
        h = self.cfg.pitch_hysteresis_deg
        d = pitch - self.setup.split_pitch_deg
        limit = self.setup.max_pitch_deg + (h / 2 if self._row else -h / 2)
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

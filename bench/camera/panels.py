"""Aim direction -> screen and half ("middle/top", "left/bottom", ...).

Uses viewer/room_geometry.py to find where the aim ray from the eye meets the
room. The top/bottom divider is a horizontal line across each screen
(--split, as a fraction of the screen height from its bottom edge). Near it
there is a dead band (in degrees of pitch), so the half doesn't flicker when
the visitor points right at the line.
"""

from __future__ import annotations

import math

import room_geometry as rg

SCREEN_OF_WALL = {"front": "middle", "left": "left", "right": "right"}


class PanelMapper:
    def __init__(self, cfg: dict, *, split: float = 0.5, band_deg: float = 3.0, halves: bool = True):
        if not 0.0 < split < 1.0:
            raise ValueError("split must be between 0 and 1")
        self.cfg, self.split, self.band_deg, self.halves = cfg, split, band_deg, halves
        self._screen: str | None = None
        self._row: str | None = None
        self.last: dict = {}

    def midline_height(self, wall: str) -> float:
        """Height (m) of the top/bottom divider on a wall screen."""
        w = rg.walls(self.cfg)[wall]
        inset = self.cfg["screens"]["inset_m"]
        return inset + self.split * (w["height"] - 2 * inset)

    def midline_pitch(self, wall: str, point) -> float:
        """Pitch from the eye to the divider, straight above or below a point on that wall."""
        eye_h = self.cfg["room"]["eye_height_m"]
        d = math.hypot(point[0], point[2])
        return math.degrees(math.atan2(self.midline_height(wall) - eye_h, d))

    def __call__(self, yaw: float, pitch: float) -> str | None:
        h = rg.hit(self.cfg, yaw, pitch)
        self.last = {"hit": h, "wall": h["surface"], "row": None, "delta": None, "mid_pitch": None}
        screen = SCREEN_OF_WALL.get(h["surface"]) if h["on_screen"] else None
        if screen is None:
            self._screen = self._row = None
            return None
        if not self.halves:
            self._screen = screen
            return screen
        mid = self.midline_pitch(h["surface"], h["point"])
        delta = pitch - mid
        half = self.band_deg / 2
        if screen == self._screen and self._row == "top" and delta > -half:
            row = "top"
        elif screen == self._screen and self._row == "bottom" and delta < half:
            row = "bottom"
        else:
            row = "top" if delta >= 0 else "bottom"
        self._screen, self._row = screen, row
        self.last.update(row=row, delta=delta, mid_pitch=mid)
        return f"{screen}/{row}"


def screen_part(target: str | None) -> str | None:
    """'middle/top' -> 'middle'."""
    return target.split("/")[0] if target else None

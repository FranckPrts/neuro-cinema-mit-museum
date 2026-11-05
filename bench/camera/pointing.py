"""Pure maths for the camera bench: pose landmarks -> aim direction in the room frame.

No MediaPipe or OpenCV here, so it can be tested on its own.

MediaPipe world landmarks: metres, origin between the hips, camera-aligned axes
(x = image right, y = down, z = away from the camera).

Room frame (same as viewer/room_geometry.py): x = visitor's right, y = up,
-z = straight ahead (yaw 0, toward the front wall). +yaw = right, +pitch = up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# MediaPipe Pose landmark indices.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 11, 12, 13, 14
L_WRIST, R_WRIST, L_INDEX, R_INDEX = 15, 16, 19, 20
L_HIP, R_HIP = 23, 24

ARMS = {
    # name: (shoulder, elbow, wrist, index). "left"/"right" = the visitor's own arm.
    "left": (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
    "right": (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
}


def cam_to_room(v, bearing_deg: float = 0.0, tilt_deg: float = 0.0) -> np.ndarray:
    """Camera-frame vector -> room-frame vector.

    bearing_deg: where the camera is, seen from the visitor's standing spot
      (0 = straight ahead on the front wall, + = to the visitor's right;
      the front-right corner is about +29.6 deg in this room).
    tilt_deg: how far the camera looks down (0 = level).
    The camera is assumed to point at the visitor.
    """
    x, y, z = (float(c) for c in v)
    t = math.radians(tilt_deg)
    # Undo the downward tilt: express in a level camera frame (x right, y down, z forward).
    yl = y * math.cos(t) + z * math.sin(t)
    zl = -y * math.sin(t) + z * math.cos(t)
    # Level camera at bearing 0 looks toward the visitor, i.e. along room +z.
    rx, ry, rz = -x, -yl, zl
    b = math.radians(bearing_deg)
    return np.array([rx * math.cos(b) - rz * math.sin(b), ry, rx * math.sin(b) + rz * math.cos(b)])


def yaw_pitch(v) -> tuple[float, float]:
    x, y, z = (float(c) for c in v)
    return math.degrees(math.atan2(x, -z)), math.degrees(math.atan2(y, math.hypot(x, z)))


@dataclass
class Aim:
    yaw: float
    pitch: float
    source: str          # "left arm", "right arm" or "head"
    quality: float       # 0..1, how much to trust it


def _p(lms, i) -> np.ndarray:
    return np.array([lms[i].x, lms[i].y, lms[i].z], dtype=float)


def _vis(lms, *idx) -> float:
    return min(getattr(lms[i], "visibility", 1.0) or 0.0 for i in idx)


def arm_aim(world, bearing_deg=0.0, tilt_deg=0.0, *, min_straight=0.8, min_raise_deg=-40.0,
            min_vis=0.5) -> Aim | None:
    """Eye-to-fingertip ray of the most raised, straight arm, or None if no arm is pointing.

    A pointing arm must be fairly straight (shoulder-wrist distance over
    upper+lower arm length >= min_straight) and raised (shoulder->wrist pitch
    above min_raise_deg; a hanging arm is about -90).
    """
    eye = (_p(world, L_EYE) + _p(world, R_EYE)) / 2
    best: Aim | None = None
    best_raise = -math.inf
    for name, (s, e, w, i) in ARMS.items():
        vis = _vis(world, s, e, w)
        if vis < min_vis:
            continue
        S, E, W = _p(world, s), _p(world, e), _p(world, w)
        seg = np.linalg.norm(E - S) + np.linalg.norm(W - E)
        if seg < 1e-6:
            continue
        straight = float(np.linalg.norm(W - S) / seg)
        _, arm_pitch = yaw_pitch(cam_to_room(W - S, bearing_deg, tilt_deg))
        if straight < min_straight or arm_pitch < min_raise_deg:
            continue
        tip = _p(world, i) if _vis(world, i) >= min_vis else W
        d = cam_to_room(tip - eye, bearing_deg, tilt_deg)
        if np.linalg.norm(d) < 1e-6:
            continue
        yaw, pitch = yaw_pitch(d)
        q = min(1.0, vis) * min(1.0, (straight - min_straight) / (1 - min_straight) + 0.3)
        # Prefer the more raised arm (the other one is usually hanging).
        if arm_pitch > best_raise:
            best, best_raise = Aim(yaw, pitch, f"{name} arm", q), arm_pitch
    return best


def head_aim(world, bearing_deg=0.0, tilt_deg=0.0, *, min_vis=0.5) -> Aim | None:
    """Rough head yaw from the ear midpoint to the nose (pose model, not a face model).

    Pitch is left at 0: the nose sits below the ears, so this vector says little
    about up/down. Good enough to compare with the arm; a face model would do better.
    """
    if _vis(world, NOSE) < min_vis:
        return None
    ears = [i for i in (L_EAR, R_EAR) if _vis(world, i) >= 0.3]
    if not ears:
        return None
    mid = sum(_p(world, i) for i in ears) / len(ears)
    d = cam_to_room(_p(world, NOSE) - mid, bearing_deg, tilt_deg)
    d[1] = 0.0
    if np.linalg.norm(d) < 1e-6:
        return None
    yaw, _ = yaw_pitch(d)
    return Aim(yaw, 0.0, "head", _vis(world, NOSE) * (1.0 if len(ears) == 2 else 0.6))


class Selector:
    """Smooths the aim, turns it into a screen highlight, and a dwell into a selection.

    screen_of(yaw, pitch) -> screen name or None (e.g. room_geometry.hit).
    """

    def __init__(self, screen_of, *, smooth_s=0.15, hold_s=0.25, dwell_s=1.5, lost_s=0.5):
        self.screen_of = screen_of
        self.smooth_s, self.hold_s, self.dwell_s, self.lost_s = smooth_s, hold_s, dwell_s, lost_s
        self.vec: np.ndarray | None = None
        self.last_t: float | None = None
        self.last_seen: float | None = None
        self.highlight: str | None = None
        self.since: float | None = None
        self._cand: str | None = None
        self._cand_since: float | None = None
        self._selected = False

    @staticmethod
    def _unit(yaw, pitch) -> np.ndarray:
        y, p = math.radians(yaw), math.radians(pitch)
        return np.array([math.cos(p) * math.sin(y), math.sin(p), -math.cos(p) * math.cos(y)])

    def update(self, t: float, aim: Aim | None) -> list[tuple[str, str | None]]:
        """Returns events: ("highlight", screen|None) and ("select", screen)."""
        events: list[tuple[str, str | None]] = []
        if aim is not None:
            u = self._unit(aim.yaw, aim.pitch)
            if self.vec is None or self.last_seen is None or t - self.last_seen > self.lost_s:
                self.vec = u
            else:
                dt = t - self.last_t if self.last_t is not None else 0.0
                a = 1 - math.exp(-dt / self.smooth_s) if self.smooth_s > 0 else 1.0
                v = self.vec + a * (u - self.vec)
                self.vec = v / np.linalg.norm(v)
            self.last_seen = t
        elif self.last_seen is not None and t - self.last_seen > self.lost_s:
            self.vec = None
        self.last_t = t

        target = self.screen_of(*yaw_pitch(self.vec)) if self.vec is not None else None
        if target != self._cand:
            self._cand, self._cand_since = target, t
        if self._cand != self.highlight and self._cand_since is not None and t - self._cand_since >= self.hold_s:
            self.highlight, self.since, self._selected = self._cand, t, False
            events.append(("highlight", self.highlight))
        if self.highlight is not None and not self._selected and self.since is not None and t - self.since >= self.dwell_s:
            self._selected = True
            events.append(("select", self.highlight))
        return events

    def aim(self) -> tuple[float, float] | None:
        return yaw_pitch(self.vec) if self.vec is not None else None

    def dwell(self, t: float) -> float:
        if self.highlight is None or self._selected or self.since is None:
            return 0.0
        return min(1.0, (t - self.since) / self.dwell_s)

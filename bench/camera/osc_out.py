"""The camera bench's output in the engine's OSC format, under /nc/cam/.

Same addresses and arguments as nc-engine's (see the README's OSC reference),
with the /nc/cam prefix so the two never mix:

  /nc/cam/heartbeat  uptime_s                     1 Hz
  /nc/cam/state      idle | active | selected     on change and every second
  /nc/cam/target     column row                   on change and every second (none none = no target)
  /nc/cam/selected   column row                   when a dwell selects
  /nc/cam/dwell      0.0 .. 1.0                   every frame while a target is highlighted
  /nc/cam/pose       yaw pitch roll (degrees)     every frame while someone is aiming (roll = 0)
  /nc/cam/event      name [args]                  camera_started, camera_stopped, selected <col> <row>
  /nc/cam/calib      x y z  aim_x aim_y aim_z  bearing tilt  eye_height split pitch_offset
                     on change and every 5 s: camera position and aim point (room
                     frame, metres), bearing and tilt (deg), eye height (m), the
                     top/bottom divider (fraction of screen height), arm pitch offset (deg)

Columns are left / middle / right, rows top / bottom (none when halves are
off, e.g. in head mode). The older /nc/cam/aim and /nc/cam/highlight|select
are still sent until nothing reads them.
"""

from __future__ import annotations

from collections.abc import Callable

PREFIX = "/nc/cam"
REPEAT_S = 1.0
CALIB_EVERY_S = 5.0


def split_target(target: str | None) -> tuple[str, str]:
    """'middle/top' -> ('middle', 'top'); 'left' -> ('left', 'none'); None -> ('none', 'none')."""
    if not target:
        return "none", "none"
    col, _, row = target.partition("/")
    return col, row or "none"


class CamOsc:
    def __init__(self, send: Callable[..., None], t0: float, legacy: bool = True):
        self.send = send  # send(address, *args)
        self.t0 = t0
        self.legacy = legacy
        self._state: str | None = None
        self._target: tuple[str, str] | None = None
        self._next_repeat = t0
        self._next_beat = t0
        self.calib: tuple[float, ...] | None = None  # set by the bench, see /nc/cam/calib
        self._calib_sent: tuple[float, ...] | None = None
        self._next_calib = t0
        send(f"{PREFIX}/event", "camera_started")

    def frame(self, t: float, aim: tuple[float, float] | None, events: list[tuple[str, str | None]],
              highlight: str | None, dwell: float, selected: bool) -> None:
        p = PREFIX
        if aim is not None:
            self.send(f"{p}/pose", float(aim[0]), float(aim[1]), 0.0)
            if self.legacy:
                self.send(f"{p}/aim", float(aim[0]), float(aim[1]))
        for kind, screen in events:
            if self.legacy:
                self.send(f"{p}/{kind}", screen or "none")
            if kind == "select" and screen:
                col, row = split_target(screen)
                self.send(f"{p}/selected", col, row)
                self.send(f"{p}/event", "selected", col, row)
        state = "selected" if (highlight and selected) else ("active" if aim is not None or highlight else "idle")
        target = split_target(highlight)
        repeat = t >= self._next_repeat
        if state != self._state or repeat:
            self.send(f"{p}/state", state)
            self._state = state
        if target != self._target or repeat:
            self.send(f"{p}/target", *target)
            self._target = target
        if repeat:
            self._next_repeat = t + REPEAT_S
        if highlight and not selected:
            self.send(f"{p}/dwell", float(dwell))
        if self.calib is not None and (self.calib != self._calib_sent or t >= self._next_calib):
            self.send(f"{p}/calib", *(float(v) for v in self.calib))
            self._calib_sent, self._next_calib = self.calib, t + CALIB_EVERY_S
        if t >= self._next_beat:
            self.send(f"{p}/heartbeat", float(t - self.t0))
            self._next_beat = t + 1.0

    def close(self) -> None:
        self.send(f"{PREFIX}/event", "camera_stopped")

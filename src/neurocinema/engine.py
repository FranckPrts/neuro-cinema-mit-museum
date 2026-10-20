"""The interaction state machine.

    DISCONNECTED ──data──► IDLE ──motion──► CALIBRATING ──2 s still──► ACTIVE
         ▲                  ▲                                         │   ▲
         │                  └──── not worn (still / tilted) ──────────┤   │
     (any state, no data)                                  nod ───────┤   │ hold elapsed
                                                         ▼            │   │
                                                      SELECTED ───────┴───┘
                                       shake (ACTIVE / SELECTED) ──► RESETTING ──still──► ACTIVE

The engine is driven by samples (`process`) and by the clock (`tick`); it does
no I/O itself beyond calling `out.send(address, *args)`, which keeps it
deterministic and testable.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Protocol

import numpy as np

from .config import Config
from .fusion import CalibrationError, Calibrator, Fusion
from .gestures import Gesture, OscillationDetector
from .samples import ImuSample
from .targets import Target, TargetTracker

log = logging.getLogger(__name__)


class State(StrEnum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    CALIBRATING = "calibrating"
    ACTIVE = "active"
    SELECTED = "selected"
    RESETTING = "resetting"


class Phase(StrEnum):
    # Set by TouchDesigner from the film timeline. Nods only select while
    # choosing; watching is where Layer C will read EEG.
    CHOOSING = "choosing"
    WATCHING = "watching"


# States in which a visitor is wearing a calibrated headset.
_TRACKING = (State.ACTIVE, State.SELECTED, State.RESETTING)
# States that belong to a visitor's session (resumed after a short dropout).
_SESSION = (State.CALIBRATING, *_TRACKING)


class Output(Protocol):
    def send(self, address: str, *args: object) -> None: ...


class Engine:
    def __init__(self, cfg: Config, out: Output):
        self.cfg = cfg
        self.out = out
        self.p = cfg.osc.prefix
        self.fusion = Fusion(cfg.imu)
        self.targets = TargetTracker(cfg.targets)
        self.nod = OscillationDetector("nod", cfg.gestures.nod)
        self.shake = OscillationDetector("shake", cfg.gestures.shake)
        self.calibrator = Calibrator(cfg.calibration.still_s)
        self.phase = Phase(cfg.interaction.start_phase)
        self.state = State.DISCONNECTED
        self.state_since = 0.0
        self.connected = False
        self._resume_session = False
        self._calibration_slow_sent = False
        self._idle_still_since: float | None = None
        self._idle_tilt_since: float | None = None
        self._reset_still_since: float | None = None
        self._next_pose = 0.0
        self._next_snapshot = 0.0

    # --- inputs -------------------------------------------------------------

    def process(self, s: ImuSample) -> None:
        self.fusion.update(s)
        t = s.t

        if self.state == State.CALIBRATING:
            self.calibrator.add(s, self.fusion.stillness.is_still())
            if self.calibrator.ready(t):
                self._finish_calibration(t)
            return

        if s.kind != "gyro":
            return

        if self.state == State.IDLE:
            w = self.fusion.last_gyro
            if w is not None and float(np.linalg.norm(w - self.fusion.bias)) > self.cfg.interaction.wake_dps:
                self._set_state(State.CALIBRATING, t, "motion while idle")
            return

        if self.state not in _TRACKING:
            return

        f = self.fusion
        nod = self.nod.update(t, f.pitch_rate, f.yaw_rate)
        shake = self.shake.update(t, f.yaw_rate, f.pitch_rate)
        if shake is not None:
            self._on_shake(shake)
        elif nod is not None:
            self._on_nod(nod)

        # Hold top/bottom while a nod may be under way, so its pitch swing does
        # not flip the highlight. The column keeps tracking: nods barely yaw.
        if self.state == State.ACTIVE:
            if self.targets.update(t, f.yaw_deg, f.pitch_deg, hold_row=self.nod.busy):
                self._emit_target()

    def tick(self, now: float) -> None:
        ic = self.cfg.interaction

        if self.state == State.SELECTED and now - self.state_since >= ic.selected_hold_s:
            self._set_state(State.ACTIVE, now, "selection hold elapsed")

        elif self.state == State.RESETTING:
            if self.fusion.stillness.is_still():
                if self._reset_still_since is None:
                    self._reset_still_since = now
            else:
                self._reset_still_since = None
            settled = (
                self._reset_still_since is not None
                and now - self._reset_still_since >= ic.reset_settle_s
            )
            if settled or now - self.state_since >= ic.reset_timeout_s:
                self._apply_reset(now, "shake")

        elif self.state == State.CALIBRATING:
            if (not self._calibration_slow_sent
                    and now - self.state_since >= self.cfg.calibration.slow_after_s):
                self._calibration_slow_sent = True
                self._event("calibration_slow")

        if self.state in _TRACKING:
            self._check_not_worn(now)

        if now >= self._next_pose:
            self._next_pose = now + 1.0 / self.cfg.osc.pose_rate_hz
            self._emit_continuous(now)

        if now >= self._next_snapshot:
            self._next_snapshot = now + 1.0 / self.cfg.osc.snapshot_rate_hz
            self.emit_snapshot()

    def on_connection(self, connected: bool, now: float) -> None:
        if connected == self.connected:
            return
        self.connected = connected
        if not connected:
            self._resume_session = self.state in _SESSION
            self._set_state(State.DISCONNECTED, now, "no data from headset")
            return
        # The gap invalidates orientation and bias: start again from scratch.
        self.fusion.reset()
        self.nod.reset()
        self.shake.reset()
        if self._resume_session:
            self._set_state(State.CALIBRATING, now, "headset back during a session")
        else:
            self._set_state(State.IDLE, now, "headset connected")

    def command(self, name: str, args: tuple, now: float) -> None:
        """Commands from TouchDesigner (/<prefix>/cmd/<name>)."""
        if name in ("start", "recalibrate"):
            if self.connected:
                self._set_state(State.CALIBRATING, now, f"command {name}")
        elif name == "reset":
            if self.fusion.calibrated and self.state in (*_TRACKING, State.IDLE):
                self._apply_reset(now, "command")
        elif name == "idle":
            if self.connected:
                self._set_state(State.IDLE, now, "command idle")
        elif name == "phase":
            value = str(args[0]) if args else ""
            try:
                phase = Phase(value)
            except ValueError:
                log.warning("ignoring /cmd/phase %r (expected choosing|watching)", value)
                return
            if phase != self.phase:
                self.phase = phase
                log.info("phase -> %s", phase.value)
                self.out.send(f"{self.p}/phase", phase.value)
        elif name == "ping":
            self.out.send(f"{self.p}/pong", *args)
        elif name == "snapshot":
            self.emit_snapshot()
        else:
            log.warning("unknown command %r", name)

    # --- transitions ----------------------------------------------------------

    def _set_state(self, new: State, now: float, reason: str) -> None:
        if new == self.state:
            return
        log.info("state %s -> %s (%s)", self.state.value, new.value, reason)
        self.state = new
        self.state_since = now
        if new == State.CALIBRATING:
            self.calibrator.reset()
            self._calibration_slow_sent = False
        if new in (State.CALIBRATING, State.IDLE, State.DISCONNECTED):
            self._clear_target()
        if new == State.RESETTING:
            self._reset_still_since = None
            self._clear_target()
        if new == State.ACTIVE:
            self._idle_still_since = None
            self._idle_tilt_since = None
            self.nod.reset()
            self.shake.reset()
        self.out.send(f"{self.p}/state", new.value)

    def _finish_calibration(self, t: float) -> None:
        gyro_mean, acc_mean = self.calibrator.result()
        try:
            self.fusion.calibrate(gyro_mean, acc_mean)
        except CalibrationError as e:
            log.warning("calibration rejected: %s", e)
            self.calibrator.reset()
            return
        log.info("calibrated: bias=%s deg/s", np.round(self.fusion.bias, 2).tolist())
        self.targets.reset()
        self._event("calibrated")
        self._set_state(State.ACTIVE, t, "calibrated")

    def _apply_reset(self, now: float, why: str) -> None:
        self.fusion.zero_yaw()
        self._clear_target()
        log.info("yaw zeroed (%s)", why)
        self._event("reset")
        self._set_state(State.ACTIVE, now, f"reset by {why}")

    def _on_shake(self, g: Gesture) -> None:
        self._event("shake")
        if self.state in (State.ACTIVE, State.SELECTED):
            self._set_state(State.RESETTING, g.t, "shake")

    def _on_nod(self, g: Gesture) -> None:
        self._event("nod")
        if self.state != State.ACTIVE:
            return
        if self.phase != Phase.CHOOSING:
            self._event("nod_ignored", "watching")
            return
        # What the visitor was looking at when the nod began, not now.
        target, since = self.targets.at(g.onset)
        if target is None or since is None:
            self._event("nod_ignored", "no_target")
            return
        if g.onset - since < self.cfg.interaction.min_dwell_s:
            self._event("nod_ignored", "dwell")
            return
        self._select(target, g.t)

    def _select(self, target: Target, t: float) -> None:
        centre = self.targets.column_centre(target.column)
        self.fusion.reanchor_yaw(centre, self.cfg.interaction.reanchor_gain)
        log.info("selected %s", target)
        self.out.send(f"{self.p}/selected", target.column, target.row)
        self._event("selected", target.column, target.row)
        self._set_state(State.SELECTED, t, f"nod on {target}")

    def _check_not_worn(self, now: float) -> None:
        ic = self.cfg.interaction
        motion = self.fusion.stillness.gyro_range()
        if motion is not None and motion < ic.idle_gyro_range_dps:
            if self._idle_still_since is None:
                self._idle_still_since = now
            elif now - self._idle_still_since >= ic.idle_still_s:
                self._set_state(State.IDLE, now, "no motion at all (headset off?)")
                return
        else:
            self._idle_still_since = None

        tilt = self.fusion.tilt_from_calibration_deg()
        if tilt is not None and tilt > ic.idle_tilt_deg:
            if self._idle_tilt_since is None:
                self._idle_tilt_since = now
            elif now - self._idle_tilt_since >= ic.idle_tilt_s:
                self._set_state(State.IDLE, now, f"tilted {tilt:.0f} deg (headset off?)")
        else:
            self._idle_tilt_since = None

    # --- outputs --------------------------------------------------------------

    def _event(self, name: str, *args: object) -> None:
        log.info("event %s %s", name, " ".join(map(str, args)))
        self.out.send(f"{self.p}/event", name, *args)

    def _emit_target(self) -> None:
        t = self.targets.current
        if t is None:
            self.out.send(f"{self.p}/target", "none", "none")
        else:
            self.out.send(f"{self.p}/target", t.column, t.row)

    def _clear_target(self) -> None:
        had = self.targets.current is not None
        self.targets.reset()
        if had:
            self._emit_target()

    def _emit_continuous(self, now: float) -> None:
        if self.state == State.CALIBRATING:
            self.out.send(f"{self.p}/calibration", round(self.calibrator.progress(now), 3))
        if self.state in _TRACKING:
            f = self.fusion
            self.out.send(f"{self.p}/pose", round(f.yaw_deg, 2), round(f.pitch_deg, 2), round(f.roll_deg, 2))
        if self.state == State.ACTIVE:
            dwell = self.targets.dwell(now) / max(self.cfg.interaction.min_dwell_s, 1e-6)
            self.out.send(f"{self.p}/dwell", round(min(1.0, dwell), 3))

    def emit_snapshot(self) -> None:
        """Re-send the latest discrete values, so a late-starting TD catches up."""
        self.out.send(f"{self.p}/state", self.state.value)
        self.out.send(f"{self.p}/phase", self.phase.value)
        self._emit_target()

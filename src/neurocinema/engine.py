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
from .fusion import CalibrationError, Calibrator, make_fusion
from .gestures import Gesture, OscillationDetector
from .samples import ImuSample
from .setups import SelectionSetup, SetupError, load_setups, startup_name
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
# Entering these drops the target (/target none none).
_CLEARS_TARGET = (State.CALIBRATING, State.IDLE, State.DISCONNECTED, State.RESETTING)


class Output(Protocol):
    def send(self, address: str, *args: object) -> None: ...


class Engine:
    def __init__(self, cfg: Config, out: Output, setups: dict[str, SelectionSetup] | None = None):
        """`setups`: the selection setups (default: loaded from targets.setups_file)."""
        self.cfg = cfg
        self.out = out
        self.p = cfg.osc.prefix
        self.fusion = make_fusion(cfg.imu)
        self.setups = setups if setups is not None else load_setups(cfg.targets)
        self.setup_name = startup_name(cfg.targets, self.setups)
        log.info("selection setup %s: %s", self.setup_name, self.setups[self.setup_name].describe())
        self.targets = TargetTracker(cfg.targets, self.setups[self.setup_name])
        self.nod = OscillationDetector("nod", cfg.gestures.nod)
        self.shake = OscillationDetector("shake", cfg.gestures.shake)
        self.calibrator = Calibrator(cfg.calibration.still_s)
        self.phase = Phase(cfg.interaction.start_phase)
        self.state = State.DISCONNECTED
        self.state_since = 0.0
        self.connected = False
        self.headset: str | None = None  # identity of the connected headset, if the source knows it
        self._resume_session = False
        self._calibration_slow_sent = False
        self._idle_still_since: float | None = None
        self._idle_tilt_since: float | None = None
        self._reset_still_since: float | None = None
        # /target goes out at most once per tick: a burst of samples (the Muse
        # sends 3 at a time) can cross a boundary and back, or change column and
        # then row, and TouchDesigner must not see the in-between targets.
        self._sent_target: Target | None = None
        self._target_pending = False
        self._next_pose = 0.0
        self._next_snapshot = 0.0
        # For /calib/imu: when calibration happened, yaw corrected since, clipped samples then.
        self._calib_t: float | None = None
        self._yaw_corrected = 0.0
        self._clipped_at_calib = 0
        self._next_calib = 0.0

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
                self._target_pending = True

    def tick(self, now: float) -> None:
        ic = self.cfg.interaction
        self._flush_target()

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

        if now >= self._next_calib:
            self.emit_calibration(now)

    def on_connection(self, connected: bool, now: float, headset: str | None = None) -> None:
        """Data started or stopped. `headset` identifies the headset (None = unknown)."""
        if connected == self.connected:
            return
        self.connected = connected
        if not connected:
            self._resume_session = self.state in _SESSION
            self._set_state(State.DISCONNECTED, now, "no data from headset")
            return
        # The gap invalidates the orientation: start again. The bias learned
        # off the head belongs to the headset, so it survives if it is the same one.
        same = headset is not None and headset == self.headset
        if headset != self.headset:
            log.info("headset %s -> %s", self.headset or "?", headset or "?")
        self.headset = headset
        self.fusion.reset(keep_idle_bias=same)
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
            self.emit_calibration(now)
        elif name == "setup":
            self._command_setup(str(args[0]) if args else "")
        else:
            log.warning("unknown command %r", name)

    def _command_setup(self, name: str) -> None:
        """/cmd/setup <name>: switch selection setup. No name: re-read the
        setups file and re-apply the active one. Yaw, calibration and state
        are untouched; the target starts again from none."""
        if not name:
            try:
                setups = load_setups(self.cfg.targets)
            except SetupError as e:
                log.warning("ignoring /cmd/setup: %s", e)
                return
            if self.setup_name not in setups:
                log.warning("setup %s is gone from the file; keeping it", self.setup_name)
                setups[self.setup_name] = self.setups[self.setup_name]
            self.setups = setups
            log.info("selection setups re-read: %s", ", ".join(setups))
            self.switch_setup(self.setup_name, force=True)
            return
        if name not in self.setups:
            log.warning("ignoring /cmd/setup %r (have %s)", name, ", ".join(self.setups))
            return
        self.switch_setup(name)

    def switch_setup(self, name: str, force: bool = False) -> None:
        setup = self.setups[name]
        if force or name != self.setup_name or setup != self.targets.setup:
            self.setup_name = name
            self.targets.set_setup(setup)
            self._sent_target, self._target_pending = None, False
            log.info("selection setup %s: %s", name, setup.describe())
            self.out.send(f"{self.p}/target", "none", "none")
            self._event("setup", name)
        self._emit_setup()

    # --- transitions ----------------------------------------------------------

    def _set_state(self, new: State, now: float, reason: str) -> None:
        if new == self.state:
            return
        log.info("state %s -> %s (%s)", self.state.value, new.value, reason)
        if new not in _CLEARS_TARGET:
            self._flush_target()  # the target as it stands goes out before the new state
        self.state = new
        self.state_since = now
        self.fusion.learn_idle_bias = new == State.IDLE
        if new == State.CALIBRATING:
            self.calibrator.reset()
            self._calibration_slow_sent = False
        if new in _CLEARS_TARGET:
            self._clear_target()
        if new == State.RESETTING:
            self._reset_still_since = None
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
        log.info("calibrated: bias=%s deg/s (from %s)",
                 np.round(self.fusion.bias, 2).tolist(), self.fusion.bias_source)
        self.targets.reset()
        self._calib_t, self._yaw_corrected = t, 0.0
        self._clipped_at_calib = self.fusion.clipped_samples
        self._event("calibrated")
        self.emit_calibration(t)
        self._set_state(State.ACTIVE, t, "calibrated")

    def _apply_reset(self, now: float, why: str) -> None:
        self._yaw_corrected += abs(self.fusion.yaw_deg)
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
        before = self.fusion.yaw_deg
        self.fusion.reanchor_yaw(centre, self.cfg.interaction.reanchor_gain)
        self._yaw_corrected += abs(self.fusion.yaw_deg - before)
        log.info("selected %s", target)
        self._flush_target()
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

    def _emit_setup(self) -> None:
        """/setup name l_lo l_hi m_lo m_hi r_lo r_hi c_l c_m c_r split max:
        the active selection setup, yaw ranges, centres and pitch, degrees."""
        self.out.send(f"{self.p}/setup", *self.targets.setup.osc_args())

    def _flush_target(self) -> None:
        """Send /target if it changed since the last one sent."""
        if self._target_pending:
            self._target_pending = False
            if self.targets.current != self._sent_target:
                self._emit_target()

    def _emit_target(self) -> None:
        t = self.targets.current
        self._sent_target, self._target_pending = t, False
        if t is None:
            self.out.send(f"{self.p}/target", "none", "none")
        else:
            self.out.send(f"{self.p}/target", t.column, t.row)

    def _clear_target(self) -> None:
        self.targets.reset()
        self._target_pending = False
        if self._sent_target is not None:
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

    CALIB_EVERY_S = 5.0

    def emit_calibration(self, now: float) -> None:
        """What calibration set, for the viewer (on calibration and every 5 s).

        /calib/imu  backend bias_source bias_dps since_s yaw_corrected_deg clipped
          since_s = -1 before calibration; yaw_corrected_deg = |re-anchoring| +
          |shake resets| since calibration (large = the yaw drifted);
          clipped = gyro samples at full scale since calibration.
        /calib/room  from_room room_hash left middle right lo left|middle middle|right hi
          screen centres and column bounds in degrees; room_hash = the room
          file's first 12 hex digits of SHA-1 ("" when from_room = 0).
        /setup  the active selection setup (see _emit_setup).
        """
        self._next_calib = now + self.CALIB_EVERY_S
        f, t = self.fusion, self.cfg.targets
        since = -1.0 if self._calib_t is None or not f.calibrated else round(now - self._calib_t, 1)
        self.out.send(f"{self.p}/calib/imu", self.cfg.imu.backend, f.bias_source,
                      round(float(np.linalg.norm(f.bias)), 3), since, round(self._yaw_corrected, 1),
                      int(f.clipped_samples - self._clipped_at_calib) if since >= 0 else 0)
        self.out.send(f"{self.p}/calib/room", int(t.from_room), t.room_hash,
                      *(round(v, 2) for v in (t.left_yaw_deg, t.middle_yaw_deg, t.right_yaw_deg)),
                      *(round(v, 2) for v in t.bounds()))
        self._emit_setup()

    def status(self, now: float) -> dict:
        """What the status page shows: state, calibration, and the numbers the
        "headset put down" rules look at (`App` copies it for the page's thread)."""
        f, ic = self.fusion, self.cfg.interaction
        motion = f.stillness.gyro_range()
        tilt = f.tilt_from_calibration_deg() if f.calibrated else None
        tracking = self.state in _TRACKING
        return {
            "state": self.state.value,
            "state_s": round(now - self.state_since, 1),
            "connected": self.connected,
            "phase": self.phase.value,
            "setup": self.setup_name,
            "backend": self.cfg.imu.backend,
            "calibrated": f.calibrated,
            "calibrated_s_ago": (round(now - self._calib_t) if f.calibrated and self._calib_t is not None
                                 else None),
            "calibration_progress": (round(self.calibrator.progress(now), 2)
                                     if self.state == State.CALIBRATING else None),
            "bias_source": f.bias_source,
            "bias_dps": [round(float(v), 2) for v in f.bias],
            "bias_limit_dps": self.cfg.imu.vqf_bias_clip_dps if self.cfg.imu.backend == "vqf" else None,
            "stand_still_s": round(f.stand_still_s(), 1),
            "stand_needed_s": self.cfg.imu.idle_bias_min_s,
            "motion_dps": None if motion is None else round(motion, 2),
            "put_down_motion_dps": ic.idle_gyro_range_dps,
            "dead_still_s": (round(now - self._idle_still_since, 1)
                             if tracking and self._idle_still_since is not None else None),
            "put_down_after_s": ic.idle_still_s,
            "tilt_deg": None if tilt is None else round(tilt),
            "put_down_tilt_deg": ic.idle_tilt_deg,
            "yaw_deg": round(f.yaw_deg, 1) if tracking else None,
            "pitch_deg": round(f.pitch_deg, 1) if tracking else None,
        }

    def emit_snapshot(self) -> None:
        """Re-send the latest discrete values, so a late-starting TD catches up."""
        self.out.send(f"{self.p}/state", self.state.value)
        self.out.send(f"{self.p}/phase", self.phase.value)
        self._emit_setup()
        self._emit_target()

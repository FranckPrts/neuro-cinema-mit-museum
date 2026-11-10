"""Head orientation from the Muse accelerometer + gyroscope (no magnetometer).

Conventions, after calibration:
  yaw_deg    + = visitor turned to their right, relative to the last zero.
                 Integrated from the gyro, so it drifts; it is re-zeroed by a
                 shake and pulled back on every selection.
  pitch_deg  + = looking up, relative to the calibration gaze. Absolute
                 (referenced to gravity), so it does not drift.
  roll_deg   + = right ear down. Absolute.

"Up" is tracked in the sensor frame with a complementary filter: the gyro
rotates it, the accelerometer slowly pulls it back. Yaw is integrated about that
vector, so it is correct however the band sits on the head, and the head's
forward/left axes are fixed at calibration from the configured forward axis.

Gyro bias: while the engine is idle (headset off the head) and the headset is
still, a long running mean of the raw gyro is kept ("idle bias"). Calibration
starts from it instead of the 2 s worn mean, which postural sway or a slow turn
can spoil.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from .config import ConfigError, ImuConfig
from .samples import ImuSample

log = logging.getLogger(__name__)

_AXES = {
    "x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}

# A gap longer than this between two samples is a dropout, not a time step.
MAX_DT_S = 0.25

# The Muse gyro reads at most 245 deg/s per axis (16 bits at 0.0074768 deg/s).
# A reading this close is clipped: the head turned faster and yaw lost the rest.
GYRO_CLIP_DPS = 244.5
# Clipped samples closer together than this are one burst (one log line).
_CLIP_BURST_GAP_S = 0.5


class CalibrationError(RuntimeError):
    pass


# Small 3-vector helpers: np.cross and np.linalg.norm carry a lot of per-call
# overhead, and these run on every sample.
def _norm(v: np.ndarray) -> float:
    return math.sqrt(float(v @ v))


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, a1, a2 = a.tolist()
    b0, b1, b2 = b.tolist()
    return np.array((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    return v / n if n > 1e-9 else v


class Stillness:
    """Is the head still right now? Uses ranges, so it does not depend on bias."""

    def __init__(self, window_s: float, gyro_range_dps: float, acc_range_g: float):
        self.window_s = window_s
        self.gyro_range_dps = gyro_range_dps
        self.acc_range_g = acc_range_g
        self._gyro: deque[tuple[float, np.ndarray]] = deque()
        self._acc: deque[tuple[float, float]] = deque()
        self._gyro_arr: np.ndarray | None = None  # the window as an array, until the next sample

    def clear(self) -> None:
        self._gyro.clear()
        self._acc.clear()
        self._gyro_arr = None

    def _trim(self, dq: deque, t: float) -> None:
        while dq and dq[0][0] < t - self.window_s:
            dq.popleft()

    def add_gyro(self, t: float, w: np.ndarray) -> None:
        self._gyro.append((t, w))
        self._trim(self._gyro, t)
        self._gyro_arr = None

    def _gyro_window(self) -> np.ndarray:
        if self._gyro_arr is None:
            self._gyro_arr = np.array([w for _, w in self._gyro])
        return self._gyro_arr

    def add_acc(self, t: float, norm: float) -> None:
        self._acc.append((t, norm))
        self._trim(self._acc, t)

    def _covered(self, dq: deque) -> bool:
        return len(dq) >= 3 and dq[-1][0] - dq[0][0] >= 0.8 * self.window_s

    def gyro_range(self) -> float | None:
        """Largest per-axis peak-to-peak gyro over the window, deg/s."""
        if not self._covered(self._gyro):
            return None
        g = self._gyro_window()
        return float((g.max(axis=0) - g.min(axis=0)).max())

    def gyro_mean(self) -> np.ndarray | None:
        if not self._gyro:
            return None
        return self._gyro_window().mean(axis=0)

    def is_still(self) -> bool:
        g = self.gyro_range()
        if g is None or g >= self.gyro_range_dps:
            return False
        if not self._covered(self._acc):
            return False
        a = [n for _, n in self._acc]
        return max(a) - min(a) < self.acc_range_g


class Calibrator:
    """Collects a continuous still period and averages gyro (bias) and acc (gravity)."""

    def __init__(self, still_s: float):
        self.still_s = still_s
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None
        self._gyro_sum = np.zeros(3)
        self._acc_sum = np.zeros(3)
        self._n_gyro = 0
        self._n_acc = 0

    def add(self, sample: ImuSample, still: bool) -> None:
        if not still:
            self.reset()
            return
        if self._since is None:
            self._since = sample.t
        v = np.array((sample.x, sample.y, sample.z))
        if sample.kind == "gyro":
            self._gyro_sum += v
            self._n_gyro += 1
        else:
            self._acc_sum += v
            self._n_acc += 1

    def progress(self, t: float) -> float:
        if self._since is None:
            return 0.0
        return max(0.0, min(1.0, (t - self._since) / self.still_s))

    def ready(self, t: float) -> bool:
        return self.progress(t) >= 1.0 and self._n_gyro > 0 and self._n_acc > 0

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        return self._gyro_sum / self._n_gyro, self._acc_sum / self._n_acc


class Fusion:
    """Complementary-filter backend; also the shared logic for other backends."""

    # False when the backend estimates the gyro bias itself (VQF): then the
    # idle-bias learning and the still-time bias update are skipped.
    _own_bias = True

    def __init__(self, cfg: ImuConfig):
        self.cfg = cfg
        self._forward_axis = np.array(_AXES[cfg.forward_axis])
        self.stillness = Stillness(cfg.still_window_s, cfg.still_gyro_range_dps, cfg.still_acc_range_g)
        # Set by the engine while the headset is not worn (IDLE): learn the
        # gyro bias whenever it is still.
        self.learn_idle_bias = False
        # Gyro samples at full scale since the engine started (not reset).
        self.clipped_samples = 0
        self._clip_burst: list[float] = []  # [first t, last t, count] of the current burst
        self._forget_idle_bias()
        self.reset()

    def reset(self, keep_idle_bias: bool = False) -> None:
        """Forget the orientation (used after a dropout or a headset change).

        `keep_idle_bias`: keep the bias learned off the head (same headset).
        """
        if not keep_idle_bias:
            self._forget_idle_bias()
        self.up: np.ndarray | None = None  # world up, in sensor coordinates
        self.bias = np.zeros(3)
        self.bias_source = "none"  # where the calibration bias came from: "idle", "worn" or "vqf"
        self.calibrated = False
        self._up0: np.ndarray | None = None
        self._fwd: np.ndarray | None = None  # head forward, sensor coordinates
        self._left: np.ndarray | None = None  # head left, sensor coordinates
        self.yaw_deg = 0.0
        self.yaw_rate = 0.0  # deg/s, + = turning right
        self.pitch_rate = 0.0  # deg/s, + = tilting up
        self.last_gyro: np.ndarray | None = None  # raw, deg/s
        self._t_gyro: float | None = None
        self._t_acc: float | None = None
        self.stillness.clear()

    # --- updates -----------------------------------------------------------

    def update(self, s: ImuSample) -> None:
        v = np.array((s.x, s.y, s.z), dtype=float)
        if s.kind == "gyro":
            self._update_gyro(s.t, v)
        else:
            self._update_acc(s.t, v)

    def _update_gyro(self, t: float, w: np.ndarray) -> None:
        if self._t_gyro is not None and t <= self._t_gyro:
            return  # out of order / duplicate
        dt = None if self._t_gyro is None else t - self._t_gyro
        self._t_gyro = t
        self.last_gyro = w
        self._count_clipping(t, w)
        self.stillness.add_gyro(t, w)
        self._track_gyro(w, dt)
        if dt is None or dt > MAX_DT_S:
            return

        wc = w - self.bias
        yaw_rate = -float(wc @ self.up) if self.up is not None else 0.0
        left = self._lateral()
        pitch_rate = -float(wc @ left) if left is not None else 0.0
        if self.cfg.invert_yaw:
            yaw_rate = -yaw_rate
        if self.cfg.invert_pitch:
            pitch_rate = -pitch_rate
        self.yaw_rate = yaw_rate
        self.pitch_rate = pitch_rate

        still = self.stillness.is_still()
        if still and self.learn_idle_bias and self._own_bias:
            self._learn_idle_bias(t, w, dt)
            if not self.calibrated:
                # Track "up" with it until calibration picks the bias for good.
                idle = self.idle_bias()
                if idle is not None:
                    self.bias = idle
        frozen = False
        if self.calibrated and still:
            mean = self.stillness.gyro_mean()
            if mean is not None and _norm(mean - self.bias) < self.cfg.bias_update_max_dps:
                if self._own_bias:
                    k = dt / (self.cfg.bias_tau_s + dt)
                    self.bias = self.bias + k * (mean - self.bias)
                frozen = self.cfg.freeze_yaw_when_still
        if not frozen:
            self.yaw_deg += yaw_rate * dt

    def _update_acc(self, t: float, a: np.ndarray) -> None:
        if self._t_acc is not None and t <= self._t_acc:
            return
        dt = None if self._t_acc is None else t - self._t_acc
        self._t_acc = t
        norm = _norm(a)
        self.stillness.add_acc(t, norm)
        if norm < 0.3:
            return  # free fall or garbage; nothing to learn about "up"
        self._track_acc(a, norm, dt)

    def _count_clipping(self, t: float, w: np.ndarray) -> None:
        burst = self._clip_burst
        if max(abs(x) for x in w.tolist()) >= GYRO_CLIP_DPS:
            self.clipped_samples += 1
            if burst:
                burst[1], burst[2] = t, burst[2] + 1
            else:
                self._clip_burst = [t, t, 1]
        elif burst and t - burst[1] > _CLIP_BURST_GAP_S:
            log.info("gyro at full scale (245 deg/s) on %d samples over %.2f s: "
                     "yaw may have slipped until the next shake or selection",
                     burst[2], burst[1] - burst[0])
            self._clip_burst = []

    # --- how "up" is tracked (the part a backend replaces) -------------------

    def _track_gyro(self, w: np.ndarray, dt: float | None) -> None:
        """Every new gyro sample (raw, deg/s); dt is None or > MAX_DT_S after a gap."""
        if dt is None or dt > MAX_DT_S or self.up is None:
            return
        # A world-fixed vector seen from the rotating head: dv/dt = -w x v.
        wc = np.radians(w - self.bias)
        self.up = _unit(self.up - _cross(wc, self.up) * dt)

    def _track_acc(self, a: np.ndarray, norm: float, dt: float | None) -> None:
        """Every new accelerometer sample (g) that is not garbage."""
        meas = a / norm
        if self.up is None:
            self.up = meas
            return
        if dt is None or dt > MAX_DT_S:
            dt = 1.0 / 52.0
        if abs(norm - 1.0) <= self.cfg.acc_trust_band_g:
            k = dt / (self.cfg.gravity_tau_s + dt)
            self.up = _unit(self.up + k * (meas - self.up))

    def _lateral(self) -> np.ndarray | None:
        if self._left is not None:
            return self._left
        if self.up is None:
            return None
        fwd = self._forward_axis - (self._forward_axis @ self.up) * self.up
        if _norm(fwd) < 0.3:
            return None
        return _cross(self.up, _unit(fwd))

    # --- gyro bias learned off the head ---------------------------------------

    def _forget_idle_bias(self) -> None:
        self._idle_bias: np.ndarray | None = None
        self._idle_bias_t: float | None = None  # last update, sample clock
        self._idle_still_s = 0.0  # stillness accumulated into it

    def _learn_idle_bias(self, t: float, w: np.ndarray, dt: float) -> None:
        # A plain mean at first, then an exponential one over idle_bias_tau_s.
        self._idle_still_s += dt
        k = dt / min(self._idle_still_s, self.cfg.idle_bias_tau_s)
        prev = w if self._idle_bias is None else self._idle_bias
        self._idle_bias = prev + k * (w - prev)
        self._idle_bias_t = t

    def stand_still_s(self) -> float:
        """Seconds of stillness off the head learned into the bias so far (the
        bias is used once this reaches imu.idle_bias_min_s)."""
        return self._idle_still_s

    def idle_bias(self) -> np.ndarray | None:
        """The bias learned off the head, or None if there is none usable now."""
        c = self.cfg
        if (c.idle_bias_max_age_s <= 0 or self._idle_bias is None or self._idle_bias_t is None
                or self._t_gyro is None or self._idle_still_s < c.idle_bias_min_s):
            return None
        if self._t_gyro - self._idle_bias_t > c.idle_bias_max_age_s:
            return None
        if np.linalg.norm(self._idle_bias) > c.max_bias_dps:
            return None
        return self._idle_bias.copy()

    def _calibration_bias(self, worn_mean: np.ndarray) -> tuple[np.ndarray, str]:
        """The bias to start tracking with, and where it came from."""
        idle = self.idle_bias()
        if idle is None:
            return worn_mean.copy(), "worn"
        # Further apart than the in-session update can correct: the headset was
        # power-cycled, or the head was turning. Today's behaviour is the safe side.
        diff = float(np.linalg.norm(idle - worn_mean))
        if diff >= self.cfg.bias_update_max_dps:
            log.warning("idle gyro bias is %.1f deg/s away from the worn mean; using the worn mean", diff)
            return worn_mean.copy(), "worn"
        return idle, "idle"

    # --- calibration and corrections ----------------------------------------

    def calibrate(self, gyro_mean: np.ndarray, acc_mean: np.ndarray) -> None:
        """Define the neutral pose: looking at the centre of the middle screen.

        `gyro_mean` / `acc_mean` are averaged over the calibration stillness.
        The bias starts from the idle bias when there is a usable one.
        """
        gyro_mean = np.asarray(gyro_mean, dtype=float)
        if np.linalg.norm(gyro_mean) > self.cfg.max_bias_dps:
            raise CalibrationError(
                f"gyro bias {np.linalg.norm(gyro_mean):.1f} deg/s is implausible (head was moving?)"
            )
        up = _unit(np.asarray(acc_mean, dtype=float))
        fwd = self._forward_axis - (self._forward_axis @ up) * up
        if np.linalg.norm(fwd) < 0.3:
            raise CalibrationError(
                "the configured forward axis is nearly vertical; check imu.forward_axis"
            )
        fwd = _unit(fwd)
        self.bias, self.bias_source = self._calibration_bias(gyro_mean)
        self.up = up
        self._up0 = up
        self._fwd = fwd
        self._left = np.cross(up, fwd)
        self.yaw_deg = 0.0
        self.calibrated = True

    def zero_yaw(self) -> None:
        self.yaw_deg = 0.0

    def reanchor_yaw(self, centre_deg: float, gain: float) -> None:
        self.yaw_deg += gain * (centre_deg - self.yaw_deg)

    # --- outputs ------------------------------------------------------------

    @property
    def pitch_deg(self) -> float:
        if not self.calibrated or self.up is None:
            return 0.0
        p = math.degrees(math.asin(max(-1.0, min(1.0, float(self._fwd @ self.up)))))
        return -p if self.cfg.invert_pitch else p

    @property
    def roll_deg(self) -> float:
        if not self.calibrated or self.up is None:
            return 0.0
        return math.degrees(math.asin(max(-1.0, min(1.0, float(self._left @ self.up)))))

    def tilt_from_calibration_deg(self) -> float | None:
        """Angle between "up" now and at calibration (large = headset taken off)."""
        if self._up0 is None or self.up is None:
            return None
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self._up0 @ self.up)))))


def make_fusion(cfg: ImuConfig) -> Fusion:
    """The orientation backend chosen by `imu.backend`."""
    if cfg.backend == "vqf":
        try:
            from .fusion_vqf import VqfFusion
        except ModuleNotFoundError as e:
            if e.name != "vqf":
                raise
            raise ConfigError("imu.backend = 'vqf' needs the vqf package: pip install -e . "
                              "(or set imu.backend = 'complementary')") from e
        return VqfFusion(cfg)
    if cfg.backend != "complementary":
        raise ConfigError(f"unknown imu.backend {cfg.backend!r}")
    return Fusion(cfg)

"""VQF orientation backend (`imu.backend = "vqf"`). Needs the optional `vqf` package.

VQF (Laidig & Seel, Information Fusion 2023; https://vqf.readthedocs.io)
filters the accelerometer in a near-inertial frame and estimates the gyro bias
with a Kalman filter, at rest and during motion. Only two things change
compared with `Fusion`:

- `up` (world up in sensor coordinates) comes from VQF's 6D quaternion;
- with `imu.vqf_bias_from = "vqf"`, `bias` is VQF's estimate. VQF runs from
  the moment data arrives, so it learns the bias while the headset rests on
  its stand. Calibration keeps that estimate under the same rules as the idle
  bias (rest seen off the head, recent, close to the worn mean) and when VQF
  is confident about it; otherwise it starts from the worn mean.

Everything else is `Fusion`'s: yaw integrated about `up`, the zero-velocity
update, re-anchoring, calibration of the neutral pose, outputs and sign
conventions. Heading is unobservable without a magnetometer, so yaw drifts as
before.

Units: `ImuSample` is in deg/s and g; VQF wants rad/s and m/s^2, and its bias
getter/setter work in rad/s (its parameters are in deg/s).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from vqf import VQF

from .config import ImuConfig
from .fusion import MAX_DT_S, Fusion

log = logging.getLogger(__name__)

G = 9.80665  # m/s^2 per g
# Muse 2 IMU rate. VQF assumes a fixed sample time; missing gyro samples are
# filled in by repeating the latest one (see `_track_gyro`).
NOMINAL_RATE_HZ = 52.0
# VQF's default biasForgettingTime: its bias variance grows by (0.1 deg/s)^2 in
# this time without updates. Used to age a bias carried across a dropout.
_BIAS_FORGETTING_S = 100.0
# Uncertainty given to the worn mean when calibration falls back to it
# (VQF's default initial bias uncertainty).
_FALLBACK_SIGMA_DPS = 0.5


class VqfFusion(Fusion):
    def __init__(self, cfg: ImuConfig):
        self._ts = 1.0 / NOMINAL_RATE_HZ
        self._vqf_bias = cfg.vqf_bias_from == "vqf"
        self._own_bias = not self._vqf_bias
        self._vqf = VQF(
            self._ts,
            tauAcc=cfg.vqf_tau_acc_s,
            restThGyr=cfg.vqf_rest_th_gyr_dps,
            restThAcc=cfg.vqf_rest_th_acc_ms2,
            restMinT=cfg.vqf_rest_min_s,
            biasClip=cfg.vqf_bias_clip_dps,
            restBiasEstEnabled=self._vqf_bias,
            motionBiasEstEnabled=self._vqf_bias,
            magDistRejectionEnabled=False,
        )
        self._has_acc = False  # VQF has seen an accelerometer sample since its last reset
        self._carry: tuple[np.ndarray, float, float] | None = None  # (bias rad/s, sigma rad/s, t)
        self.bias_sigma_dps: float | None = None
        super().__init__(cfg)

    def _forget_idle_bias(self) -> None:
        super()._forget_idle_bias()
        self._stand_rest_s = 0.0  # VQF rest seen while the engine is idle (off the head)
        self._stand_rest_t: float | None = None

    def reset(self, keep_idle_bias: bool = False) -> None:
        carry = None
        if keep_idle_bias and self._vqf_bias:
            t_last = getattr(self, "_t_gyro", None)
            if t_last is not None:
                b, s = self._vqf.getBiasEstimate()
                carry = (b.copy(), float(s), t_last)
            else:
                carry = self._carry  # reset twice in a row: keep what we carried
        super().reset(keep_idle_bias)
        self._vqf.resetState()
        self._has_acc = False
        self._carry = carry
        self.bias_sigma_dps = None

    # --- how "up" (and the bias) are tracked ---------------------------------------

    def _track_gyro(self, w: np.ndarray, dt: float | None) -> None:
        if dt is not None and dt > MAX_DT_S:
            self._restart()
        if self._carry is not None:
            self._apply_carry()
        if not self._vqf_bias:
            self._vqf.setBiasEstimate(np.radians(self.bias), -1.0)
        # Gyro samples lost in transit (or dropped as duplicate timestamps):
        # repeat the latest one so VQF's fixed-rate integration keeps up.
        steps = 1 if dt is None or dt > MAX_DT_S else max(1, round(dt / self._ts))
        g = np.ascontiguousarray(np.radians(w), dtype=np.float64)
        for _ in range(steps):
            self._vqf.updateGyr(g)
        if self._vqf_bias:
            b, s = self._vqf.getBiasEstimate()
            self.bias = np.degrees(b)
            self.bias_sigma_dps = math.degrees(s)
            if self.learn_idle_bias and dt is not None and dt <= MAX_DT_S and self._vqf.getRestDetected():
                self._stand_rest_s += dt
                self._stand_rest_t = self._t_gyro
        if self._has_acc:
            self.up = self._vqf_up()

    def _track_acc(self, a: np.ndarray, norm: float, dt: float | None) -> None:
        self._vqf.updateAcc(np.ascontiguousarray(a * G, dtype=np.float64))
        self._has_acc = True
        self.up = self._vqf_up()

    def _vqf_up(self) -> np.ndarray:
        # q = [w, x, y, z], sensor -> earth (z up). World up in sensor
        # coordinates is the third row of its rotation matrix.
        w, x, y, z = self._vqf.getQuat6D()
        return np.array((2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)))

    def _restart(self) -> None:
        """After a gap: VQF assumes a fixed sample time, so start it again, keeping the bias."""
        b, s = self._vqf.getBiasEstimate()
        self._vqf.resetState()
        self._vqf.setBiasEstimate(np.ascontiguousarray(b, dtype=np.float64), float(s))
        self._has_acc = False

    def _apply_carry(self) -> None:
        b, s, t_prev = self._carry
        self._carry = None
        gap = max(0.0, (self._t_gyro or t_prev) - t_prev)
        sigma_dps = math.sqrt(math.degrees(s) ** 2 + 0.1 ** 2 * gap / _BIAS_FORGETTING_S)
        self._vqf.setBiasEstimate(np.ascontiguousarray(b, dtype=np.float64), math.radians(sigma_dps))

    # --- calibration ---------------------------------------------------------------------

    def stand_still_s(self) -> float:
        return self._stand_rest_s if self._vqf_bias else super().stand_still_s()

    def _calibration_bias(self, worn_mean: np.ndarray) -> tuple[np.ndarray, str]:
        if not self._vqf_bias:
            return super()._calibration_bias(worn_mean)
        c = self.cfg
        if float(np.max(np.abs(worn_mean))) > 0.9 * c.vqf_bias_clip_dps:
            log.warning("gyro bias %s deg/s is at or above imu.vqf_bias_clip_dps (%g): VQF cannot "
                        "follow it, so pitch will be off by a few degrees; raise vqf_bias_clip_dps",
                        np.round(worn_mean, 1).tolist(), c.vqf_bias_clip_dps)
        b, s = self._vqf.getBiasEstimate()
        bias, sigma = np.degrees(b), math.degrees(s)
        # VQF also "rests" on a swaying head and then grows confident too
        # early, so only rest seen off the head counts, as for the idle bias.
        stand = (c.idle_bias_max_age_s > 0 and self._stand_rest_t is not None
                 and self._stand_rest_s >= c.idle_bias_min_s
                 and self._t_gyro - self._stand_rest_t <= c.idle_bias_max_age_s)
        if (stand and sigma <= c.vqf_bias_max_sigma_dps and np.linalg.norm(bias) <= c.max_bias_dps
                and np.linalg.norm(bias - worn_mean) < c.bias_update_max_dps):
            return bias, "vqf"
        self._vqf.setBiasEstimate(np.ascontiguousarray(np.radians(worn_mean), dtype=np.float64),
                                  math.radians(_FALLBACK_SIGMA_DPS))
        return worn_mean.copy(), "worn"

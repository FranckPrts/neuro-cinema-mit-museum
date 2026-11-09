"""A virtual head wearing a virtual Muse: turns poses into raw IMU readings.

Used by the simulator source and by the tests, so the whole pipeline (fusion,
gestures, state machine) runs on realistic signals: gyro bias and noise,
gravity seen through a band that sits crooked on the head, small postural sway.

World frame: x forward (towards the middle screen), y left, z up.
Pose: yaw + = right, pitch + = up, roll + = right ear down (degrees).
"""

from __future__ import annotations

import math

import numpy as np

RATE_HZ = 52.0
# Full scale of the Muse gyro: 16-bit readings at 0.0074768 deg/s per step
# (muselsl's scale factor). Faster motion reads as exactly this value.
GYRO_LIMIT_DPS = 245.0
# Putting the headset on or taking it off moves it continuously for this long.
MOVE_S = 1.0


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """World-from-body rotation for a pose in degrees (conventions above):
    Rz(-yaw) @ Ry(-pitch) @ Rx(roll), written out (it runs on every sample)."""
    a, b, c = -math.radians(yaw), -math.radians(pitch), math.radians(roll)
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return np.array((
        (ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc),
        (sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc),
        (-sb, cb * sc, cb * cc),
    ))


def _exp_so3(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (radians)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def _log_so3(r: np.ndarray) -> np.ndarray:
    """Rotation vector (radians) of a rotation matrix."""
    cos_angle = max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2.0
    if angle < 1e-6:
        return skew
    s = math.sin(angle)
    if s < 1e-6:  # ~180 degrees: any axis will do for a simulator
        return np.array([angle, 0.0, 0.0])
    return skew * (angle / s)


class VirtualHead:
    def __init__(
        self,
        seed: int = 0,
        mount: tuple[float, float, float] = (3.0, 12.0, -4.0),
        gyro_bias_dps: tuple[float, float, float] = (0.9, -1.3, 0.6),
        gyro_noise_dps: float = 0.15,
        acc_noise_g: float = 0.004,
        bias_walk_dps: float = 0.02,
        sway: bool = True,
        worn: bool = True,
        lever_arm_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.rng = np.random.default_rng(seed)
        # How the sensor sits on the head (yaw, pitch, roll offsets, degrees).
        self.r_head_sensor = rotation(*mount)
        # Sensor position relative to the neck pivot, head frame (x forward,
        # y left, z up), metres. Non-zero adds the linear acceleration of head
        # rotation to the accelerometer (about (0.10, 0, 0.12) on a real head).
        self.lever_arm = np.array(lever_arm_m, dtype=float)
        self._p_hist: list[np.ndarray] = []
        self.r_table = rotation(25.0, 0.0, 90.0)  # lying on its side
        self.bias = np.array(gyro_bias_dps, dtype=float)
        self.gyro_noise = gyro_noise_dps
        self.acc_noise = acc_noise_g
        self.bias_walk = bias_walk_dps
        self.sway = sway
        self.worn = worn

        self.yaw = 0.0
        self.pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.tau_s = 0.18
        self.max_speed_dps = 120.0
        self._gestures: list[tuple[str, float, float]] = []  # (kind, start, amplitude deg)
        self._put_on_start: float | None = None
        self._move_from: np.ndarray | None = None  # orientation when put on / taken off
        self._move_start = 0.0
        self._t: float | None = None
        self._r_prev: np.ndarray | None = None

    # --- control ------------------------------------------------------------

    def look(self, yaw: float | None = None, pitch: float | None = None) -> None:
        if yaw is not None:
            self.target_yaw = yaw
        if pitch is not None:
            self.target_pitch = pitch

    def nod(self, t: float) -> None:
        self._gestures.append(("nod", t, 16.0))

    def shake(self, t: float, amplitude_deg: float = 14.0) -> None:
        """Peak speed is about 15.7 x amplitude deg/s: the default peaks near
        220 deg/s, 20 deg or more goes past the gyro's full scale."""
        self._gestures.append(("shake", t, amplitude_deg))

    def set_worn(self, worn: bool, t: float) -> None:
        if worn != self.worn and self._r_prev is not None:
            self._move_from, self._move_start = self._r_prev, t
        if worn and not self.worn:
            self._put_on_start = t
            self.yaw = self.pitch = 0.0
            self.target_yaw = self.target_pitch = 0.0
        self.worn = worn

    # --- kinematics -----------------------------------------------------------

    def pose(self, t: float) -> tuple[float, float, float]:
        yaw, pitch, roll = self.yaw, self.pitch, 0.0
        if self.sway:
            yaw += 0.4 * math.sin(2 * math.pi * 0.23 * t)
            pitch += 0.3 * math.sin(2 * math.pi * 0.31 * t + 1.0)
            roll += 0.3 * math.sin(2 * math.pi * 0.17 * t + 2.0)
        for kind, t0, amp in self._gestures:
            tau = t - t0
            if kind == "nod" and 0.0 <= tau <= 0.6:
                # Down and back up: two velocity lobes, peak ~85 deg/s.
                pitch -= amp * (1 - math.cos(2 * math.pi * tau / 0.6)) / 2
            elif kind == "shake" and 0.0 <= tau <= 1.2:
                # Left-right oscillation under a smooth envelope.
                yaw += amp * math.sin(2 * math.pi * 2.5 * tau) * math.sin(math.pi * tau / 1.2) ** 2
        if self._put_on_start is not None:
            tau = t - self._put_on_start
            if tau <= 1.5:
                k = (1.5 - tau) / 1.5
                yaw += 35.0 * k * math.sin(2 * math.pi * 1.1 * tau)
                pitch += 25.0 * k * math.sin(2 * math.pi * 0.9 * tau + 0.5)
                roll += 20.0 * k * math.sin(2 * math.pi * 1.3 * tau + 1.0)
        return yaw, pitch, roll

    def _advance(self, dt: float) -> None:
        k = 1.0 - math.exp(-dt / self.tau_s)
        limit = self.max_speed_dps * dt
        for name in ("yaw", "pitch"):
            cur = getattr(self, name)
            step = (getattr(self, "target_" + name) - cur) * k
            setattr(self, name, cur + max(-limit, min(limit, step)))
        self._gestures = [g for g in self._gestures if self._t is None or self._t - g[1] < 2.0]

    def step(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Readings at time t: (acc in g, gyro in deg/s), both in the sensor frame."""
        dt = 1.0 / RATE_HZ if self._t is None else max(t - self._t, 1e-6)
        self._advance(dt)
        self._t = t
        # One draw per step: bias walk, gyro noise, acc noise (same stream as
        # three rng.normal calls, cheaper).
        z = self.rng.standard_normal(9)
        self.bias = self.bias + z[:3] * (self.bias_walk * math.sqrt(dt))

        r = rotation(*self.pose(t)) @ self.r_head_sensor if self.worn else self.r_table
        if self._move_from is not None:  # on its way to / from the head
            x = (t - self._move_start) / MOVE_S
            if x >= 1.0:
                self._move_from = None
            else:
                x = max(0.0, x)
                r = self._move_from @ _exp_so3(x * x * (3 - 2 * x) * _log_so3(self._move_from.T @ r))

        lin_g = np.zeros(3)  # linear acceleration of the sensor, world frame, in g
        if self.worn and self.lever_arm.any():
            self._p_hist = (self._p_hist + [r @ self.r_head_sensor.T @ self.lever_arm])[-3:]
            if len(self._p_hist) == 3:
                p0, p1, p2 = self._p_hist
                lin_g = (p2 - 2.0 * p1 + p0) / (dt * dt) / 9.80665
        else:
            self._p_hist = []

        if self._r_prev is None:
            omega = np.zeros(3)
        else:
            omega = np.degrees(_log_so3(self._r_prev.T @ r)) / dt
        self._r_prev = r

        noise_scale = 1.0 if self.worn else 0.4
        gyro = omega + self.bias + z[3:6] * (self.gyro_noise * noise_scale)
        gyro = np.clip(gyro, -GYRO_LIMIT_DPS, GYRO_LIMIT_DPS)
        acc = r.T @ (np.array([0.0, 0.0, 1.0]) + lin_g) + z[6:] * self.acc_noise
        return acc, gyro

    def up_in_sensor(self) -> np.ndarray:
        """Ground truth for the last `step`: world up in sensor coordinates."""
        return self._r_prev.T @ np.array([0.0, 0.0, 1.0])

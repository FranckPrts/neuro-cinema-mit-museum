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
GYRO_LIMIT_DPS = 500.0


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """World-from-body rotation for a pose in degrees (conventions above)."""
    return _rz(-math.radians(yaw)) @ _ry(-math.radians(pitch)) @ _rx(math.radians(roll))


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
    ):
        self.rng = np.random.default_rng(seed)
        # How the sensor sits on the head (yaw, pitch, roll offsets, degrees).
        self.r_head_sensor = rotation(*mount)
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
        self._gestures: list[tuple[str, float]] = []
        self._put_on_start: float | None = None
        self._t: float | None = None
        self._r_prev: np.ndarray | None = None

    # --- control ------------------------------------------------------------

    def look(self, yaw: float | None = None, pitch: float | None = None) -> None:
        if yaw is not None:
            self.target_yaw = yaw
        if pitch is not None:
            self.target_pitch = pitch

    def nod(self, t: float) -> None:
        self._gestures.append(("nod", t))

    def shake(self, t: float) -> None:
        self._gestures.append(("shake", t))

    def set_worn(self, worn: bool, t: float) -> None:
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
        for kind, t0 in self._gestures:
            tau = t - t0
            if kind == "nod" and 0.0 <= tau <= 0.6:
                # Down and back up: two velocity lobes, peak ~85 deg/s.
                pitch -= 16.0 * (1 - math.cos(2 * math.pi * tau / 0.6)) / 2
            elif kind == "shake" and 0.0 <= tau <= 1.2:
                # Left-right oscillation under a smooth envelope.
                yaw += 14.0 * math.sin(2 * math.pi * 2.5 * tau) * math.sin(math.pi * tau / 1.2) ** 2
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
        self._gestures = [(k_, t0) for k_, t0 in self._gestures if self._t is None or self._t - t0 < 2.0]

    def step(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Readings at time t: (acc in g, gyro in deg/s), both in the sensor frame."""
        dt = 1.0 / RATE_HZ if self._t is None else max(t - self._t, 1e-6)
        self._advance(dt)
        self._t = t
        self.bias = self.bias + self.rng.normal(0.0, self.bias_walk * math.sqrt(dt), 3)

        if self.worn:
            r = rotation(*self.pose(t)) @ self.r_head_sensor
        else:
            r = self.r_table

        if self._r_prev is None:
            omega = np.zeros(3)
        else:
            omega = np.degrees(_log_so3(self._r_prev.T @ r)) / dt
        self._r_prev = r

        noise_scale = 1.0 if self.worn else 0.4
        gyro = omega + self.bias + self.rng.normal(0.0, self.gyro_noise * noise_scale, 3)
        gyro = np.clip(gyro, -GYRO_LIMIT_DPS, GYRO_LIMIT_DPS)
        acc = r.T @ np.array([0.0, 0.0, 1.0]) + self.rng.normal(0.0, self.acc_noise, 3)
        return acc, gyro

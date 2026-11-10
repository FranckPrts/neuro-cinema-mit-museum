import functools
import importlib.util
import math

import numpy as np
import pytest

from neurocinema.config import Config, ConfigError
from neurocinema.fusion import CalibrationError, Calibrator, make_fusion
from neurocinema.samples import ImuSample
from neurocinema.sim.head import RATE_HZ, VirtualHead

# Orientation backends under test. "vqf-stillness" is VQF for "up" with the
# complementary filter's own bias logic (imu.vqf_bias_from = "stillness").
needs_vqf = pytest.mark.skipif(importlib.util.find_spec("vqf") is None,
                               reason="vqf not installed: pip install -e '.[vqf]'")
VQF = pytest.param("vqf", marks=needs_vqf)
VQF_STILLNESS = pytest.param("vqf-stillness", marks=needs_vqf)
ALL = ["complementary", VQF, VQF_STILLNESS]
OWN_BIAS = ["complementary", VQF_STILLNESS]  # idle-bias logic is the filter's own


def make_cfg(variant: str = "complementary") -> Config:
    cfg = Config()
    cfg.imu.backend = "complementary" if variant == "complementary" else "vqf"
    if variant == "vqf-stillness":
        cfg.imu.vqf_bias_from = "stillness"
    return cfg


@pytest.fixture(params=ALL)
def variant(request):
    return request.param


class FusionRig:
    def __init__(self, head: VirtualHead, cfg: Config | None = None, keep_every: int = 1):
        self.cfg = cfg or Config()
        self.f = make_fusion(self.cfg.imu)
        self.cal = Calibrator(self.cfg.calibration.still_s)
        self.head = head
        self.t = 0.0
        self.keep_every = keep_every  # >1: lose samples in transit
        self._n = 0

    def run(self, seconds: float, each=None) -> None:
        for _ in range(int(seconds * RATE_HZ)):
            acc, gyro = self.head.step(self.t)
            self._n += 1
            if self._n % self.keep_every == 0:
                for s in (ImuSample(self.t, "acc", *acc), ImuSample(self.t, "gyro", *gyro)):
                    self.f.update(s)
                    if not self.f.calibrated:
                        self.cal.add(s, self.f.stillness.is_still())
            if each is not None:
                each()
            self.t += 1.0 / RATE_HZ

    def calibrate(self) -> None:
        while not self.cal.ready(self.t):
            self.run(0.1)
        self.f.calibrate(*self.cal.result())

    def tilt_error_deg(self) -> float:
        """Angle between estimated and true "up"."""
        return math.degrees(math.acos(min(1.0, float(self.f.up @ self.head.up_in_sensor()))))


@pytest.mark.parametrize("mount", [(0, 0, 0), (3, 12, -4), (-8, 25, 10)])
def test_turns_and_tilts_are_measured_through_a_crooked_band(mount, variant):
    rig = FusionRig(VirtualHead(seed=2, mount=mount), make_cfg(variant))
    rig.calibrate()
    rig.head.look(yaw=40, pitch=15)
    rig.run(3)
    assert rig.f.yaw_deg == pytest.approx(40, abs=3)
    assert rig.f.pitch_deg == pytest.approx(15, abs=2)
    rig.head.look(yaw=-40, pitch=-15)
    rig.run(3)
    assert rig.f.yaw_deg == pytest.approx(-40, abs=3)
    assert rig.f.pitch_deg == pytest.approx(-15, abs=2)


class Replayed:
    """Readings recorded from a VirtualHead, served again in order (step(t)
    ignores t: the rig asks for them at the same times)."""

    def __init__(self, readings):
        self._it = iter(readings)

    def step(self, t):
        return next(self._it)


@functools.cache
def _still_head_readings(seconds: float) -> tuple:
    head = VirtualHead(seed=3)
    return tuple(head.step(i / RATE_HZ) for i in range(int(seconds * RATE_HZ)))


@pytest.mark.parametrize("variant", ["complementary", VQF])  # the slowest test: skip vqf-stillness
def test_yaw_holds_while_still_for_five_minutes(variant):
    rig = FusionRig(Replayed(_still_head_readings(310.0)), make_cfg(variant))  # simulated once for both
    rig.calibrate()
    rig.run(300)
    assert abs(rig.f.yaw_deg) < 2


def test_yaw_drift_while_moving_stays_small_over_a_minute(variant):
    rig = FusionRig(VirtualHead(seed=4), make_cfg(variant))
    rig.calibrate()
    for i in range(10):
        rig.head.look(yaw=[-40, 0, 40][i % 3], pitch=[10, -10][i % 2])
        rig.run(6)
    rig.head.look(yaw=0, pitch=0)
    rig.run(3)
    assert abs(rig.f.yaw_deg) < 5


def test_bias_is_learned_at_calibration(variant):
    head = VirtualHead(seed=5, sway=False)
    rig = FusionRig(head, make_cfg(variant))
    rig.calibrate()
    assert np.allclose(rig.f.bias, head.bias, atol=0.15)


def test_calibration_rejects_forward_axis_pointing_up(variant):
    cfg = make_cfg(variant)
    cfg.imu.forward_axis = "z"
    f = make_fusion(cfg.imu)
    with pytest.raises(CalibrationError):
        f.calibrate(np.zeros(3), np.array([0.0, 0.0, 1.0]))


def test_calibration_rejects_implausible_bias(variant):
    f = make_fusion(make_cfg(variant).imu)
    with pytest.raises(CalibrationError):
        f.calibrate(np.array([40.0, 0, 0]), np.array([0.0, 0.0, 1.0]))


def test_out_of_order_and_gapped_samples_do_not_corrupt_yaw(variant):
    f = make_fusion(make_cfg(variant).imu)
    f.calibrate(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    f.update(ImuSample(1.0, "gyro", 0, 0, 0))
    f.update(ImuSample(0.5, "gyro", 0, 0, -500))  # older: dropped
    f.update(ImuSample(3.0, "gyro", 0, 0, -500))  # after a 2 s gap: not integrated
    assert f.yaw_deg == 0.0


def test_two_of_three_samples_lost_still_tracks(variant):
    # What the LSL path delivers today: muselsl stamps a packet's 3 samples
    # alike, and duplicates are dropped.
    rig = FusionRig(VirtualHead(seed=6, mount=(3, 12, -4)), make_cfg(variant), keep_every=3)
    rig.calibrate()
    rig.head.look(yaw=40, pitch=15)
    rig.run(3)
    assert rig.f.yaw_deg == pytest.approx(40, abs=3)
    assert rig.f.pitch_deg == pytest.approx(15, abs=2)


def test_unknown_backend_is_rejected():
    cfg = Config()
    cfg.imu.backend = "kalman"
    with pytest.raises(ConfigError):
        make_fusion(cfg.imu)


# --- gyro bias learned while the headset lies still off the head ---------------


def _table_then_worn(rig: FusionRig, table_s: float = 20.0) -> None:
    """Headset on the table (engine idle), then put on; ready to calibrate."""
    rig.f.learn_idle_bias = True
    rig.run(table_s)
    rig.f.learn_idle_bias = False
    rig.head.set_worn(True, rig.t)
    rig.run(1.6)  # put-on motion
    rig.cal.reset()


@pytest.mark.parametrize("variant", OWN_BIAS)
def test_idle_bias_is_learned_on_the_table(variant):
    head = VirtualHead(seed=20, worn=False, bias_walk_dps=0.0)
    rig = FusionRig(head, make_cfg(variant))
    rig.f.learn_idle_bias = True
    rig.run(30)
    assert np.allclose(rig.f.idle_bias(), head.bias, atol=0.05)


@pytest.mark.parametrize("variant", OWN_BIAS)
def test_idle_bias_is_not_used_until_enough_stillness(variant):
    rig = FusionRig(VirtualHead(seed=21, worn=False), make_cfg(variant))
    rig.f.learn_idle_bias = True
    rig.run(rig.cfg.imu.idle_bias_min_s - 1.0)
    assert rig.f.idle_bias() is None
    rig.run(2.0)
    assert rig.f.idle_bias() is not None


def test_table_then_worn_starts_from_the_bias_learned_on_the_table(variant):
    head = VirtualHead(seed=23, worn=False)
    rig = FusionRig(head, make_cfg(variant))
    _table_then_worn(rig)
    rig.calibrate()
    assert rig.f.bias_source == ("vqf" if variant == "vqf" else "idle")
    assert np.linalg.norm(rig.f.bias - head.bias) < 0.15


def test_calibration_uses_the_table_bias_while_the_head_turns_slowly(variant):
    head = VirtualHead(seed=22, worn=False)
    rig = FusionRig(head, make_cfg(variant))
    _table_then_worn(rig)
    # A slow drift of gaze during "keep still": still by the stillness test,
    # but it spoils the 2 s worn mean by about 1 deg/s.
    head.max_speed_dps = 1.0
    head.look(yaw=30)
    rig.calibrate()
    worn_mean = rig.cal.result()[0]
    assert np.linalg.norm(worn_mean - head.bias) > 0.5
    assert rig.f.bias_source in ("idle", "vqf")
    assert np.linalg.norm(rig.f.bias - head.bias) < 0.2


@pytest.mark.parametrize("variant", ["complementary", VQF])
def test_stale_table_bias_is_not_used(variant):
    cfg = make_cfg(variant)
    cfg.imu.idle_bias_max_age_s = 5.0
    rig = FusionRig(VirtualHead(seed=24, worn=False), cfg)
    _table_then_worn(rig)
    rig.run(5.0)  # worn, too long after the table
    rig.cal.reset()
    rig.calibrate()
    assert rig.f.bias_source == "worn"


@pytest.mark.parametrize("variant", ["complementary", VQF])
def test_table_bias_off_when_max_age_is_zero(variant):
    cfg = make_cfg(variant)
    cfg.imu.idle_bias_max_age_s = 0.0
    rig = FusionRig(VirtualHead(seed=25, worn=False), cfg)
    _table_then_worn(rig)
    rig.calibrate()
    assert rig.f.bias_source == "worn"


@pytest.mark.parametrize("variant", ["complementary", VQF])
def test_table_bias_far_from_the_worn_mean_is_not_used(variant):
    head = VirtualHead(seed=26, worn=False)
    rig = FusionRig(head, make_cfg(variant))
    _table_then_worn(rig)
    head.bias = head.bias + np.array([0.0, 0.0, 3.0])  # e.g. power-cycled: bias moved
    rig.calibrate()
    assert rig.f.bias_source == "worn"
    assert np.linalg.norm(rig.f.bias - head.bias) < 0.6


@pytest.mark.parametrize("variant", OWN_BIAS)
def test_reset_keeps_the_idle_bias_only_when_asked(variant):
    rig = FusionRig(VirtualHead(seed=27, worn=False), make_cfg(variant))
    rig.f.learn_idle_bias = True
    rig.run(10)
    learned = rig.f.idle_bias()
    rig.f.reset(keep_idle_bias=True)
    rig.run(0.1)
    assert np.allclose(rig.f.idle_bias(), learned, atol=0.05)
    rig.f.reset()
    rig.run(0.1)
    assert rig.f.idle_bias() is None


# --- VQF specifics --------------------------------------------------------------


@needs_vqf
def test_vqf_does_not_trust_a_bias_learned_on_a_swaying_head():
    # VQF counts gentle sway as rest and grows confident within a second;
    # only rest seen off the head (engine idle) counts at calibration.
    rig = FusionRig(VirtualHead(seed=28), make_cfg("vqf"))
    rig.calibrate()
    assert rig.f.bias_sigma_dps is not None and rig.f.bias_sigma_dps < rig.cfg.imu.vqf_bias_max_sigma_dps
    assert rig.f.bias_source == "worn"


@needs_vqf
def test_vqf_keeps_its_bias_across_a_gap_and_a_same_headset_reset():
    head = VirtualHead(seed=29, worn=False)
    rig = FusionRig(head, make_cfg("vqf"))
    rig.f.learn_idle_bias = True
    rig.run(20)
    learned = rig.f.bias.copy()
    assert np.linalg.norm(learned - head.bias) < 0.15
    rig.t += 1.0  # dropout inside fusion (longer than MAX_DT_S)
    rig.run(0.1)
    assert np.allclose(rig.f.bias, learned, atol=0.05)
    rig.f.reset(keep_idle_bias=True)  # dropout seen by the engine, same headset
    rig.t += 3.0
    rig.run(0.1)
    assert np.allclose(rig.f.bias, learned, atol=0.05)
    rig.f.reset()  # another headset
    rig.run(0.1)
    assert np.linalg.norm(rig.f.bias) < 0.05


@needs_vqf
def test_vqf_tilt_is_at_least_as_good_during_fast_motion():
    """Table, then worn; fast turns, nods and shakes with the sensor 15 cm from
    the neck pivot, so the accelerometer sees the head's linear acceleration."""

    def p95_tilt_error(variant: str) -> float:
        head = VirtualHead(seed=30, worn=False, lever_arm_m=(0.10, 0.0, 0.12))
        rig = FusionRig(head, make_cfg(variant))
        _table_then_worn(rig)
        rig.calibrate()
        errors: list[float] = []
        for i in range(10):
            head.look(yaw=[-40, 40][i % 2], pitch=[12, -12][(i // 2) % 2])
            rig.run(1.5, each=lambda: errors.append(rig.tilt_error_deg()))
            if i % 3 == 2:
                head.nod(rig.t) if i % 2 else head.shake(rig.t)
                rig.run(1.6, each=lambda: errors.append(rig.tilt_error_deg()))
        return float(np.percentile(errors, 95))

    complementary, vqf = p95_tilt_error("complementary"), p95_tilt_error("vqf")
    assert vqf <= complementary, (vqf, complementary)


def test_gyro_clips_at_full_scale_and_bursts_are_logged_once(caplog):
    """A shake faster than the Muse gyro's 245 deg/s reads as exactly 245."""
    rig = FusionRig(VirtualHead(seed=31))
    rig.calibrate()
    peak = []
    rig.head.shake(rig.t)  # ordinary shake, ~220 deg/s
    rig.run(2.0, each=lambda: peak.append(np.max(np.abs(rig.f.last_gyro))))
    assert max(peak) < 245 and rig.f.clipped_samples == 0
    with caplog.at_level("INFO", logger="neurocinema.fusion"):
        rig.head.shake(rig.t, amplitude_deg=28)  # ~440 deg/s
        rig.run(2.0, each=lambda: peak.append(np.max(np.abs(rig.f.last_gyro))))
    assert max(peak) == pytest.approx(245.0)
    assert rig.f.clipped_samples > 10
    assert len([r for r in caplog.records if "full scale" in r.message]) == 1


def test_a_large_gyro_bias_does_not_tilt_the_pitch(variant):
    """Headset C has ~6 deg/s of bias on the pitch axis (more than VQF's old
    5 deg/s limit, which left pitch ~3 deg off while looking level)."""
    head = VirtualHead(seed=40, gyro_bias_dps=(-1.7, 6.0, 0.45))
    rig = FusionRig(head, make_cfg(variant))
    rig.calibrate()
    errors: list[float] = []
    rig.run(20, each=lambda: errors.append(abs(rig.f.pitch_deg - head.pose(rig.t)[1])))
    assert np.median(errors[-15 * 52:]) < 1.0


@needs_vqf
def test_vqf_warns_when_the_bias_is_above_its_limit(caplog):
    cfg = make_cfg("vqf")
    cfg.imu.vqf_bias_clip_dps = 5.0
    rig = FusionRig(VirtualHead(seed=40, gyro_bias_dps=(-1.7, 6.0, 0.45)), cfg)
    with caplog.at_level("WARNING", logger="neurocinema.fusion_vqf"):
        rig.calibrate()
    assert any("vqf_bias_clip_dps" in r.message for r in caplog.records)

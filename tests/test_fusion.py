import numpy as np
import pytest

from neurocinema.config import Config
from neurocinema.fusion import CalibrationError, Calibrator, Fusion
from neurocinema.samples import ImuSample
from neurocinema.sim.head import RATE_HZ, VirtualHead


class FusionRig:
    def __init__(self, head: VirtualHead, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.f = Fusion(self.cfg.imu)
        self.cal = Calibrator(self.cfg.calibration.still_s)
        self.head = head
        self.t = 0.0

    def run(self, seconds: float) -> None:
        for _ in range(int(seconds * RATE_HZ)):
            acc, gyro = self.head.step(self.t)
            for s in (ImuSample(self.t, "acc", *acc), ImuSample(self.t, "gyro", *gyro)):
                self.f.update(s)
                if not self.f.calibrated:
                    self.cal.add(s, self.f.stillness.is_still())
            self.t += 1.0 / RATE_HZ

    def calibrate(self) -> None:
        while not self.cal.ready(self.t):
            self.run(0.1)
        self.f.calibrate(*self.cal.result())


@pytest.mark.parametrize("mount", [(0, 0, 0), (3, 12, -4), (-8, 25, 10)])
def test_turns_and_tilts_are_measured_through_a_crooked_band(mount):
    rig = FusionRig(VirtualHead(seed=2, mount=mount))
    rig.calibrate()
    rig.head.look(yaw=40, pitch=15)
    rig.run(3)
    assert rig.f.yaw_deg == pytest.approx(40, abs=3)
    assert rig.f.pitch_deg == pytest.approx(15, abs=2)
    rig.head.look(yaw=-40, pitch=-15)
    rig.run(3)
    assert rig.f.yaw_deg == pytest.approx(-40, abs=3)
    assert rig.f.pitch_deg == pytest.approx(-15, abs=2)


def test_yaw_holds_while_still_for_five_minutes():
    rig = FusionRig(VirtualHead(seed=3))
    rig.calibrate()
    rig.run(300)
    assert abs(rig.f.yaw_deg) < 2


def test_yaw_drift_while_moving_stays_small_over_a_minute():
    rig = FusionRig(VirtualHead(seed=4))
    rig.calibrate()
    for i in range(10):
        rig.head.look(yaw=[-40, 0, 40][i % 3], pitch=[10, -10][i % 2])
        rig.run(6)
    rig.head.look(yaw=0, pitch=0)
    rig.run(3)
    assert abs(rig.f.yaw_deg) < 5


def test_bias_is_learned_at_calibration():
    head = VirtualHead(seed=5, sway=False)
    rig = FusionRig(head)
    rig.calibrate()
    assert np.allclose(rig.f.bias, head.bias, atol=0.15)


def test_calibration_rejects_forward_axis_pointing_up():
    cfg = Config()
    cfg.imu.forward_axis = "z"
    f = Fusion(cfg.imu)
    with pytest.raises(CalibrationError):
        f.calibrate(np.zeros(3), np.array([0.0, 0.0, 1.0]))


def test_calibration_rejects_implausible_bias():
    f = Fusion(Config().imu)
    with pytest.raises(CalibrationError):
        f.calibrate(np.array([40.0, 0, 0]), np.array([0.0, 0.0, 1.0]))


def test_out_of_order_and_gapped_samples_do_not_corrupt_yaw():
    f = Fusion(Config().imu)
    f.calibrate(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    f.update(ImuSample(1.0, "gyro", 0, 0, 0))
    f.update(ImuSample(0.5, "gyro", 0, 0, -500))  # older: dropped
    f.update(ImuSample(3.0, "gyro", 0, 0, -500))  # after a 2 s gap: not integrated
    assert f.yaw_deg == 0.0

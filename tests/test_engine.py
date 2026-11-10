import importlib.util

import numpy as np
import pytest

from neurocinema.config import Config
from neurocinema.engine import State
from neurocinema.sim.head import VirtualHead

from helpers import Rig


def test_full_flow_select_left_top():
    rig = Rig()
    rig.start_session()
    assert rig.out.events("calibrated")

    rig.head.look(yaw=-40, pitch=12)
    rig.run(1.5)
    assert rig.out.last("/target") == ("left", "top")
    assert rig.out.last("/dwell") == (1.0,)

    rig.head.nod(rig.t)
    rig.run(1.0)
    assert rig.state == State.SELECTED
    assert rig.out.last("/selected") == ("left", "top")

    rig.run(2.5)
    assert rig.state == State.ACTIVE


def test_nod_selects_bottom_of_right_screen_without_flipping_row():
    rig = Rig(seed=1)
    rig.start_session()
    rig.head.look(yaw=40, pitch=-8)
    rig.run(1.5)
    assert rig.out.last("/target") == ("right", "bottom")
    rig.head.nod(rig.t)
    rig.run(1.0)
    assert rig.out.last("/selected") == ("right", "bottom")


def test_shake_resets_yaw_drift():
    rig = Rig(seed=2)
    rig.start_session()
    rig.engine.fusion.yaw_deg += 30  # simulate a large accumulated drift
    rig.run(1.0)
    assert rig.out.last("/target")[0] == "right"  # wrong: visitor faces middle
    rig.head.shake(rig.t)
    rig.run(3.0)
    assert rig.out.events("shake")
    assert rig.out.events("reset")
    assert rig.state == State.ACTIVE
    assert abs(rig.engine.fusion.yaw_deg) < 3
    rig.head.look(pitch=8)
    rig.run(1.0)
    assert rig.out.last("/target") == ("middle", "top")


def test_a_shake_faster_than_the_gyro_range_still_resets_yaw():
    # The gyro clips at 245 deg/s, so a vigorous shake leaves ~10 deg of yaw
    # error behind; the reset that follows the shake erases it.
    rig = Rig(seed=7)
    rig.start_session()
    rig.head.shake(rig.t, amplitude_deg=28)
    rig.run(3.0)
    assert rig.engine.fusion.clipped_samples > 10
    assert rig.out.events("reset")
    assert abs(rig.engine.fusion.yaw_deg) < 3
    rig.head.look(pitch=8)
    rig.run(1.0)
    assert rig.out.last("/target") == ("middle", "top")


def test_selection_reanchors_yaw_towards_screen_centre():
    rig = Rig(seed=3)
    rig.start_session()
    rig.head.look(yaw=-40, pitch=10)
    rig.run(1.5)
    rig.engine.fusion.yaw_deg += 8  # drift
    rig.run(0.5)
    before = rig.engine.fusion.yaw_deg
    rig.head.nod(rig.t)
    rig.run(1.0)
    after = rig.engine.fusion.yaw_deg
    assert abs(after + 40) < abs(before + 40)


def test_nod_too_soon_after_arriving_is_ignored():
    cfg = Config()
    cfg.interaction.min_dwell_s = 1.5
    rig = Rig(cfg, seed=4)
    rig.start_session()
    rig.head.look(yaw=40, pitch=10)
    rig.run(0.8)
    rig.head.nod(rig.t)
    rig.run(1.0)
    assert rig.state == State.ACTIVE
    assert ("nod_ignored", "dwell") in rig.out.events("nod_ignored")


def test_nods_do_not_select_while_watching():
    rig = Rig(seed=5)
    rig.start_session()
    rig.engine.command("phase", ("watching",), rig.t)
    rig.head.look(yaw=0, pitch=10)
    rig.run(1.5)
    rig.head.nod(rig.t)
    rig.run(1.0)
    assert rig.state == State.ACTIVE
    assert ("nod_ignored", "watching") in rig.out.events("nod_ignored")


def test_looking_around_without_gestures_never_selects():
    rig = Rig(seed=6)
    rig.start_session()
    for yaw, pitch in [(-40, 10), (0, -10), (40, 10), (40, -10), (0, 10), (-40, -10)] * 3:
        rig.head.look(yaw=yaw, pitch=pitch)
        rig.run(1.2)
    assert not rig.out.events("selected")
    assert not rig.out.events("shake")


def test_dropout_mid_session_recalibrates_on_return():
    rig = Rig(seed=7)
    rig.start_session()
    rig.drop()
    rig.run(1.0)
    assert rig.state == State.DISCONNECTED
    rig.streaming = True
    rig.run(0.1)
    assert rig.state == State.CALIBRATING
    rig.run(3.5)
    assert rig.state == State.ACTIVE


def test_headset_taken_off_goes_idle_and_wakes_when_put_back_on():
    cfg = Config()
    rig = Rig(cfg, seed=8)
    rig.start_session()
    rig.head.set_worn(False, rig.t)
    # Gravity takes a couple of seconds to settle on the new orientation.
    rig.run(cfg.interaction.idle_tilt_s + 3.0)
    assert rig.state == State.IDLE
    rig.head.set_worn(True, rig.t)
    rig.run(1.0)
    assert rig.state == State.CALIBRATING


def test_snapshot_repeats_state_for_late_listeners():
    rig = Rig(seed=9)
    rig.start_session()
    rig.out.messages.clear()
    rig.run(1.2)
    assert rig.out.last("/state") == ("active",)
    assert rig.out.last("/phase") == ("choosing",)


def test_reset_command_zeroes_yaw():
    rig = Rig(seed=10)
    rig.start_session()
    rig.engine.fusion.yaw_deg = 25
    rig.engine.command("reset", (), rig.t)
    assert rig.engine.fusion.yaw_deg == 0
    assert rig.out.events("reset")


def test_target_follows_a_turn_promptly():
    rig = Rig(seed=11)
    rig.start_session()
    rig.head.look(yaw=40, pitch=10)
    rig.run(0.5)
    assert rig.out.last("/target") == ("right", "top")


def test_nod_right_after_looking_up_never_selects_the_previous_row():
    cfg = Config()
    cfg.interaction.min_dwell_s = 0.3
    rig = Rig(cfg, seed=12)
    rig.start_session()
    rig.head.look(yaw=0, pitch=-10)
    rig.run(1.5)
    assert rig.out.last("/target") == ("middle", "bottom")
    rig.head.look(pitch=10)
    rig.run(0.15)  # still moving up
    rig.head.nod(rig.t)
    rig.run(1.0)
    assert ("middle", "bottom") not in [s for s in rig.out.events("selected")]


def _on_the_stand_then_worn(rig: Rig, stand_s: float = 15.0) -> None:
    rig.head.set_worn(False, rig.t)
    rig.run(stand_s)
    assert rig.state == State.IDLE
    rig.head.set_worn(True, rig.t)
    rig.run(1.6)
    assert rig.state == State.CALIBRATING
    rig.run(3.5)
    assert rig.state == State.ACTIVE


def complementary() -> Config:
    """The complementary filter, whose own idle-bias logic some tests check."""
    cfg = Config()
    cfg.imu.backend = "complementary"
    return cfg


def test_bias_learned_on_the_stand_is_used_at_calibration():
    rig = Rig(complementary(), head=VirtualHead(seed=13, worn=False))
    _on_the_stand_then_worn(rig)
    assert rig.engine.fusion.bias_source == "idle"
    assert np.linalg.norm(rig.engine.fusion.bias - rig.head.bias) < 0.15


def test_bias_is_learned_only_while_idle():
    rig = Rig(seed=14)
    rig.run(1.0)
    assert rig.state == State.IDLE
    assert rig.engine.fusion.learn_idle_bias
    rig.engine.command("start", (), rig.t)
    assert rig.state == State.CALIBRATING
    assert not rig.engine.fusion.learn_idle_bias


def test_stand_bias_survives_a_dropout_of_the_same_headset_only():
    rig = Rig(complementary(), head=VirtualHead(seed=15, worn=False))
    rig.headset = "A"
    rig.run(10.0)
    assert rig.engine.fusion.idle_bias() is not None
    rig.drop()
    rig.run(1.0)
    rig.streaming = True
    rig.run(0.5)
    assert rig.state == State.IDLE
    assert rig.engine.fusion.idle_bias() is not None  # same headset: kept
    rig.drop()
    rig.run(1.0)
    rig.headset = "B"
    rig.streaming = True
    rig.run(0.5)
    assert rig.engine.fusion.idle_bias() is None  # another headset: dropped


def test_stand_bias_is_dropped_when_the_headset_is_unknown():
    rig = Rig(head=VirtualHead(seed=16, worn=False))
    rig.run(10.0)
    rig.drop()
    rig.run(1.0)
    rig.streaming = True
    rig.run(0.5)
    assert rig.engine.fusion.idle_bias() is None


@pytest.mark.skipif(importlib.util.find_spec("vqf") is None, reason="pip install -e '.[vqf]'")
def test_full_flow_with_the_vqf_backend():
    cfg = Config()
    cfg.imu.backend = "vqf"
    rig = Rig(cfg, head=VirtualHead(seed=17, worn=False))
    _on_the_stand_then_worn(rig)
    assert rig.engine.fusion.bias_source == "vqf"
    rig.head.look(yaw=-40, pitch=12)
    rig.run(1.5)
    assert rig.out.last("/target") == ("left", "top")
    rig.head.nod(rig.t)
    rig.run(1.0)
    assert rig.out.last("/selected") == ("left", "top")
    rig.run(2.5)
    rig.head.look(yaw=0, pitch=0)
    rig.run(1.5)
    rig.engine.fusion.yaw_deg += 30
    rig.head.shake(rig.t)
    rig.run(3.0)
    assert rig.out.events("reset")
    assert abs(rig.engine.fusion.yaw_deg) < 3


def test_calibration_is_reported_for_the_viewer():
    rig = Rig(complementary(), seed=8)
    rig.run(0.5)
    imu = rig.out.last("/calib/imu")
    assert imu[:2] == ("complementary", "none") and imu[3] == -1.0  # not calibrated yet
    rig.start_session()
    backend, source, bias, since, corrected, clipped = rig.out.last("/calib/imu")
    assert source in ("idle", "worn") and 0.5 < bias < 3 and 0 <= since < 1 and corrected == 0 and clipped == 0
    rig.head.look(yaw=40, pitch=10)
    rig.run(1.5)
    rig.engine.fusion.yaw_deg += 6  # drift
    rig.head.nod(rig.t)
    rig.run(6.0)  # selection re-anchors; /calib/imu repeats every 5 s
    assert rig.out.last("/calib/imu")[4] > 1
    room = rig.out.last("/calib/room")
    assert room[0] == 1 and len(room) == 9  # Config() defaults: from_room, but no room file loaded


def test_setup_command_switches_columns_live():
    rig = Rig(seed=12)
    rig.start_session()
    rig.head.look(yaw=17, pitch=8)
    rig.run(1.0)
    assert rig.out.last("/target") == ("middle", "top")  # built-in: middle up to 20
    yaw, calibrated = rig.engine.fusion.yaw_deg, rig.engine.fusion.calibrated
    rig.out.messages.clear()
    rig.engine.command("setup", ("narrow-centre",), rig.t)
    sent = [(a, args) for _, a, args in rig.out.messages]
    assert ("/nc/target", ("none", "none")) in sent
    assert ("/nc/setup", ("narrow-centre", -60, -12, -18, 18, 12, 60, -30, 0, 30, 0, 40)) in sent
    assert rig.out.events("setup") == [("setup", "narrow-centre")]
    assert rig.engine.fusion.yaw_deg == yaw and rig.engine.fusion.calibrated == calibrated
    assert rig.state == State.ACTIVE
    rig.run(0.5)
    assert rig.out.last("/target") == ("right", "top")  # narrow-centre: middle ends at 18


def test_unknown_setup_changes_nothing():
    rig = Rig(seed=13)
    rig.start_session()
    before = rig.engine.targets.setup
    rig.out.messages.clear()
    rig.engine.command("setup", ("nope",), rig.t)
    assert rig.engine.setup_name == "room" and rig.engine.targets.setup is before
    assert not rig.out.messages


def test_setup_without_name_rereads_the_file(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text("[x]\nleft = [-60, -12]\nmiddle = [-18, 18]\nright = [12, 60]\n")
    cfg = Config()
    cfg.targets.setups_file = str(f)
    cfg.targets.setup = "x"
    rig = Rig(cfg)
    assert rig.engine.targets.setup.ranges["middle"] == (-18, 18)
    f.write_text("[x]\nleft = [-60, -8]\nmiddle = [-10, 10]\nright = [8, 60]\n"
                 "[y]\nleft = [-60, -20]\nmiddle = [-20, 20]\nright = [20, 60]\n")
    rig.engine.command("setup", (), rig.t)
    assert rig.engine.setup_name == "x" and rig.engine.targets.setup.ranges["middle"] == (-10, 10)
    assert rig.out.last("/setup")[:5] == ("x", -60, -8, -10, 10)
    assert set(rig.engine.setups) == {"room", "x", "y"}
    f.write_text("[x]\nleft = 3\n")  # broken: keep what we have
    rig.engine.command("setup", (), rig.t)
    assert rig.engine.targets.setup.ranges["middle"] == (-10, 10)


def test_snapshot_and_calibration_carry_the_setup():
    rig = Rig(seed=14)
    rig.run(1.2)
    assert rig.out.last("/setup")[0] == "room"
    assert len(rig.out.last("/setup")) == 12

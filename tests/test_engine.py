import numpy as np

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


def test_bias_learned_on_the_stand_is_used_at_calibration():
    rig = Rig(head=VirtualHead(seed=13, worn=False))
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
    rig = Rig(head=VirtualHead(seed=15, worn=False))
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

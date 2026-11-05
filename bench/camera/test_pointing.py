"""Checks for the camera bench maths: python -m pytest bench/camera"""

import sys
from pathlib import Path
from types import SimpleNamespace as L

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "viewer"))

from pointing import Selector, arm_aim, cam_to_room, head_aim, yaw_pitch  # noqa: E402


def body(right_wrist=(-0.18, 0.0, 0.0), right_index=None, left_wrist=(0.18, 0.05, 0.0)):
    """A person facing a level camera at bearing 0 (camera frame: x image-right, y down, z away).

    The visitor's right side is image-left (-x). Right arm hangs unless the wrist is given.
    """
    p = {i: (0.0, 0.0, 0.0) for i in range(33)}
    p[0] = (0.0, -0.62, -0.08)               # nose, nearer the camera
    p[2], p[5] = (0.03, -0.66, -0.06), (-0.03, -0.66, -0.06)   # eyes (left = image right)
    p[7], p[8] = (0.07, -0.64, 0.0), (-0.07, -0.64, 0.0)       # ears
    p[11], p[12] = (0.18, -0.48, 0.0), (-0.18, -0.48, 0.0)     # shoulders
    sx, sy, sz = p[12]
    wx, wy, wz = right_wrist
    p[14] = ((sx + wx) / 2, (sy + wy) / 2, (sz + wz) / 2)      # elbow on the line: straight arm
    p[16] = right_wrist
    p[20] = right_index or (wx + (wx - sx) * 0.15, wy + (wy - sy) * 0.15, wz + (wz - sz) * 0.15)
    p[13] = (0.18, -0.2, 0.0)                                  # left arm hanging
    p[15], p[19] = left_wrist, (0.18, 0.1, 0.0)
    p[23], p[24] = (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)
    return [L(x=x, y=y, z=z, visibility=0.99) for x, y, z in (p[i] for i in range(33))]


def test_toward_camera_has_the_camera_bearing():
    for b in (0, 29.6, -45, 90):
        yaw, pitch = yaw_pitch(cam_to_room((0, 0, -1), b))
        assert yaw == pytest.approx(b, abs=1e-6) and pitch == pytest.approx(0, abs=1e-6)


def test_tilted_camera_is_above_the_visitor():
    yaw, pitch = yaw_pitch(cam_to_room((0, 0, -1), 0, tilt_deg=20))
    assert pitch == pytest.approx(20, abs=1e-6)


def test_visitor_right_is_image_left():
    yaw, _ = yaw_pitch(cam_to_room((-1, 0, 0)))
    assert yaw == pytest.approx(90)


def test_hanging_arms_are_not_pointing():
    assert arm_aim(body()) is None


def test_point_right_and_forward():
    a = arm_aim(body(right_wrist=(-0.78, -0.48, 0.0)))
    assert a.source == "right arm" and a.yaw == pytest.approx(90, abs=8)
    a = arm_aim(body(right_wrist=(-0.18, -0.5, -0.6)))
    assert abs(a.yaw) < 20


def test_head_faces_the_camera():
    h = head_aim(body())
    assert h.yaw == pytest.approx(0, abs=1)
    assert head_aim(body(), bearing_deg=29.6).yaw == pytest.approx(29.6, abs=1)


def test_selector_hold_then_dwell():
    from pointing import Aim
    sel = Selector(lambda y, p: "right" if y > 30 else "middle", smooth_s=0, hold_s=0.2, dwell_s=1.0)
    ev = []
    for i in range(80):
        t = i * 0.02
        ev += [(round(t, 2), e) for e in sel.update(t, Aim(60, 0, "right arm", 1))]
    assert ev[0][1] == ("highlight", "right") and ev[0][0] == pytest.approx(0.2, abs=0.03)
    assert ev[1][1] == ("select", "right") and ev[1][0] == pytest.approx(1.2, abs=0.03)
    assert len(ev) == 2


def test_selector_survives_short_dropouts():
    from pointing import Aim
    sel = Selector(lambda y, p: "middle", smooth_s=0, hold_s=0.0, dwell_s=10, lost_s=0.5)
    sel.update(0.0, Aim(0, 0, "head", 1))
    sel.update(0.3, None)
    assert sel.highlight == "middle"
    sel.update(0.9, None)
    assert sel.highlight is None

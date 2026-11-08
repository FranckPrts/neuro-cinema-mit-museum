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


# --- top/bottom halves (panels.py) ---

import room_geometry as rg  # noqa: E402
from panels import PanelMapper  # noqa: E402


def mapper(**kw):
    return PanelMapper(rg.load_room(None), **kw)


def test_front_screen_halves():
    m = mapper()
    mid = m.midline_pitch("front", rg.hit(m.cfg, 0, 0)["point"])
    assert -3 < mid < 1          # divider of a 10 ft screen is near eye height
    assert m(0, 10) == "middle/top"
    assert mapper()(0, -10) == "middle/bottom"
    assert mapper()(0, 30) is None   # above the screen: ceiling


def test_side_screen_halves():
    assert mapper()(60, 15) == "right/top"
    assert mapper()(-60, -20) == "left/bottom"


def test_dead_band_holds_the_half():
    m = mapper(band_deg=4.0)
    mid = m.midline_pitch("front", rg.hit(m.cfg, 0, 0)["point"])
    assert m(0, mid + 5) == "middle/top"
    assert m(0, mid - 1.5) == "middle/top"      # inside the band: stays top
    assert m(0, mid - 2.5) == "middle/bottom"   # past the band
    assert m(0, mid + 1.5) == "middle/bottom"   # inside the band: stays bottom
    assert m(0, mid + 2.5) == "middle/top"


def test_halves_off():
    assert mapper(halves=False)(0, -10) == "middle"


def test_arm_pointing_up_and_down():
    up = arm_aim(body(right_wrist=(-0.18, -0.7, -0.55)))      # forward, a little up
    down = arm_aim(body(right_wrist=(-0.18, -0.5, -0.6)))     # forward, level with the shoulder
    assert up.pitch > 3 and down.pitch < -8
    assert mapper()(up.yaw, up.pitch) == "middle/top"
    assert mapper()(down.yaw, down.pitch) == "middle/bottom"


def test_selector_with_halves():
    from pointing import Aim
    sel = Selector(mapper(), smooth_s=0, hold_s=0.1, dwell_s=0.5)
    ev = []
    for i in range(80):
        t = i * 0.02
        ev += sel.update(t, Aim(0, 10 if t < 0.5 else -10, "right arm", 1))
    assert ("highlight", "middle/top") in ev and ("highlight", "middle/bottom") in ev
    assert ("select", "middle/bottom") in ev


# --- camera position (camera_pose.py) ---

from camera_pose import CameraConfigError, load_camera, pose_from_points  # noqa: E402


def test_camera_forward_matches_cam_to_room():
    """The camera's own forward axis, turned into the room, points at the aim point."""
    for pos in [(0, 1.3, -0.6), (2.45, 2.5, -4.36), (-2.45, 2.5, -4.36), (2.45, 1.0, 0.5)]:
        aim = (0.0, 1.3, 0.0)
        pose = pose_from_points(pos, aim)
        f = cam_to_room((0, 0, 1), pose.bearing_deg, pose.tilt_deg)
        want = [a - p for a, p in zip(aim, pos)]
        n = sum(c * c for c in want) ** 0.5
        assert f == pytest.approx([c / n for c in want], abs=1e-9)


def test_front_right_corner_numbers():
    pose = pose_from_points((2.45, 2.5, -4.36), (0, 1.3, 0))
    assert pose.bearing_deg == pytest.approx(29.3, abs=0.2)
    assert pose.tilt_deg == pytest.approx(13.5, abs=0.5)


def test_default_camera_file_is_a_level_desk_camera():
    pose = load_camera(HERE / "camera.toml")
    assert pose.bearing_deg == pytest.approx(0) and pose.tilt_deg == pytest.approx(0)


def test_camera_file_rejects_typos(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[camera]\nx = 1\n")
    with pytest.raises(CameraConfigError):
        load_camera(p)

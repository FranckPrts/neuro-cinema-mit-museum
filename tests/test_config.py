from dataclasses import asdict
from pathlib import Path

import pytest

from neurocinema.config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_installation_toml_mirrors_defaults():
    # Without the room step, which fills the screen yaws from config/room.toml.
    assert asdict(load_config([ROOT / "config" / "installation.toml"], apply_room=False)) == asdict(Config())


def test_later_files_override(tmp_path):
    a = tmp_path / "a.toml"
    a.write_text("[targets]\nfrom_room = false\nleft_yaw_deg = -50\n")
    b = tmp_path / "b.toml"
    b.write_text("[targets]\nleft_yaw_deg = -45.5\n[gestures.nod]\npeak_dps = 40\n")
    cfg = load_config([a, b])
    assert cfg.targets.left_yaw_deg == -45.5
    assert cfg.gestures.nod.peak_dps == 40.0
    assert cfg.gestures.shake.peak_dps == 70.0


@pytest.mark.parametrize("text", [
    "[targets]\nleft_yaw = -40\n",                 # typo
    "[imu]\nforward_axis = 'q'\n",
    "[targets]\nfrom_room = false\nleft_yaw_deg = 10\n",  # left must be left of middle
    "[targets]\nleft_yaw_deg = -45\n",            # yaws come from the room
    "[targets]\nroom_file = 'nowhere.toml'\n",
    "[osc]\ndestinations = ['localhost']\n",
    "[interaction]\nmin_dwell_s = 'long'\n",
    "[gestures.nod]\nfirst_sign = 2\n",
    "[imu]\nbackend = 'kalman'\n",
    "[imu]\nvqf_bias_from = 'magic'\n",
])
def test_bad_config_is_rejected(tmp_path, text):
    p = tmp_path / "bad.toml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config([p])


def test_screen_yaws_and_edges_come_from_the_room():
    t = load_config([ROOT / "config" / "installation.toml"]).targets
    # config/room.toml: 16.5 ft square room, visitor 2 ft from the back wall,
    # screens on the front and both side walls.
    assert (t.left_yaw_deg, t.middle_yaw_deg, t.right_yaw_deg) == pytest.approx((-52.9, 0.0, 52.9), abs=0.1)
    assert t.bounds() == pytest.approx((-102.8, -29.6, 29.6, 102.8), abs=0.1)


def test_room_change_moves_the_screens(tmp_path):
    room = tmp_path / "room.toml"
    room.write_text("[room]\nwidth_m = 8.0\n")  # a wider room: side walls further out
    cfg = tmp_path / "c.toml"
    cfg.write_text(f"[targets]\nroom_file = '{room}'\n")
    t = load_config([cfg]).targets
    assert t.right_yaw_deg > 60 and t.bounds()[2] > 40


def test_targets_follow_the_room_screen_edges():
    from neurocinema.targets import TargetTracker

    tr = TargetTracker(load_config([ROOT / "config" / "installation.toml"]).targets)
    col = lambda yaw: (tr.reset(), tr.update(0.0, yaw, 5.0), tr.current and tr.current.column)[2]
    assert col(25) == "middle" and col(34) == "right" and col(-34) == "left"
    assert col(98) == "right" and col(108) is None  # side screens end at the back corners (102.8)

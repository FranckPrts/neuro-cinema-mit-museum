import random
from pathlib import Path

import pytest

from neurocinema.config import Config, TargetsConfig, load_config
from neurocinema.setups import SelectionSetup, SetupError, builtin_setup, load_setups, startup_name
from neurocinema.targets import TargetTracker

ROOT = Path(__file__).resolve().parents[1]


def old_classify(cfg: TargetsConfig, current: str | None, yaw: float) -> str | None:
    """TargetTracker._classify_column before selection setups (bounds + dead band)."""
    lo, lm, mr, hi = cfg.bounds()
    h2 = cfg.yaw_hysteresis_deg / 2
    out = h2 if current else -h2
    if yaw < lo - out or yaw > hi + out:
        return None
    plain = "left" if yaw < lm else ("middle" if yaw < mr else "right")
    if current is None or plain == current:
        return plain
    if current == "left":
        return plain if yaw >= lm + h2 else current
    if current == "right":
        return plain if yaw < mr - h2 else current
    if yaw < lm - h2 or yaw >= mr + h2:
        return plain
    return current


@pytest.mark.parametrize("targets", [
    Config().targets,                                                       # manual yaws
    load_config([ROOT / "config" / "installation.toml"]).targets,           # room
])
def test_builtin_setup_reproduces_the_old_columns(targets):
    tr = TargetTracker(targets)
    rng = random.Random(0)
    yaw, current = 0.0, None
    for _ in range(20000):
        yaw = max(-130.0, min(130.0, yaw + rng.uniform(-9, 9)))
        if rng.random() < 0.02:
            yaw = rng.choice([-1, 1]) * rng.uniform(0, 120)  # jumps, also from no target
        if rng.random() < 0.05:
            yaw = round(yaw)  # land exactly on boundaries sometimes
        expected = old_classify(targets, current, yaw)
        got = tr._classify_column(yaw)
        assert got == expected, (current, yaw)
        tr._column = current = got


def narrow() -> SelectionSetup:
    return load_setups(Config().targets)["narrow-centre"]


def column(tr: TargetTracker, *yaws: float) -> str | None:
    for i, yaw in enumerate(yaws):
        tr.update(float(i), yaw, 5.0)
    return tr.current and tr.current.column


def test_overlap_keeps_the_column_it_came_from():
    # narrow-centre: left [-60, -12], middle [-18, 18]; shared zone -18..-12, midpoint -15.
    assert column(TargetTracker(Config().targets, narrow()), 0, -13) == "middle"
    assert column(TargetTracker(Config().targets, narrow()), -30, -17) == "left"
    assert column(TargetTracker(Config().targets, narrow()), -30, -17, -11) == "middle"
    assert column(TargetTracker(Config().targets, narrow()), 0, -13, -19) == "left"


def test_overlap_entered_from_no_target_follows_the_midpoint():
    assert column(TargetTracker(Config().targets, narrow()), -16) == "left"
    assert column(TargetTracker(Config().targets, narrow()), -14) == "middle"
    assert column(TargetTracker(Config().targets, narrow()), 120, 14) == "middle"  # none, then the zone
    assert column(TargetTracker(Config().targets, narrow()), 120, 16) == "right"


def test_gap_and_outer_ends_give_no_target():
    t = Config().targets
    gap = SelectionSetup("gap", {"left": (-60, -20), "middle": (-10, 10), "right": (20, 60)},
                         {"left": -40, "middle": 0, "right": 40}, 0.0, 40.0)
    tr = TargetTracker(t, gap)
    assert column(tr, 0, -15) is None
    assert column(tr, 15) is None
    assert column(tr, 40) == "right"
    assert column(tr, 62) == "right"  # stays up to 60 + 3...
    assert column(tr, 64) is None
    assert column(tr, 59) is None     # ...but enters only from 60 - 3
    assert column(tr, 56) == "right"


def test_setup_pitch_values_replace_the_targets_ones():
    s = SelectionSetup("low", {"left": (-60, -20), "middle": (-20, 20), "right": (20, 60)},
                       {"left": -40, "middle": 0, "right": 40}, split_pitch_deg=-10.0, max_pitch_deg=20.0)
    tr = TargetTracker(Config().targets, s)
    tr.update(0.0, 0, -5)
    assert tr.current.row == "top"
    tr.reset()
    tr.update(0.0, 0, 15)
    assert tr.current is None


def test_set_setup_switches_and_resets():
    tr = TargetTracker(Config().targets)
    assert column(tr, 0, 22) == "middle"  # manual defaults: middle kept up to 20 + 3
    tr.set_setup(narrow())
    assert tr.current is None
    assert column(tr, 22) == "right"      # narrow-centre: middle ends at 18
    assert tr.column_centre("right") == 30


def test_file_setup_centres_default_to_the_middle_of_each_range(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text("[wide]\nleft = [-90, -20]\nmiddle = [-25, 25]\nright = [20, 90]\nmax_pitch_deg = 30\n")
    s = load_setups(Config().targets, f)["wide"]
    assert s.centres == {"left": -55, "middle": 0, "right": 55}
    assert (s.split_pitch_deg, s.max_pitch_deg) == (0.0, 30.0)
    assert s.osc_args() == ("wide", -90, -20, -25, 25, 20, 90, -55, 0, 55, 0, 30)


def test_builtin_setup_is_always_there():
    t = Config().targets
    t.from_room = False
    setups = load_setups(t)
    assert list(setups)[0] == "manual" and setups["manual"] == builtin_setup(t)
    assert "narrow-centre" in setups
    assert startup_name(t, setups) == "manual"
    room = load_config([ROOT / "config" / "installation.toml"]).targets
    assert startup_name(room, load_setups(room)) == "narrow-centre"
    assert builtin_setup(room).name == "room"
    assert builtin_setup(room).ranges["middle"] == pytest.approx((-29.6 - 3, 29.6 + 3), abs=0.1)
    t.setup = "nope"
    with pytest.raises(SetupError):
        startup_name(t, setups)


def test_missing_file_leaves_only_the_builtin(tmp_path):
    assert list(load_setups(Config().targets, tmp_path / "none.toml")) == ["room"]


@pytest.mark.parametrize("text", [
    "[a]\nleft = [-60, -12]\nmiddle = [-18, 18]\nright = [12, 60]\nwidth = 3\n",        # unknown key
    "[a]\nleft = [-12, -60]\nmiddle = [-18, 18]\nright = [12, 60]\n",                   # lo >= hi
    "[a]\nleft = [-60, -12]\nmiddle = [-18, 18]\nright = [12, 12]\n",                   # lo == hi
    "[a]\nleft = [-18, 18]\nmiddle = [-60, -12]\nright = [12, 60]\n",                   # wrong order
    "[a]\nleft = [-60, 20]\nmiddle = [-18, 30]\nright = [12, 60]\n",                    # left overlaps right
    "[a]\nleft = [-60, -12]\nmiddle = [-18, 18]\n",                                     # missing column
    "[a]\nleft = [-60, 'x']\nmiddle = [-18, 18]\nright = [12, 60]\n",                   # not a number
    "[a]\nleft = [-60, -12]\nmiddle = [-18, 18]\nright = [12, 60]\ncentres = [0, 0, 30]\n",  # centre outside
    "[room]\nleft = [-60, -12]\nmiddle = [-18, 18]\nright = [12, 60]\n",                # built-in name
    "[a b]\nleft = [-60, -12]\nmiddle = [-18, 18]\nright = [12, 60]\n",                 # space in name
    "a = 3\n",                                                                          # not a table
])
def test_bad_setups_file_is_rejected(tmp_path, text):
    f = tmp_path / "s.toml"
    f.write_text(text)
    with pytest.raises(SetupError):
        load_setups(Config().targets, f)


def test_middle_without_yaw_zero_is_only_a_warning(tmp_path, caplog):
    f = tmp_path / "s.toml"
    f.write_text("[off]\nleft = [-60, 2]\nmiddle = [5, 30]\nright = [28, 60]\n")
    assert "off" in load_setups(Config().targets, f)
    assert "outside middle" in caplog.text


def test_shipped_setups_file_loads():
    t = load_config([ROOT / "config" / "installation.toml"]).targets
    s = load_setups(t)["narrow-centre"]
    assert s.ranges == {"left": (-60, -12), "middle": (-18, 18), "right": (12, 60)}

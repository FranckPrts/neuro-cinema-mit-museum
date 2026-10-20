from neurocinema.config import Config
from neurocinema.targets import Target, TargetTracker


def tracker():
    return TargetTracker(Config().targets)


def test_six_targets():
    tr = tracker()
    cases = {
        (-40, 10): Target("left", "top"),
        (-40, -10): Target("left", "bottom"),
        (0, 10): Target("middle", "top"),
        (0, -10): Target("middle", "bottom"),
        (40, 10): Target("right", "top"),
        (40, -10): Target("right", "bottom"),
    }
    for (yaw, pitch), expected in cases.items():
        tr.reset()
        tr.update(0.0, yaw, pitch)
        assert tr.current == expected


def test_yaw_hysteresis_around_boundary():
    tr = tracker()
    tr.update(0.0, 0, 5)
    assert tr.current.column == "middle"
    tr.update(0.1, 21, 5)  # just past the midpoint (20) but inside the dead band
    assert tr.current.column == "middle"
    tr.update(0.2, 24, 5)  # clearly past
    assert tr.current.column == "right"
    tr.update(0.3, 19, 5)  # back just past midpoint: stays right
    assert tr.current.column == "right"
    tr.update(0.4, 16, 5)
    assert tr.current.column == "middle"


def test_pitch_hysteresis_around_divider():
    tr = tracker()
    tr.update(0.0, 0, 5)
    assert tr.current.row == "top"
    tr.update(0.1, 0, -1.5)
    assert tr.current.row == "top"
    tr.update(0.2, 0, -3)
    assert tr.current.row == "bottom"


def test_out_of_range_is_no_target():
    tr = tracker()
    tr.update(0.0, 90, 0)
    assert tr.current is None
    tr.update(0.1, 0, 60)
    assert tr.current is None


def test_since_only_changes_on_change():
    tr = tracker()
    assert tr.update(1.0, 0, 5)
    assert not tr.update(2.0, 1, 6)
    assert tr.since == 1.0
    assert tr.dwell(3.0) == 2.0


def test_history_returns_target_at_a_past_time():
    tr = tracker()
    tr.update(1.0, -40, 10)
    tr.update(2.0, 0, 10)
    tr.update(3.0, 40, 10)
    assert tr.at(2.5) == (Target("middle", "top"), 2.0)
    assert tr.at(0.5) == (None, None)


def test_hold_row_keeps_top_bottom_but_tracks_column():
    tr = tracker()
    tr.update(0.0, 0, 10)
    tr.update(0.1, 40, -10, hold_row=True)
    assert tr.current == Target("right", "top")

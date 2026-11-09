"""The camera's OSC output has the engine's shapes, under /nc/cam."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from osc_out import CamOsc, split_target  # noqa: E402

# Argument types of the engine's messages (README, OSC reference).
ENGINE_SHAPES = {
    "state": (str,), "target": (str, str), "selected": (str, str),
    "dwell": (float,), "pose": (float, float, float), "heartbeat": (float,),
}
COLUMNS, ROWS = {"left", "middle", "right", "none"}, {"top", "bottom", "none"}


def run(frames):
    sent = []
    osc = CamOsc(lambda a, *args: sent.append((a, args)), t0=0.0, legacy=False)
    for f in frames:
        osc.frame(*f)
    osc.close()
    return sent


def test_split_target():
    assert split_target("middle/top") == ("middle", "top")
    assert split_target("left") == ("left", "none")
    assert split_target(None) == ("none", "none")


def test_messages_match_the_engine_shapes():
    sent = run([
        (0.0, None, [], None, 0.0, False),
        (0.5, (31.0, 5.0), [("highlight", "right/top")], "right/top", 0.2, False),
        (2.0, (31.0, 5.0), [("select", "right/top")], "right/top", 0.0, True),
        (3.1, None, [("highlight", None)], None, 0.0, False),
    ])
    for address, args in sent:
        assert address.startswith("/nc/cam/")
        name = address.removeprefix("/nc/cam/")
        if name in ENGINE_SHAPES:
            assert tuple(type(a) for a in args) == ENGINE_SHAPES[name], (address, args)
        if name in ("target", "selected"):
            assert args[0] in COLUMNS and args[1] in ROWS
    names = [a for a, _ in sent]
    assert ("/nc/cam/selected", ("right", "top")) in sent
    assert ("/nc/cam/event", ("selected", "right", "top")) in sent
    assert names[0] == "/nc/cam/event" and sent[-1] == ("/nc/cam/event", ("camera_stopped",))
    states = [args[0] for a, args in sent if a == "/nc/cam/state"]
    assert states[:3] == ["idle", "active", "selected"] and states[-1] == "idle"


def test_target_and_state_repeat_every_second():
    sent = run([(t / 10, (0.0, 0.0), [], "middle/top", 0.1, False) for t in range(35)])
    assert len([1 for a, _ in sent if a == "/nc/cam/target"]) == 4  # t = 0, 1, 2, 3 s


def test_calibration_is_sent_on_change_and_every_5_s():
    sent = []
    osc = CamOsc(lambda a, *args: sent.append((a, args)), t0=0.0, legacy=False)
    osc.calib = (0.0, 1.2, -0.6, 0.0, 1.2, -4.42, 0.0, 0.0, 1.6, 0.5, 0.0)
    for i in range(90):
        if i == 30:  # the operator recalibrates the arm pitch
            osc.calib = osc.calib[:-1] + (2.5,)
        osc.frame(i / 10, None, [], None, 0.0, False)
    calibs = [(round(i), args) for i, (a, args) in enumerate(sent) if a == "/nc/cam/calib"]
    assert len(calibs) == 3  # start, the change at 3 s, then 5 s later
    assert all(len(args) == 11 and all(type(v) is float for v in args) for _, args in calibs)
    assert calibs[1][1][-1] == 2.5

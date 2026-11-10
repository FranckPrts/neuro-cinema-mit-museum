from neurocinema.config import Config
from neurocinema.engine import Engine
from neurocinema.sim.head import VirtualHead
from neurocinema.sim.keyboard import SimKeyboard
from neurocinema.sources.sim import SimSource

from helpers import RecordingOutput


def test_keys_drive_the_virtual_head():
    source = SimSource(VirtualHead(seed=0))
    quit_called = []
    kb = SimKeyboard(source, Engine(Config(), RecordingOutput()), lambda: quit_called.append(True))
    head = source.head

    kb._handle("3", 0.0)
    assert head.target_yaw == 40.0
    kb._handle("left", 0.0)
    assert head.target_yaw == 30.0
    kb._handle("t", 0.0)
    assert head.target_pitch == 12.0
    kb._handle("0", 0.0)
    assert (head.target_yaw, head.target_pitch) == (0.0, 0.0)

    kb._handle("n", 1.0)
    kb._handle("h", 2.0)
    assert [g[0] for g in head._gestures] == ["nod", "shake"]

    kb._handle("o", 3.0)
    assert not head.worn
    kb._handle("c", 3.0)
    assert not source.streaming
    kb._handle("q", 3.0)
    assert quit_called
    kb.close()


def test_s_cycles_setups_and_column_keys_follow_their_centres():
    source = SimSource(VirtualHead(seed=0))
    engine = Engine(Config(), RecordingOutput())
    kb = SimKeyboard(source, engine, lambda: None)
    assert list(engine.setups) == ["room", "narrow-centre"]
    kb._handle("3", 0.0)
    assert source.head.target_yaw == 40.0
    kb._handle("s", 0.0)
    assert engine.setup_name == "narrow-centre"
    kb._handle("3", 0.0)
    assert source.head.target_yaw == 30.0
    kb._handle("1", 0.0)
    assert source.head.target_yaw == -30.0
    kb._handle("s", 0.0)
    assert engine.setup_name == "room"
    kb.close()

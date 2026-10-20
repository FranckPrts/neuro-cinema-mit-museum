from neurocinema.config import Config
from neurocinema.sim.head import VirtualHead
from neurocinema.sim.keyboard import SimKeyboard
from neurocinema.sources.sim import SimSource


def test_keys_drive_the_virtual_head():
    source = SimSource(VirtualHead(seed=0))
    quit_called = []
    kb = SimKeyboard(source, Config().targets, lambda: quit_called.append(True))
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
    assert [k for k, _ in head._gestures] == ["nod", "shake"]

    kb._handle("o", 3.0)
    assert not head.worn
    kb._handle("c", 3.0)
    assert not source.streaming
    kb._handle("q", 3.0)
    assert quit_called
    kb.close()

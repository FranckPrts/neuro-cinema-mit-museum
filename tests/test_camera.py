"""The camera relay: the bench's /nc/cam/* in, re-published by the engine."""

import time

import pytest
from helpers import RecordingOutput
from pythonosc.udp_client import SimpleUDPClient

from neurocinema.app import App
from neurocinema.camera import CameraRelay
from neurocinema.config import Config, ConfigError
from neurocinema.engine import Phase, State
from neurocinema.sources.none import NoSource


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(stale=3.0):
    out, clock = RecordingOutput(), Clock()
    return CameraRelay(out, "/nc", stale, clock=clock), out, clock


def sent(out):
    return [(a, args) for _, a, args in out.messages]


def test_forwards_and_announces_connection():
    relay, out, _ = make()
    relay.receive("/nc/cam/event", "camera_started")
    relay.receive("/nc/cam/state", "active")
    relay.receive("/nc/cam/target", "left", "top")
    relay.receive("/nc/cam/dwell", 0.5)
    relay.receive("/nc/cam/pose", 10.0, -3.0, 0.0)
    relay.receive("/nc/cam/calib", *([1.0] * 11))
    relay.poll("choosing")
    assert sent(out) == [
        ("/nc/cam/event", ("camera_connected",)),
        ("/nc/cam/state", ("active",)),
        ("/nc/cam/target", ("left", "top")),
        ("/nc/cam/dwell", (0.5,)),
        ("/nc/cam/pose", (10.0, -3.0, 0.0)),
        ("/nc/cam/calib", tuple([1.0] * 11)),
    ]


def test_drops_heartbeat_legacy_and_foreign_addresses():
    relay, out, _ = make()
    for address, args in [("/nc/cam/heartbeat", (1.0,)), ("/nc/cam/aim", (1.0, 2.0)),
                          ("/nc/cam/highlight", ("left",)), ("/nc/cam/select", ("left",)),
                          ("/nc/state", ("idle",))]:
        relay.receive(address, *args)
    relay.poll("choosing")
    assert sent(out) == [("/nc/cam/event", ("camera_connected",))]


def test_selection_passes_while_choosing():
    relay, out, _ = make()
    relay.receive("/nc/cam/selected", "right", "bottom")
    relay.receive("/nc/cam/event", "selected", "right", "bottom")
    relay.receive("/nc/cam/state", "selected")
    relay.poll("choosing")
    assert sent(out)[1:] == [
        ("/nc/cam/selected", ("right", "bottom")),
        ("/nc/cam/event", ("selected", "right", "bottom")),
        ("/nc/cam/state", ("selected",)),
    ]


def test_selection_blocked_while_watching():
    relay, out, _ = make()
    relay.receive("/nc/cam/selected", "right", "bottom")
    relay.receive("/nc/cam/event", "selected", "right", "bottom")
    relay.receive("/nc/cam/state", "selected")
    relay.poll("watching")
    assert sent(out)[1:] == [
        ("/nc/cam/event", ("selected_ignored", "watching")),
        ("/nc/cam/state", ("active",)),
    ]


def test_lost_after_silence_clears_state_and_target():
    relay, out, clock = make(stale=3.0)
    relay.receive("/nc/cam/target", "middle", "top")
    relay.poll("choosing")
    clock.t += 2.9
    relay.poll("choosing")
    assert relay.connected
    clock.t += 0.2
    relay.poll("choosing")
    assert not relay.connected
    assert sent(out)[-3:] == [
        ("/nc/cam/event", ("camera_lost",)),
        ("/nc/cam/state", ("idle",)),
        ("/nc/cam/target", ("none", "none")),
    ]
    relay.receive("/nc/cam/heartbeat", 5.0)  # back again
    relay.poll("choosing")
    assert relay.connected and sent(out)[-1] == ("/nc/cam/event", ("camera_connected",))


def test_camera_stopped_is_lost_at_once():
    relay, out, _ = make()
    relay.receive("/nc/cam/state", "active")
    relay.receive("/nc/cam/event", "camera_stopped")
    relay.poll("choosing")
    assert not relay.connected
    assert ("/nc/cam/event", ("camera_lost",)) in sent(out)
    assert ("/nc/cam/event", ("camera_stopped",)) not in sent(out)


def test_snapshot_resends_state_and_target():
    relay, out, _ = make()
    relay.receive("/nc/cam/state", "active")
    relay.receive("/nc/cam/target", "left", "bottom")
    relay.poll("choosing")
    out.messages.clear()
    relay.snapshot()
    assert sent(out) == [("/nc/cam/state", ("active",)), ("/nc/cam/target", ("left", "bottom"))]


def test_app_relays_and_gates_by_phase():
    cfg = Config()
    out = RecordingOutput()
    relay = CameraRelay(out, "/nc", 3.0)
    app = App(cfg, NoSource(), out, camera=relay)
    app.handle_command("phase", ("watching",), 0.0)
    relay.receive("/nc/cam/selected", "left", "top")
    app.step()
    assert app.engine.phase == Phase.WATCHING
    assert ("/nc/cam/selected", ("left", "top")) not in sent(out)
    assert app.engine.state == State.DISCONNECTED  # no headset: the IMU side stays disconnected
    out.messages.clear()
    app.handle_command("snapshot", (), 0.0)
    assert not [a for a, _ in sent(out) if a.startswith("/nc/cam/")]  # no camera state or target yet


def test_over_udp():
    relay, out, _ = make()
    relay.listen("127.0.0.1", 0)
    port = relay._server.server_address[1]
    try:
        SimpleUDPClient("127.0.0.1", port).send_message("/nc/cam/target", ["right", "top"])
        deadline = time.time() + 3
        while time.time() < deadline and ("/nc/cam/target", ("right", "top")) not in sent(out):
            relay.poll("choosing")
            time.sleep(0.02)
        assert ("/nc/cam/target", ("right", "top")) in sent(out)
    finally:
        relay.close()


def test_config_rejects_port_clash():
    cfg = Config()
    cfg.camera.listen_port = 7001
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg.camera.listen_port = 7004
    cfg.source.kind = "none"
    cfg.validate()

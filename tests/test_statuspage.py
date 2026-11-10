"""Status page engine card and buttons, with and without the headset manager."""

import json
import urllib.error
import urllib.request

import pytest

from neurocinema.app import App, PageLink
from neurocinema.config import Config
from neurocinema.engine import State
from neurocinema.samples import ImuSample
from neurocinema.sim.head import RATE_HZ, VirtualHead
from neurocinema.statuspage import StatusPage


class StepSource:
    """A virtual head on a simulated clock: one sample pair per App.step()."""

    def __init__(self, head: VirtualHead):
        self.head = head
        self.t = 0.0

    def poll(self):
        acc, gyro = self.head.step(self.t)
        out = [ImuSample(self.t, "acc", *acc), ImuSample(self.t, "gyro", *gyro)]
        self.t += 1.0 / RATE_HZ
        return out

    def now(self):
        return self.t

    def connected(self):
        return True

    def last_receive(self):
        return self.t

    def rate_hz(self):
        return RATE_HZ

    def headset_id(self):
        return "test"

    def reset(self):
        pass


class EventsOut:
    def __init__(self):
        self.events = []

    def send(self, address, *args):
        if address.endswith("/event"):
            self.events.append(args[0])


def run(app: App, seconds: float) -> None:
    for _ in range(int(seconds * RATE_HZ)):
        app.step()


def test_page_buttons_drive_the_engine():
    out = EventsOut()
    cfg = Config()
    cfg.imu.backend = "complementary"  # its own stand bias: bias_source "idle" below
    app = App(cfg, StepSource(VirtualHead(seed=5, worn=False)), out)
    run(app, 7.0)  # on its stand
    status = app.engine.status(app.source.now())
    assert status["state"] == "idle"
    assert status["stand_still_s"] >= status["stand_needed_s"]

    app.page.submit("recalibrate")
    run(app, 0.1)
    assert app.engine.state == State.CALIBRATING
    assert app.page.status()["state"] == "calibrating"  # published at once, not at the next status tick
    run(app, 3.0)
    assert app.engine.state == State.ACTIVE
    status = app.engine.status(app.source.now())
    assert status["calibrated"] and status["bias_source"] == "idle"
    assert status["yaw_deg"] is not None and status["tilt_deg"] is not None

    app.page.submit("reset")
    run(app, 0.1)
    assert "reset" in out.events

    app.page.submit("idle")
    run(app, 0.1)
    assert app.engine.state == State.IDLE
    assert app.page.status()["state"] == "idle"


def test_page_link_only_takes_page_commands():
    link = PageLink()
    with pytest.raises(ValueError):
        link.submit("phase")
    link.submit("idle")
    assert link.drain() == ["idle"]
    assert link.drain() == []


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=5))


def _post(url, body=b"{}"):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=5))


def test_status_page_without_headsets_serves_the_engine_and_its_buttons():
    link = PageLink()
    link.publish({"state": "active", "calibrated": True})
    page = StatusPage(None, "127.0.0.1", 0, engine=link)
    page.start()
    try:
        assert _get(page.address + "/api/engine")["state"] == "active"
        assert _get(page.address + "/api/status") == {"managed": False}
        assert b"Calibrate now" in urllib.request.urlopen(page.address + "/", timeout=5).read()
        assert _post(page.address + "/api/engine/recalibrate") == {"ok": True}
        assert _post(page.address + "/api/engine/idle") == {"ok": True}
        for path in ("/api/engine/phase", "/api/select"):  # not a page command; no headsets here
            with pytest.raises(urllib.error.HTTPError):
                _post(page.address + path, b'{"label": "A"}')
    finally:
        page.close()
    assert link.drain() == ["recalibrate", "idle"]

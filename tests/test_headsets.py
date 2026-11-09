"""Headset manager + Bluetooth worker with simulated headsets on a fake clock."""

import json
import urllib.request

import pytest

from neurocinema.ble import BleWorker, FakeDriver
from neurocinema.config import ConfigError, HeadsetsConfig, load_config
from neurocinema.headsets import Alert, HeadsetManager, Phase

REGISTRY = ["A=Muse-AAAA", "B=Muse-BBBB"]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Harness:
    def __init__(self, batteries=None, lifted_after_s=3.0, **cfg):
        self.clock = Clock()
        self.driver = FakeDriver(
            batteries or {"Muse-AAAA": 90.0, "Muse-BBBB": 80.0, "Muse-ZZZZ": 50.0},
            clock=self.clock, realtime=False, lifted_after_s=lifted_after_s,
        )
        self.worker = BleWorker(self.driver, start_thread=False)
        self.cfg = HeadsetsConfig(registry=cfg.pop("registry", REGISTRY), **cfg)
        self.m = HeadsetManager(self.cfg, self.worker, clock=self.clock)
        self.samples = []

    def run(self, seconds, dt=0.05):
        end = self.clock.t + seconds
        while self.clock.t < end:
            self.clock.t += dt
            for _ in range(4):
                self.worker.step()
            self.samples += self.m.poll()

    def use(self, label):
        self.m.submit("select", label)
        self.run(5.0)

    @property
    def snap(self):
        return self.m.snapshot()


def test_staff_swap_is_used_only_once_lifted():
    h = Harness()
    h.run(1.0)
    assert h.m.phase == Phase.NO_HEADSET
    h.m.submit("select", "A")
    h.run(1.0)
    assert h.m.phase == Phase.CONFIRMING
    assert h.samples == []  # connected, but not used until lifted
    assert not h.m.connected()
    h.run(4.0)  # the fake headset is lifted 3 s after connecting
    assert h.m.phase == Phase.ACTIVE
    assert h.m.connected()
    assert len(h.samples) > 50
    assert h.snap["current"] == "A"


def test_headset_not_lifted_is_released():
    h = Harness(lifted_after_s=None, confirm_timeout_s=10.0)
    h.m.submit("select", "B")
    h.run(12.0)
    assert h.m.phase == Phase.NO_HEADSET
    assert h.m.current is None
    assert "not picked up" in h.m.message
    assert h.samples == []


def test_switching_headsets_without_restart():
    h = Harness()
    h.use("A")
    assert h.m.phase == Phase.ACTIVE
    assert h.m.headset_id() == "A"
    h.use("B")
    assert h.m.phase == Phase.ACTIVE
    assert h.m.current == "B"
    assert h.m.headset_id() == "B"  # the engine drops what it learned about A
    assert h.driver._name == "Muse-BBBB"


def test_unregistered_headset_is_listed_but_never_used():
    h = Harness()
    h.run(1.0)  # first scan
    assert h.snap["unknown"] == ["Muse-ZZZZ"]
    assert [x["label"] for x in h.snap["headsets"]] == ["A", "B"]
    h.m.submit("select", "Z")  # not a label: ignored
    h.run(2.0)
    assert h.m.phase == Phase.NO_HEADSET


def test_lost_link_reconnects_the_same_headset():
    h = Harness(reconnect_interval_s=2.0)
    h.use("A")
    h.driver.link_up = False  # data stops (out of range)...
    h.driver.switched_on.discard("Muse-AAAA")  # ...and it cannot be reached
    h.run(6.0)
    assert h.m.phase == Phase.RECONNECTING
    assert not h.m.connected()
    h.driver.switched_on.add("Muse-AAAA")  # back in range
    h.run(3.0)
    assert h.m.phase == Phase.ACTIVE
    assert h.m.current == "A"
    assert h.m.headset_id() == "A"  # same headset: the engine keeps its idle bias


def test_failed_connects_give_up_with_a_message():
    h = Harness(connect_attempts=2)
    h.driver.fail_connect.add("Muse-BBBB")
    h.m.submit("select", "B")
    h.run(2.0)
    assert h.m.phase == Phase.NO_HEADSET
    assert "Could not reach headset B" in h.m.message


def test_battery_low_means_swap_soon():
    h = Harness(batteries={"Muse-AAAA": 19.0, "Muse-BBBB": 80.0})
    h.use("A")
    assert h.m.alert(h.clock.t) == Alert.SWAP_SOON
    assert "Swap headset A soon" in h.snap["headline"]
    h.use("B")
    assert h.m.alert(h.clock.t) == Alert.OK
    assert h.snap["headline"].startswith("All good: headset B, 80 %")


def test_time_left_estimate_from_drain():
    h = Harness(batteries={"Muse-AAAA": 60.0, "Muse-BBBB": 80.0})
    h.driver.drain_per_min = 1.0
    h.use("A")
    h.run(8 * 60.0, dt=0.5)
    a = next(x for x in h.snap["headsets"] if x["label"] == "A")
    assert a["battery"] == pytest.approx(52, abs=1)
    assert a["minutes_left"] == pytest.approx(52, abs=3)
    assert h.m.alert(h.clock.t) == Alert.OK
    h.run(25 * 60.0, dt=0.5)  # under 30 min left
    assert h.m.alert(h.clock.t) == Alert.SWAP_SOON


def test_offline_after_a_minute_without_headset():
    h = Harness(offline_after_s=60.0)
    h.run(30.0, dt=0.5)
    assert h.m.alert(h.clock.t) == Alert.OK
    h.run(31.0, dt=0.5)
    assert h.m.alert(h.clock.t) == Alert.OFFLINE
    assert h.snap["headline"].startswith("Offline")


def test_auto_adopt_only_with_one_registered_headset_in_range():
    h = Harness(auto_adopt_after_s=30.0)
    h.run(40.0, dt=0.5)
    assert h.m.phase == Phase.NO_HEADSET  # A and B both on: do not guess
    h.driver.switched_on.discard("Muse-BBBB")
    h.run(25.0, dt=0.5)
    assert h.m.phase == Phase.ACTIVE
    assert h.m.current == "A" and h.m.auto_adopted


def test_registry_config(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[source]\nkind = "muse_direct"\n[headsets]\nregistry = ["A=Muse-41D2", "B = Muse-0C3A"]\n')
    cfg = load_config([p])
    assert cfg.headsets.entries() == [("A", "Muse-41D2"), ("B", "Muse-0C3A")]
    for bad in ('["Muse-41D2"]', '["A=Muse-1", "A=Muse-2"]', '["A=Muse-1", "B=Muse-1"]'):
        p.write_text(f"[headsets]\nregistry = {bad}\n")
        with pytest.raises(ConfigError):
            load_config([p])


def test_status_page_serves_status_and_takes_commands():
    from neurocinema.statuspage import StatusPage

    h = Harness()
    page = StatusPage(h.m, "127.0.0.1", 0)
    page.start()
    try:
        status = json.load(urllib.request.urlopen(page.address + "/api/status", timeout=5))
        assert status["phase"] == "no_headset"
        html = urllib.request.urlopen(page.address + "/", timeout=5).read()
        assert b"Neurocinema status" in html
        req = urllib.request.Request(page.address + "/api/select", data=b'{"label": "B"}',
                                     headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req, timeout=5)) == {"ok": True}
        bad = urllib.request.Request(page.address + "/api/select", data=b'{"label": "Z"}')
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad, timeout=5)
    finally:
        page.close()
    h.run(1.0)
    assert h.m.current == "B"


def test_threaded_worker_end_to_end():
    """Real thread, real time, simulated headsets: the engine sees samples."""
    import time

    driver = FakeDriver({"Muse-AAAA": 90.0}, lifted_after_s=0.3)
    worker = BleWorker(driver)
    m = HeadsetManager(HeadsetsConfig(registry=["A=Muse-AAAA"]), worker)
    try:
        m.submit("select", "A")
        got = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(got) < 40:
            got += m.poll()
            time.sleep(0.02)
        assert m.phase == Phase.ACTIVE
        assert len(got) >= 40
        assert m.connected()
    finally:
        m.close()


class RecordingOut:
    def __init__(self):
        self.sent = []

    def send(self, address, *args):
        self.sent.append((address, args))


def test_app_sends_headset_status_only_for_the_headset_manager():
    from neurocinema.app import App
    from neurocinema.config import Config
    from neurocinema.sim.head import VirtualHead
    from neurocinema.sources.sim import SimSource

    h = Harness()
    h.use("A")
    assert h.snap["phase"] == "active"
    out = RecordingOut()
    App(Config(), h.m, out)._emit_status(h.clock.t)
    headset = [args for address, args in out.sent if address == "/nc/headset"]
    assert len(headset) == 1
    label, battery, left, phase, alert = headset[0]
    assert (label, phase, alert) == ("A", "active", "ok")
    assert battery == float(h.snap["headsets"][0]["battery"]) and battery > 0
    assert left == -1.0  # not enough history for an estimate yet

    out = RecordingOut()
    App(Config(), SimSource(VirtualHead(seed=1)), out)._emit_status(0.0)
    assert any(address == "/nc/status" for address, _ in out.sent)
    assert not any(address == "/nc/headset" for address, _ in out.sent)

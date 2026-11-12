"""Heart rate from the headset PPG: estimator, PPG stream, engine output."""

import numpy as np
import pytest

from neurocinema.app import App
from neurocinema.ble import PPG_PRESET, BleWorker, FakeDriver
from neurocinema.config import Config, ConfigError, HeartRateConfig, HeadsetsConfig
from neurocinema.headsets import HeadsetManager, Phase
from neurocinema.heartrate import HeartRate, estimate_bpm

from helpers import RecordingOutput

RATE = 64.0


def pulse(bpm, seconds=8.0, noise=150.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / RATE)
    phase = (t * bpm / 60.0) % 1.0
    beat = np.exp(-((phase - 0.15) ** 2) / 0.004) + 0.4 * np.exp(-((phase - 0.45) ** 2) / 0.01)
    drift = 300.0 * np.sin(2 * np.pi * 0.1 * t)
    return t, 120_000.0 + drift - 2_000.0 * beat + rng.normal(0.0, noise, len(t))


@pytest.mark.parametrize("bpm", [45, 60, 72, 95, 130, 170])
def test_estimate_finds_the_pulse_rate(bpm):
    _, v = pulse(bpm)
    got, quality = estimate_bpm(v, RATE, 40.0, 180.0)
    assert got == pytest.approx(bpm, abs=2.0)
    assert quality >= 0.8


@pytest.mark.parametrize("seed", range(5))
def test_noise_alone_is_not_trusted(seed):
    rng = np.random.default_rng(seed)
    for v in (rng.normal(0, 1, 512), rng.normal(0, 1, 512).cumsum()):
        _, quality = estimate_bpm(v, RATE, 40.0, 180.0)
        assert quality < HeartRateConfig().min_quality


def feed(hr, t, v, start=0.0):
    for ti, vi in zip(t, v):
        hr.add(start + ti, 0.0, vi, 0.0)
    return start + t[-1]


def test_heart_rate_waits_for_a_full_window_then_reports():
    hr = HeartRate(HeartRateConfig())
    t, v = pulse(72, seconds=4.0)
    now = feed(hr, t, v)
    assert hr.update(now).bpm is None
    t, v = pulse(72, seconds=10.0)
    now = feed(hr, t, v, start=now + 1.0 / RATE)
    reading = hr.update(now + 1.0)  # past the next scheduled update
    assert reading.bpm == pytest.approx(72, abs=2.0)
    assert reading.age_s == pytest.approx(0.0)
    assert not reading.moving


def test_head_movement_holds_the_last_value_then_drops_it():
    cfg = HeartRateConfig(hold_s=5.0)
    hr = HeartRate(cfg)
    t, v = pulse(80, seconds=10.0)
    now = feed(hr, t, v)
    assert hr.update(now).bpm == pytest.approx(80, abs=2.0)
    # The head turns; the signal turns to junk, but the last estimate is held.
    rng = np.random.default_rng(3)
    for k in range(1, 5):
        hr.motion(now + k, 90.0)
        for i in range(int(RATE)):
            hr.add(now + k - 1 + i / RATE, 0.0, 120_000 + rng.normal(0, 5_000), 0.0)
        reading = hr.update(now + k)
        assert reading.moving
        assert reading.bpm == pytest.approx(80, abs=2.0)
        assert reading.age_s == pytest.approx(k)
    assert hr.update(now + 6.0).bpm is None  # held longer than hold_s


def test_heart_rate_config_is_checked():
    cfg = Config()
    cfg.heart_rate.channel = "green"
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = Config()
    cfg.heart_rate.min_bpm = 200.0
    with pytest.raises(ConfigError):
        cfg.validate()


# --- PPG stream through the headset manager --------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def harness(ppg=True):
    clock = Clock()
    driver = FakeDriver({"Muse-AAAA": 90.0}, clock=clock, realtime=False, lifted_after_s=1.0)
    worker = BleWorker(driver, start_thread=False, ppg=ppg)
    manager = HeadsetManager(HeadsetsConfig(registry=["A=Muse-AAAA"]), worker, clock=clock)
    return clock, driver, worker, manager


def run(clock, worker, manager, seconds, dt=0.05, ppg=None):
    end = clock.t + seconds
    while clock.t < end:
        clock.t += dt
        for _ in range(4):
            worker.step()
        manager.poll()
        readings = manager.poll_ppg()
        if ppg is not None:
            ppg += readings


def test_ppg_is_its_own_stream_from_the_headset_in_use():
    clock, driver, worker, manager = harness()
    manager.submit("select", "A")
    run(clock, worker, manager, 3.0)
    assert manager.phase == Phase.ACTIVE
    got = []
    run(clock, worker, manager, 2.0, ppg=got)
    assert len(got) == pytest.approx(2.0 * RATE, abs=4)
    assert {r.label for r in got} == {"A"}
    assert all(b.t > a.t for a, b in zip(got, got[1:]))


def test_ppg_off_streams_nothing():
    clock, driver, worker, manager = harness(ppg=False)
    manager.submit("select", "A")
    got = []
    run(clock, worker, manager, 4.0, ppg=got)
    assert manager.phase == Phase.ACTIVE
    assert got == []


def test_ppg_preset_switches_the_sensor_on():
    assert PPG_PRESET == "p50"


def test_engine_sends_heart_rate_from_the_simulated_headset():
    clock, driver, worker, manager = harness()
    cfg = Config()
    out = RecordingOutput()
    app = App(cfg, manager, out)
    manager.submit("select", "A")
    end = clock.t + 15.0
    while clock.t < end:
        clock.t += 0.02
        for _ in range(2):
            worker.step()
        app.step()
    app._next_status = 0.0
    app._emit_status(clock.t)
    bpm, quality, age, moving = out.last("/heart")
    expected = driver.heart_bpm["Muse-AAAA"]
    assert bpm == pytest.approx(expected, abs=3.0)
    assert quality >= cfg.heart_rate.min_quality
    assert 0.0 <= age <= 2.0
    assert moving == 0


# --- on-a-head check for the staff page --------------------------------------------------------


def test_worn_when_a_pulse_is_found_and_not_worn_after_a_few_clear_misses():
    hr = HeartRate(HeartRateConfig())
    assert hr.update(0.0).worn is None
    t, v = pulse(70, seconds=10.0)
    now = feed(hr, t, v)
    assert hr.update(now).worn is True
    # Put on the stand (still): the sensor only sees noise.
    rng = np.random.default_rng(4)
    worn = []
    for k in range(1, 15):
        for i in range(int(RATE)):
            hr.add(now + k - 1 + (i + 1) / RATE, 0.0, 120_000 + rng.normal(0, 150), 0.0)
        worn.append(hr.update(now + k).worn)
    assert worn[0] is True  # one bad window is not enough
    assert worn[-1] is False


def test_worn_unknown_when_ppg_stops():
    hr = HeartRate(HeartRateConfig())
    t, v = pulse(70, seconds=10.0)
    now = feed(hr, t, v)
    assert hr.update(now).worn is True
    assert hr.update(now + 3.0).worn is None


def test_page_status_carries_only_the_worn_state():
    clock, driver, worker, manager = harness()
    app = App(Config(), manager, RecordingOutput())
    assert app._page_status(clock.t)["pulse"] == "waiting"
    manager.submit("select", "A")
    end = clock.t + 15.0
    while clock.t < end:
        clock.t += 0.02
        for _ in range(2):
            worker.step()
        app.step()
    status = app._page_status(clock.t)
    assert status["pulse"] == "worn"
    assert not any("bpm" in k or "heart" in k for k in status)


def test_page_status_without_ppg_has_no_pulse():
    clock, driver, worker, manager = harness(ppg=False)
    cfg = Config()
    cfg.heart_rate.enabled = False
    app = App(cfg, manager, RecordingOutput())
    assert app._page_status(clock.t)["pulse"] is None

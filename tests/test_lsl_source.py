"""LslImuSource against fake outlets shaped exactly like muselsl's."""

import time

import pytest

pylsl = pytest.importorskip("pylsl")

from neurocinema.config import MuseConfig  # noqa: E402
from neurocinema.sources.lsl import LslImuSource  # noqa: E402


def make_outlets(address):
    outlets = {}
    for stype in ("ACC", "GYRO"):
        info = pylsl.StreamInfo("Muse", stype, 3, 52, "float32", f"Muse{address}")
        outlets[stype] = pylsl.StreamOutlet(info, 1)
    return outlets


def pump(source, outlets, seconds, until=None):
    """Push muselsl-like samples while polling the source."""
    got = []
    end = time.time() + seconds
    while time.time() < end:
        if outlets is not None:
            ts = pylsl.local_clock()
            outlets["ACC"].push_sample([0.0, 0.0, 1.0], ts)
            outlets["GYRO"].push_sample([0.5, -0.2, 0.1], ts)
        got += source.poll()
        if until is not None and until():
            break
        time.sleep(1 / 52)
    return got


def test_discovers_reads_and_recovers_after_outlet_restart():
    address = f"TEST{int(time.time() * 1000) % 100000}"
    cfg = MuseConfig(source_id=f"Muse{address}", stale_after_s=1.0, resolve_wait_s=0.5)
    source = LslImuSource(cfg)
    try:
        outlets = make_outlets(address)
        pump(source, outlets, 10.0, until=source.connected)
        got = pump(source, outlets, 1.0)
        assert {s.kind for s in got} == {"acc", "gyro"}
        assert source.connected()
        assert source.rate_hz() > 10  # 2 s window, about 1 s of data so far
        gyro = [s for s in got if s.kind == "gyro"]
        assert gyro[-1].x == pytest.approx(0.5)
        assert abs(gyro[-1].t - pylsl.local_clock()) < 1.0  # same clock as now()

        # muselsl restarts: old outlets vanish, new ones appear.
        del outlets
        pump(source, None, 2.0)
        assert not source.connected()
        outlets = make_outlets(address)
        pump(source, outlets, 10.0, until=source.connected)
        assert source.connected()
    finally:
        source.close()

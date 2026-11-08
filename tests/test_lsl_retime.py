"""muselsl stamps a Muse IMU packet's 3 samples alike; the LSL source spreads them.

No real LSL here: fake inlets are handed to the source directly.
"""

from itertools import pairwise

import pytest

pytest.importorskip("pylsl")

from neurocinema.config import MuseConfig
from neurocinema.sources import lsl
from neurocinema.sources.lsl import LslImuSource, Retimer

DT = 1 / 52


def muselsl_stamps(t0, packets):
    """Arrival stamps as muselsl gives them: one per packet, repeated 3 times."""
    return [t0 + p * 3 * DT for p in range(packets) for _ in range(3)]


def increasing(ts):
    return all(b > a for a, b in pairwise(ts))


def test_spreads_packets_back_over_the_sample_period():
    out = Retimer()(muselsl_stamps(100.0, 4))
    assert out == pytest.approx([100.0 - 2 * DT + k * DT for k in range(12)])
    assert out[2] == 100.0 and out[-1] == pytest.approx(100.0 + 9 * DT)


def test_burst_is_spread_over_the_time_since_the_previous_sample():
    out = Retimer()([10.0] * 3 + [10.012] * 3)
    assert out[:3] == pytest.approx([10.0 - 2 * DT, 10.0 - DT, 10.0])
    assert out[3:] == pytest.approx([10.004, 10.008, 10.012])


def test_distinct_stamps_pass_through():
    stamps = [1.0, 1.02, 1.04, 1.06]
    assert Retimer()(stamps) == stamps


def test_a_packet_split_across_pulls_gets_the_same_times_as_a_whole_one():
    stamps = muselsl_stamps(5.0, 20)
    whole = Retimer()(stamps)
    r = Retimer()
    cuts = [0, 4, 5, 13, 20, 21, 22, 60]
    split = [t for a, b in pairwise(cuts) for t in r(stamps[a:b])]
    assert split == pytest.approx(whole)
    assert increasing(split)


def test_stays_increasing_when_a_run_is_longer_than_expected():
    r = Retimer()
    out = r([1.0] * 3 + [1.06] * 3)  # learns runs of 3
    out += r([1.07])  # a lone stamp, right after: burst rule
    out += r([1.07] * 4)  # and more of it than a packet
    out += r([1.08])  # stamp now behind what was handed out
    assert increasing(out)


class FakeInlet:
    def __init__(self):
        self.queue = []

    def push(self, values, stamps):
        self.queue.append((values, stamps))

    def pull_chunk(self, timeout=0.0):
        data, stamps = [], []
        for v, s in self.queue:
            data += v
            stamps += s
        self.queue.clear()
        return data, stamps

    def close_stream(self):
        pass


def test_poll_delivers_every_muselsl_sample_with_increasing_stamps(monkeypatch):
    monkeypatch.setattr(LslImuSource, "_resolver", lambda self: None)
    source = LslImuSource(MuseConfig(stale_after_s=1e9))
    inlets = {"ACC": FakeInlet(), "GYRO": FakeInlet()}
    source._pending = inlets
    try:
        t = source.now()
        got = []
        for p in range(30):
            ts = t + p * 3 * DT
            inlets["ACC"].push([[0.0, 0.0, 1.0]] * 3, [ts] * 3)
            inlets["GYRO"].push([[0.1 * k, 0.0, 0.0] for k in range(3)], [ts] * 3)
            if p % 4 == 3:  # several packets per poll, sometimes
                got += source.poll()
        got += source.poll()

        for kind in ("acc", "gyro"):
            ts = [s.t for s in got if s.kind == kind]
            assert len(ts) == 90
            assert increasing(ts)
        gyro = [s for s in got if s.kind == "gyro"]
        assert [s.x for s in gyro[:3]] == pytest.approx([0.0, 0.1, 0.2])  # order kept
    finally:
        source.close()


def test_new_inlets_start_from_a_clean_retimer(monkeypatch):
    monkeypatch.setattr(LslImuSource, "_resolver", lambda self: None)
    source = LslImuSource(MuseConfig(stale_after_s=1e9))
    try:
        first = {"ACC": FakeInlet(), "GYRO": FakeInlet()}
        source._pending = first
        first["GYRO"].push([[0.0, 0.0, 0.0]] * 3, [50.0] * 3)
        source.poll()
        source.reset()
        # A new outlet whose clock is behind the old one is not pushed forward.
        second = {"ACC": FakeInlet(), "GYRO": FakeInlet()}
        source._pending = second
        second["GYRO"].push([[0.0, 0.0, 0.0]] * 3, [20.0] * 3)
        ts = [s.t for s in source.poll()]
        assert ts == pytest.approx([20.0 - 2 * DT, 20.0 - DT, 20.0])
    finally:
        source.close()


def test_rate_comes_from_ble_constant():
    assert Retimer().dt == pytest.approx(1 / lsl.IMU_RATE_HZ)

"""Bench tools on a simulated session (no headset): python -m pytest bench/imu"""

import importlib.util
from itertools import pairwise

import numpy as np
import protocol
import pytest
import replay
from recording import load_session, read_events, read_imu, retime, timing
from simulate import simulate

from neurocinema.samples import ImuSample

HAVE_VQF = importlib.util.find_spec("vqf") is not None


@pytest.fixture(scope="module")
def quick_session(tmp_path_factory):
    return simulate(tmp_path_factory.mktemp("rec") / "SIM01", quick=True, seed=3, duplicates=True)


def test_protocol_never_repeats_a_target_and_has_ground_truth():
    steps = protocol.build(protocol.Layout(), stand_s=120)
    faces = [(s.column, s.row) for s in steps if s.name in ("face", "nod")]
    assert all(a != b for a, b in pairwise(faces))
    assert all(s.yaw_deg is not None for s in steps if s.kind == "prompt")
    assert 5.5 * 60 < protocol.duration_s(steps) < 6.5 * 60


def test_session_files_round_trip(quick_session):
    samples = read_imu(quick_session / "imu.csv")
    events = read_events(quick_session / "events.csv")
    assert {s.kind for s in samples} == {"acc", "gyro"}
    assert [e.name for e in events if e.kind == "phase"][:3] == ["stand", "put_on", "targets"]
    assert load_session(quick_session).meta["headset"] == "sim"


def test_retime_spreads_duplicate_timestamps():
    # Packets 3/52 s apart, then one arriving in a burst 10 ms after the last.
    s = [ImuSample(t, "gyro", 0, 0, 0) for t in [1.0] * 3 + [1.0577] * 3 + [1.0677] * 3]
    out, squeezed = retime(s)
    ts = [x.t for x in out]
    assert all(b > a for a, b in pairwise(ts))
    assert ts[2] == 1.0 and ts[5] == 1.0577 and ts[-1] == 1.0677
    assert ts[1] == pytest.approx(1.0 - 1 / 52)
    assert timing(s).duplicates["gyro"] == 6 and timing(out).duplicates["gyro"] == 0
    assert squeezed["gyro"] == 3  # the burst is spread over the 10 ms since the previous packet


def test_variants_parse_overrides():
    name, data = replay.parse_variant("vqf:vqf_tau_acc_s=2.0,gestures.nod.peak_dps=50")
    assert name == "vqf:vqf_tau_acc_s=2.0,gestures.nod.peak_dps=50"
    assert data["imu"] == {"backend": "vqf", "vqf_tau_acc_s": 2.0}
    assert data["gestures"] == {"nod": {"peak_dps": 50}}


@pytest.mark.parametrize("variant", ["complementary", pytest.param(
    "vqf", marks=pytest.mark.skipif(not HAVE_VQF, reason="pip install -e '.[vqf]'"))])
def test_replay_of_a_simulated_session(quick_session, variant):
    session = load_session(quick_session)
    results = replay.replay([session], [replay.parse_variant(variant)], [replay.ROOT / "config/installation.toml"])
    (r,) = results[variant]
    s = replay.summarize([r])
    assert r.calibrated_after_s is not None and r.calibrated_after_s < 10
    assert r.bias_source in ("idle", "vqf")
    assert s["yaw_med"] < 3 and s["pitch_med"] < 2
    assert s["target_ok"] == 1.0
    assert s["sel_ok"] == "3/3" and s["shakes"] == "1/1" and s["false"] == "0/0"


def test_duplicates_without_retime_lose_samples(quick_session):
    session = load_session(quick_session)
    kept = replay.replay([session], [replay.parse_variant("complementary")],
                         [replay.ROOT / "config/installation.toml"], do_retime=False)
    spread = replay.replay([session], [replay.parse_variant("complementary")],
                           [replay.ROOT / "config/installation.toml"], do_retime=True)
    err = {k: np.median([abs(p.yaw_err) for p in v["complementary"][0].face]) for k, v in
           (("kept", kept), ("spread", spread))}
    assert err["spread"] < err["kept"]


def test_sign_test_needs_a_clear_majority():
    assert replay.sign_test_p(2, 2) == 0.5
    assert replay.sign_test_p(6, 6) < 0.05 < replay.sign_test_p(5, 5)
    assert replay.sign_test_p(0, 0) == 1.0


@pytest.mark.skipif(not HAVE_VQF, reason="pip install -e '.[vqf]'")
def test_report_applies_the_decision_rule(quick_session, tmp_path):
    session = load_session(quick_session)
    variants = [replay.parse_variant(v) for v in ("complementary", "vqf")]
    base = [replay.ROOT / "config/installation.toml"]
    results = replay.replay([session], variants, base)
    switch, _ = replay.decide(results)
    assert not switch  # one short session can never be a clear difference
    report = tmp_path / "results.md"
    replay.write_report(report, [session], results, base, (2.5, 4.5), retimed=True)
    text = report.read_text()
    assert "Keep `complementary` as the default" in text
    assert "Includes simulated sessions" in text
    assert "| complementary |" in text and "| vqf |" in text

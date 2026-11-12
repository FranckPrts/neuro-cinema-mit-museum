"""nc-osc-record / nc-osc-play: what is recorded is played back the same, with its timing."""

from __future__ import annotations

import json
import socket
import threading
import time

from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_packet import OscPacket
from pythonosc.udp_client import UDPClient

from neurocinema import oscrec


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _msg(address: str, *typed: tuple[object, str]) -> bytes:
    b = OscMessageBuilder(address=address)
    for v, t in typed:
        b.add_arg(v, t)
    return b.build().dgram


def _record(tmp_path, datagrams: list[tuple[float, bytes]], prefixes=()) -> "Path":
    port = _free_port()
    path = tmp_path / "rec.jsonl"
    rec = oscrec.Recorder(path, "127.0.0.1", port, list(prefixes), quiet=True)
    stop = [False]
    th = threading.Thread(target=rec.run, args=(stop,))
    th.start()
    client = UDPClient("127.0.0.1", port)
    t0 = time.monotonic()
    for at, dgram in datagrams:
        time.sleep(max(0.0, t0 + at - time.monotonic()))
        client._sock.sendto(dgram, ("127.0.0.1", port))
    time.sleep(0.1)
    stop[0] = True
    th.join()
    rec.close()
    return path


MESSAGES = [
    (0.0, _msg("/nc/event", ("engine_started", "s"))),
    (0.1, _msg("/nc/pose", (12.5, "f"), (-3.0, "f"), (0.0, "f"))),
    (0.2, _msg("/nc/status", (True, "T"), (17, "i"), (52.0, "f"), (1.25, "d"))),
    (0.3, _msg("/nc/cam/target", ("left", "s"), ("top", "s"))),
    (0.4, _msg("/nc/blob", (b"\x00\x01\xff", "b"))),
]


def test_record_writes_jsonl_with_types(tmp_path):
    path = _record(tmp_path, MESSAGES)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["meta"]["schema"] == oscrec.SCHEMA
    assert lines[-1]["end"]["message_count"] == 5
    msgs = [x for x in lines if "address" in x]
    assert [m["address"] for m in msgs] == ["/nc/event", "/nc/pose", "/nc/status", "/nc/cam/target", "/nc/blob"]
    assert msgs[2]["types"] == "Tifd"
    assert msgs[2]["args"] == [True, 17, 52.0, 1.25]
    assert 0.35 < msgs[4]["t"] - msgs[0]["t"] < 0.6


def test_record_filter(tmp_path):
    path = _record(tmp_path, MESSAGES, prefixes=["/nc/cam"])
    _, msgs = oscrec.load_recording(path)
    assert [m["address"] for m in msgs] == ["/nc/cam/target"]


def test_playback_sends_same_bytes_with_timing(tmp_path):
    path = _record(tmp_path, MESSAGES)
    port = _free_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(2.0)
    got: list[tuple[float, bytes]] = []

    def listen():
        for _ in MESSAGES:
            data, _ = sock.recvfrom(65536)
            got.append((time.monotonic(), data))

    th = threading.Thread(target=listen)
    th.start()
    assert oscrec.play_main([str(path), "--to", f"127.0.0.1:{port}", "--speed", "2"]) == 0
    th.join()
    sock.close()
    assert [d for _, d in got] == [d for _, d in MESSAGES]  # identical datagrams, types included
    span = got[-1][0] - got[0][0]
    assert 0.15 < span < 0.35  # 0.4 s recorded, at x2


def test_select_window_and_exclude():
    msgs = [{"t": float(i), "address": a, "args": []}
            for i, a in enumerate(["/nc/pose", "/nc/cam/pose", "/nc/event", "/nc/cam/event"])]
    assert [m["t"] for m in oscrec.select(msgs, 1.0, 2.0, [], [])] == [1.0, 2.0]
    assert [m["address"] for m in oscrec.select(msgs, 0.0, None, [], ["/nc/cam"])] == ["/nc/pose", "/nc/event"]
    assert [m["address"] for m in oscrec.select(msgs, 0.0, None, ["/nc/cam"], [])] == ["/nc/cam/pose", "/nc/cam/event"]


def test_reads_neurotheater_json(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"meta": {"recorded_at": "x"}, "messages": [
        {"t": 1700000000.5, "address": "/eeg/alpha", "args": [0.25]},
        {"t": 1700000000.0, "address": "/eeg/beta", "args": [1, "a"]},
    ]}))
    meta, msgs = oscrec.load_recording(path)
    assert meta["recorded_at"] == "x"
    assert [(m["t"], m["address"]) for m in msgs] == [(0.0, "/eeg/beta"), (0.5, "/eeg/alpha")]
    # No type tags: inferred from the values.
    msg = next(iter(OscPacket(oscrec.build_message(msgs[0])).messages)).message
    assert msg.params == [1, "a"]


def test_truncated_last_line_is_skipped(tmp_path):
    path = tmp_path / "cut.jsonl"
    path.write_text('{"meta":{}}\n{"t":0.0,"address":"/nc/a","args":[1],"types":"i"}\n{"t":0.1,"addr')
    _, msgs = oscrec.load_recording(path)
    assert len(msgs) == 1

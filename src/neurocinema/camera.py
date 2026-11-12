"""Camera input: the camera bench's /nc/cam/* in, re-published by the engine.

The camera bench (bench/camera) does its own pointing and dwell selection and
sends its messages to the engine (camera.listen_port, default 7004). The
engine is the only program TouchDesigner talks to, so it forwards them on its
own outputs, with these changes:

  - /nc/cam/heartbeat and the bench's camera_started / camera_stopped events
    are not forwarded; the engine sends /nc/cam/event camera_connected on the
    first message and camera_lost after camera_stopped or stale_after_s of
    silence, then /nc/cam/state idle and /nc/cam/target none none, so no stale
    highlight stays on screen;
  - while the phase is watching, selections are blocked as nods are:
    /nc/cam/selected is dropped, /nc/cam/event selected becomes
    /nc/cam/event selected_ignored watching, and /nc/cam/state selected is
    sent as active;
  - the bench's legacy /nc/cam/aim, /highlight and /select are dropped.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from .engine import Output

log = logging.getLogger(__name__)

# What the engine re-publishes (after the prefix /nc/cam/).
FORWARD = ("state", "target", "dwell", "selected", "pose", "event", "calib")


class CameraRelay:
    def __init__(self, out: Output, prefix: str, stale_after_s: float,
                 clock: Callable[[], float] = time.monotonic):
        self.out = out
        self.cam = prefix.rstrip("/") + "/cam/"
        self.stale_after_s = stale_after_s
        self.clock = clock
        self.connected = False
        self._last_msg = 0.0
        self._latest: dict[str, tuple] = {}  # last state / target sent, for snapshots
        self._queue: queue.SimpleQueue[tuple[str, tuple]] = queue.SimpleQueue()
        self._server: BlockingOSCUDPServer | None = None

    # --- input --------------------------------------------------------------------------------

    def listen(self, host: str, port: int) -> None:
        """Receive the bench's messages on host:port (raises OSError if the port is taken)."""
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.receive)
        self._server = BlockingOSCUDPServer((host, port), dispatcher)
        threading.Thread(target=self._server.serve_forever, name="osc-camera", daemon=True).start()
        host, port = self._server.server_address[:2]
        log.info("listening for the camera on %s:%d", host, port)

    def receive(self, address: str, *args: object) -> None:
        """Any thread: queue one message for the main loop."""
        self._queue.put((address, args))

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # --- main loop ----------------------------------------------------------------------------

    def poll(self, phase: str) -> None:
        """Forward what arrived since the last call; notice a camera gone quiet."""
        now = self.clock()
        while True:
            try:
                address, args = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle(address, args, phase, now)
        if self.connected and now - self._last_msg > self.stale_after_s:
            self._lose(f"nothing for {self.stale_after_s:g} s")

    def snapshot(self) -> None:
        """Re-send the camera's state and target (for /nc/cmd/snapshot)."""
        if self.connected:
            for name in ("state", "target"):
                if name in self._latest:
                    self._send(name, *self._latest[name])

    def _handle(self, address: str, args: tuple, phase: str, now: float) -> None:
        if not address.startswith(self.cam):
            log.debug("camera port: ignoring %s", address)
            return
        name = address[len(self.cam):]
        event = str(args[0]) if name == "event" and args else None
        if event == "camera_stopped":
            if self.connected:
                self._lose("camera stopped")
            return
        self._last_msg = now
        if not self.connected:
            self.connected = True
            log.info("camera connected")
            self._send("event", "camera_connected")
        if name not in FORWARD or event == "camera_started":
            return
        if phase == "watching":
            if name == "selected":
                return
            if event == "selected":
                self._send("event", "selected_ignored", "watching")
                return
            if name == "state" and args and args[0] == "selected":
                args = ("active",)
        self._send(name, *args)

    def _lose(self, why: str) -> None:
        self.connected = False
        log.info("camera lost (%s)", why)
        self._send("event", "camera_lost")
        self._send("state", "idle")
        self._send("target", "none", "none")

    def _send(self, name: str, *args: object) -> None:
        if name in ("state", "target"):
            self._latest[name] = args
        self.out.send(self.cam + name, *args)

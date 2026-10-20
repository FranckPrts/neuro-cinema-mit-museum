"""OSC over UDP: messages out to TouchDesigner, commands in from it."""

from __future__ import annotations

import logging
import queue
import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import UDPClient

from .config import parse_host_port

log = logging.getLogger(__name__)


class OscOutput:
    def __init__(self, destinations: list[str]):
        self._clients = []
        for dest in destinations:
            host, port = parse_host_port(dest)
            self._clients.append((dest, UDPClient(host, port)))
        self._last_error_log = 0.0

    def send(self, address: str, *args: object) -> None:
        builder = OscMessageBuilder(address=address)
        for a in args:
            builder.add_arg(a)
        msg = builder.build()
        for dest, client in self._clients:
            try:
                client.send(msg)
            except OSError as e:
                # UDP to a closed local port can raise; never let that stop the engine.
                now = time.monotonic()
                if now - self._last_error_log > 10:
                    self._last_error_log = now
                    log.warning("OSC send to %s failed: %s", dest, e)


class OscCommandServer:
    """Listens for <prefix>/cmd/<name> [args...] and queues (name, args)."""

    def __init__(self, host: str, port: int, prefix: str):
        self._prefix = prefix.rstrip("/") + "/cmd/"
        self._queue: queue.Queue[tuple[str, tuple]] = queue.Queue()
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._handle)
        self._server = BlockingOSCUDPServer((host, port), dispatcher)
        self._thread = threading.Thread(target=self._server.serve_forever, name="osc-in", daemon=True)

    def start(self) -> None:
        self._thread.start()
        host, port = self._server.server_address[:2]
        log.info("listening for commands on %s:%d", host, port)

    def _handle(self, address: str, *args: object) -> None:
        if address.startswith(self._prefix):
            self._queue.put((address[len(self._prefix):], args))
        else:
            log.debug("ignoring OSC %s", address)

    def drain(self) -> list[tuple[str, tuple]]:
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

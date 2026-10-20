"""End to end over real UDP sockets: simulated head -> App -> OSC, and commands back in."""

import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from neurocinema.app import App
from neurocinema.config import Config
from neurocinema.engine import State
from neurocinema.osc import OscCommandServer, OscOutput
from neurocinema.sim.head import VirtualHead
from neurocinema.sources.sim import SimSource


class Receiver:
    def __init__(self):
        self.messages = []
        self._lock = threading.Lock()
        d = Dispatcher()
        d.set_default_handler(self._handle)
        self.server = ThreadingOSCUDPServer(("127.0.0.1", 0), d)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handle(self, address, *args):
        with self._lock:
            self.messages.append((address, args))

    def has(self, address, args=None):
        with self._lock:
            return any(a == address and (args is None or tuple(x) == tuple(args)) for a, x in self.messages)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_engine_over_udp():
    rx = Receiver()
    cfg = Config()
    cfg.source.kind = "sim"
    cfg.osc.destinations = [f"127.0.0.1:{rx.port}"]
    commands = OscCommandServer("127.0.0.1", 0, cfg.osc.prefix)
    cmd_port = commands._server.server_address[1]
    source = SimSource(VirtualHead(seed=1))
    app = App(cfg, source, OscOutput(cfg.osc.destinations), commands=commands)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    tx = SimpleUDPClient("127.0.0.1", cmd_port)
    try:
        assert wait_for(lambda: rx.has("/nc/heartbeat"))
        assert wait_for(lambda: rx.has("/nc/status"))
        assert wait_for(lambda: app.engine.state == State.IDLE)

        tx.send_message("/nc/cmd/ping", [42])
        assert wait_for(lambda: rx.has("/nc/pong", [42]))

        tx.send_message("/nc/cmd/start", [])
        assert wait_for(lambda: app.engine.state == State.ACTIVE, timeout=8)
        assert rx.has("/nc/event", ["calibrated"])

        tx.send_message("/nc/cmd/phase", ["watching"])
        assert wait_for(lambda: rx.has("/nc/phase", ["watching"]))

        source.streaming = False
        assert wait_for(lambda: rx.has("/nc/state", ["disconnected"]), timeout=4)
        source.streaming = True
        assert wait_for(lambda: app.engine.state == State.CALIBRATING)
    finally:
        app.stop_event.set()
        thread.join(timeout=5)
        stopped = wait_for(lambda: rx.has("/nc/event", ["engine_stopped"]), timeout=2)
        rx.close()
    assert stopped

"""The Bluetooth side of the in-process headset connection.

One background thread (`BleWorker`) owns the Bluetooth link. It only does what
it is told (scan, connect, disconnect) and reports back through queues; every
decision is made by `headsets.HeadsetManager` on the main loop.

The real link goes through muselsl's `Muse` class (bleak underneath), without
`muselsl stream` and without LSL. `FakeDriver` stands in for it in tests and in
`nc-engine --sim-headsets`.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

IMU_RATE_HZ = 52.0

# Callbacks a driver calls from inside connect()/pump().
OnImu = Callable[[str, float, float, float, float], None]  # kind, t, x, y, z
OnBattery = Callable[[float, float], None]  # percent, t


class ConnectError(RuntimeError):
    pass


@dataclass(frozen=True)
class Seen:
    """A Muse found by a scan."""

    name: str
    address: str


class BleDriver(Protocol):
    def scan(self, timeout_s: float) -> list[Seen]:
        """Muses currently advertising (i.e. switched on and not connected)."""

    def connect(self, address: str, name: str, on_imu: OnImu, on_battery: OnBattery) -> None:
        """Connect and start streaming ACC + GYRO. Raises ConnectError."""

    def pump(self, seconds: float) -> None:
        """Let the Bluetooth stack deliver data for this long (callbacks run here)."""

    def disconnect(self) -> None: ...


# --- worker ---------------------------------------------------------------------


@dataclass(frozen=True)
class ImuReading:
    label: str
    kind: str
    t: float
    x: float
    y: float
    z: float


class BleWorker:
    """Runs a driver on its own thread. Thread-safe API for the main loop.

    Events (drained with `events()`), as tuples:
      ("scan", [Seen, ...])
      ("connected", label)
      ("connect_failed", label, message)
      ("disconnected", label)
      ("battery", label, percent, t)
    """

    def __init__(self, driver: BleDriver, start_thread: bool = True):
        self.driver = driver
        self._commands: queue.Queue[tuple] = queue.Queue()
        self._events: queue.Queue[tuple] = queue.Queue()
        self._readings: deque[ImuReading] = deque(maxlen=5000)
        self._label: str | None = None  # connected headset
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="ble-worker", daemon=True)
            self._thread.start()

    # --- main-loop API ---------------------------------------------------------------

    def request_scan(self, timeout_s: float) -> None:
        self._busy.set()
        self._commands.put(("scan", timeout_s))

    def request_connect(self, label: str, name: str, address: str | None) -> None:
        self._busy.set()
        self._commands.put(("connect", label, name, address))

    def request_disconnect(self) -> None:
        self._commands.put(("disconnect",))

    @property
    def busy(self) -> bool:
        """A scan or connect is queued or running."""
        return self._busy.is_set()

    def events(self) -> list[tuple]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def readings(self) -> list[ImuReading]:
        out = []
        while self._readings:
            out.append(self._readings.popleft())
        return out

    def close(self) -> None:
        self._stop.set()
        self._commands.put(("stop",))
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        else:
            self._drop()

    # --- worker thread -----------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step(wait_s=0.0 if self._label else 0.05)
        self._drop()

    def step(self, wait_s: float = 0.0) -> None:
        """One command, or one pump of the link. Called by the thread (or by tests)."""
        try:
            cmd = self._commands.get(timeout=wait_s) if wait_s > 0 else self._commands.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd is not None:
            try:
                self._handle(cmd)
            except Exception:  # noqa: BLE001 - the worker must never die
                log.exception("Bluetooth command %s failed", cmd[0])
            finally:
                if self._commands.empty():
                    self._busy.clear()
            return
        if self._label is not None:
            try:
                self.driver.pump(0.05)
            except Exception:  # noqa: BLE001
                log.exception("Bluetooth link to headset %s failed", self._label)
                self._drop()

    def _handle(self, cmd: tuple) -> None:
        kind = cmd[0]
        if kind == "stop":
            return
        if kind == "scan":
            if self._label is not None:
                return  # never scan while streaming
            self._events.put(("scan", self.driver.scan(cmd[1])))
        elif kind == "connect":
            _, label, name, address = cmd
            self._drop()
            if not address:
                found = [s for s in self.driver.scan(5.0) if s.name == name]
                if not found:
                    self._events.put(("connect_failed", label, f"{name} not found; is it switched on?"))
                    return
                address = found[0].address
            try:
                self.driver.connect(address, name, self._imu_cb(label), self._battery_cb(label))
            except Exception as e:  # noqa: BLE001
                log.warning("could not connect to headset %s (%s): %s", label, name, e)
                self._events.put(("connect_failed", label, str(e) or type(e).__name__))
                return
            self._label = label
            self._events.put(("connected", label))
        elif kind == "disconnect":
            self._drop()

    def _drop(self) -> None:
        if self._label is None:
            return
        label, self._label = self._label, None
        try:
            self.driver.disconnect()
        except Exception:  # noqa: BLE001 - the link may already be gone
            log.debug("disconnect from %s failed", label, exc_info=True)
        self._events.put(("disconnected", label))

    def _imu_cb(self, label: str) -> OnImu:
        def cb(kind: str, t: float, x: float, y: float, z: float) -> None:
            self._readings.append(ImuReading(label, kind, t, x, y, z))
        return cb

    def _battery_cb(self, label: str) -> OnBattery:
        def cb(percent: float, t: float) -> None:
            self._events.put(("battery", label, percent, t))
        return cb


# --- real headset ------------------------------------------------------------------------


class MuselslDriver:
    """Muse 2 through muselsl's `Muse` class and its bleak backend.

    All calls happen on the worker thread, which also runs muselsl's asyncio
    loop (in `pump`), so Bluetooth callbacks arrive on that thread.
    """

    # Battery normally arrives as telemetry. If none has come this long after
    # connecting, ask the headset for its status once (it reports "bp").
    ASK_STATUS_AFTER_S = 60.0

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        from muselsl import backends
        from muselsl.muse import Muse

        self._backends = backends
        self._Muse = Muse
        self.clock = clock
        self._muse = None
        self._connected_at = 0.0
        self._got_battery = False
        self._asked_status = False

    def scan(self, timeout_s: float) -> list[Seen]:
        devices = self._backends.BleakBackend().scan(timeout=timeout_s)
        return [Seen(d["name"], d["address"]) for d in devices if d["name"] and "Muse" in d["name"]]

    def connect(self, address: str, name: str, on_imu: OnImu, on_battery: OnBattery) -> None:
        def imu(kind: str):
            def cb(samples, timestamps):
                # muselsl stamps all three samples of a packet with the arrival
                # time; spread them back over the packet's 3 / 52 s.
                t_last = timestamps[-1]
                n = samples.shape[1]
                for i in range(n):
                    x, y, z = (float(v) for v in samples[:, i])
                    on_imu(kind, t_last - (n - 1 - i) / IMU_RATE_HZ, x, y, z)
            return cb

        def telemetry(t, battery, _fuel_gauge, _adc_volt, _temperature):
            self._got_battery = True
            on_battery(float(battery), t)

        def control(message):
            try:
                bp = json.loads(message).get("bp")
            except (ValueError, AttributeError):
                return
            if bp is not None:
                self._got_battery = True
                on_battery(float(bp), self.clock())

        muse = self._Muse(
            address,
            callback_acc=imu("acc"),
            callback_gyro=imu("gyro"),
            callback_telemetry=telemetry,
            callback_control=control,
            backend="bleak",
            time_func=self.clock,
            name=name,
        )
        if not muse.connect(retries=0):
            raise ConnectError(f"could not connect to {name}")
        muse.start()
        self._muse = muse
        self._connected_at = self.clock()
        self._got_battery = False
        self._asked_status = False

    def pump(self, seconds: float) -> None:
        if self._muse is None:
            time.sleep(seconds)
            return
        self._muse.adapter.pump(seconds)
        if (not self._got_battery and not self._asked_status
                and self.clock() - self._connected_at > self.ASK_STATUS_AFTER_S):
            self._asked_status = True
            log.info("no battery telemetry yet; asking the headset for its status")
            self._muse.ask_control()
            self._muse.resume()  # make sure streaming continues after the query

    def disconnect(self) -> None:
        muse, self._muse = self._muse, None
        if muse is None:
            return
        try:
            muse.stop()
        finally:
            muse.disconnect()


# --- simulated headsets --------------------------------------------------------------------


class FakeDriver:
    """Simulated Muses for tests and `--sim-headsets`.

    `headsets` maps BLE name -> starting battery percent. A connected headset
    streams a virtual head at 52 Hz and drains its battery `drain_per_min`
    percent per minute. `lifted_after_s` simulates staff picking it up.
    """

    def __init__(
        self,
        headsets: dict[str, float],
        clock: Callable[[], float] = time.monotonic,
        realtime: bool = True,
        drain_per_min: float = 0.33,
        lifted_after_s: float | None = 3.0,
        battery_every_s: float = 10.0,
    ):
        from .sim.head import VirtualHead

        self.battery = dict(headsets)
        self.switched_on = set(headsets)
        self.clock = clock
        self.realtime = realtime
        self.drain_per_min = drain_per_min
        self.lifted_after_s = lifted_after_s
        self.battery_every_s = battery_every_s
        self.fail_connect: set[str] = set()
        self._head_cls = VirtualHead
        self._name: str | None = None
        self._head = None
        self._t_next = 0.0
        self._connected_at = 0.0
        self._next_battery = 0.0
        self._on_imu: OnImu | None = None
        self._on_battery: OnBattery | None = None
        self.link_up = True  # set False to simulate the link dying

    def address(self, name: str) -> str:
        return f"SIM-{name}"

    def scan(self, timeout_s: float) -> list[Seen]:
        if self.realtime:
            time.sleep(min(timeout_s, 0.5))
        return [Seen(n, self.address(n)) for n in sorted(self.switched_on) if n != self._name]

    def connect(self, address: str, name: str, on_imu: OnImu, on_battery: OnBattery) -> None:
        if name not in self.switched_on or name in self.fail_connect or address != self.address(name):
            raise ConnectError(f"{name} did not answer")
        if self.battery.get(name, 0.0) <= 0.0:
            raise ConnectError(f"{name} battery is empty")
        self._name = name
        self._head = self._head_cls(seed=hash(name) & 0xFFFF, worn=False)
        self._on_imu, self._on_battery = on_imu, on_battery
        now = self.clock()
        self._connected_at = now
        self._t_next = now
        self._next_battery = now
        self._last_drain = now
        self.link_up = True
        self._lifted = False

    def pump(self, seconds: float) -> None:
        if self.realtime:
            time.sleep(seconds)
        if self._name is None or not self.link_up:
            return
        now = self.clock()
        if (not self._lifted and self.lifted_after_s is not None
                and now - self._connected_at >= self.lifted_after_s):
            self._lifted = True
            self._head.set_worn(True, now)  # the put-on motion reads as a lift
        while self._t_next <= now:
            acc, gyro = self._head.step(self._t_next)
            self._on_imu("acc", self._t_next, *map(float, acc))
            self._on_imu("gyro", self._t_next, *map(float, gyro))
            self._t_next += 1.0 / IMU_RATE_HZ
        if now >= self._next_battery:
            drained = self.drain_per_min * (now - self._last_drain) / 60.0
            self._last_drain = now
            self.battery[self._name] = max(0.0, self.battery[self._name] - drained)
            self._on_battery(round(self.battery[self._name], 1), now)
            self._next_battery = now + self.battery_every_s
            if self.battery[self._name] <= 0.0:
                self.link_up = False  # dead headset: data stops

    def disconnect(self) -> None:
        self._name = None
        self._head = None

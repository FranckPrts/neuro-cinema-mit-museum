"""Drive the engine with a virtual head on a simulated clock."""

from __future__ import annotations

from dataclasses import dataclass, field

from neurocinema.config import Config
from neurocinema.engine import Engine, State
from neurocinema.samples import ImuSample
from neurocinema.sim.head import RATE_HZ, VirtualHead


@dataclass
class RecordingOutput:
    messages: list[tuple[float, str, tuple]] = field(default_factory=list)
    clock: float = 0.0

    def send(self, address: str, *args: object) -> None:
        self.messages.append((self.clock, address, args))

    def events(self, name: str | None = None) -> list[tuple]:
        out = [args for _, a, args in self.messages if a.endswith("/event")]
        return [e for e in out if name is None or e[0] == name]

    def last(self, address_suffix: str) -> tuple | None:
        for _, a, args in reversed(self.messages):
            if a.endswith(address_suffix):
                return args
        return None


class Rig:
    def __init__(self, cfg: Config | None = None, head: VirtualHead | None = None, seed: int = 0):
        self.cfg = cfg or Config()
        self.out = RecordingOutput()
        self.engine = Engine(self.cfg, self.out)
        self.head = head or VirtualHead(seed=seed)
        self.t = 0.0
        self.streaming = True

    def run(self, seconds: float) -> None:
        dt = 1.0 / RATE_HZ
        end = self.t + seconds
        while self.t < end:
            self.out.clock = self.t
            if self.streaming:
                if not self.engine.connected:
                    self.engine.on_connection(True, self.t)
                acc, gyro = self.head.step(self.t)
                self.engine.process(ImuSample(self.t, "acc", *acc))
                self.engine.process(ImuSample(self.t, "gyro", *gyro))
            self.engine.tick(self.t)
            self.t += dt

    def drop(self) -> None:
        self.streaming = False
        self.engine.on_connection(False, self.t)

    def start_session(self) -> None:
        """Connect, wake from IDLE by putting the headset on, calibrate."""
        self.run(0.5)
        assert self.engine.state == State.IDLE
        self.head.set_worn(False, self.t)
        self.run(0.2)
        self.head.set_worn(True, self.t)
        self.run(1.6)  # put-on motion
        assert self.engine.state == State.CALIBRATING
        self.run(3.5)
        assert self.engine.state == State.ACTIVE, self.engine.state

    @property
    def state(self) -> State:
        return self.engine.state

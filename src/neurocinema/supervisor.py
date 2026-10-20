"""Keeps `muselsl stream` running.

    STOPPED ─start─► STARTING ──data──► STREAMING
                        │                   │
          no data in grace / exited   stalled / exited
                        ▼                   ▼
                     BACKOFF ◄──────────────┘   (2, 4, 8 ... 60 s)
                        │
                        └──► STARTING

Every `escalate_after_failures` consecutive failures it reports
`stream_unrecoverable` (so TouchDesigner can ask for staff) and runs the
optional escalation command, e.g. a Bluetooth restart. It keeps retrying
regardless: the headset may simply be switched off.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import psutil

from .config import MuseConfig

log = logging.getLogger(__name__)
child_log = logging.getLogger("muselsl")


class StreamState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    STREAMING = "streaming"
    BACKOFF = "backoff"


class MuseStreamSupervisor:
    def __init__(
        self,
        cfg: MuseConfig,
        on_restart: Callable[[], None],
        on_error: Callable[[str, str], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self._on_restart = on_restart
        self._on_error = on_error
        self._clock = clock
        self.state = StreamState.STOPPED
        self.failures = 0
        self.last_reason = ""
        self._proc: subprocess.Popen | None = None
        self._state_since = clock()
        self._next_start = 0.0
        self._pidfile = Path(cfg.pidfile)

    # --- public ---------------------------------------------------------------

    def start(self) -> None:
        self._spawn()

    def tick(self, receiving: bool) -> None:
        now = self._clock()
        if self.state == StreamState.BACKOFF:
            if now >= self._next_start:
                self._spawn()
            return
        if self.state not in (StreamState.STARTING, StreamState.STREAMING):
            return

        code = self._proc.poll() if self._proc else -1
        if code is not None:
            self._fail(f"muselsl exited with code {code}")
            return
        if self.state == StreamState.STARTING:
            if receiving:
                self._set(StreamState.STREAMING)
                if self.failures:
                    log.info("stream recovered after %d failure(s)", self.failures)
                    self._on_error("ok", "")
                self.failures = 0
            elif now - self._state_since > self.cfg.startup_grace_s:
                self._fail(f"no data within {self.cfg.startup_grace_s:.0f} s (headset off or out of range?)")
        elif not receiving:
            self._fail("stream stalled")

    def restart(self, reason: str = "requested") -> None:
        """Restart now, without counting as a failure."""
        log.info("restarting muselsl (%s)", reason)
        self._kill()
        self._on_restart()
        self._set(StreamState.BACKOFF)
        self._next_start = self._clock()

    def stop(self) -> None:
        self._kill()
        self._set(StreamState.STOPPED)

    @property
    def pid(self) -> int:
        return self._proc.pid if self._proc and self._proc.poll() is None else 0

    # --- internals ------------------------------------------------------------

    def _set(self, state: StreamState) -> None:
        if state != self.state:
            log.info("stream %s -> %s", self.state.value, state.value)
        self.state = state
        self._state_since = self._clock()

    def command(self) -> list[str]:
        cfg = self.cfg
        args = ["--acc", "--gyro", "--lsltime", "--retries", "1", "--backend", cfg.backend]
        if not cfg.stream_eeg:
            args.append("--disable-eeg")
        if cfg.address:
            args += ["--address", cfg.address]
        if cfg.name:
            args += ["--name", cfg.name]
        args += cfg.extra_args
        return [sys.executable, "-m", "neurocinema.musestream", "--parent-pid", str(os.getpid()), "--", *args]

    def _spawn(self) -> None:
        self._kill_orphan()
        cmd = self.command()
        log.info("starting muselsl: %s", " ".join(cmd[3:]))
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # our Ctrl-C handler stops it, not the terminal
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                **kwargs,
            )
        except OSError as e:
            self._proc = None
            self._fail(f"could not launch muselsl: {e}")
            return
        self._write_pidfile(self._proc.pid)
        threading.Thread(target=self._pump, args=(self._proc,), name="muselsl-log", daemon=True).start()
        self._set(StreamState.STARTING)

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                child_log.info(line)

    def _fail(self, reason: str) -> None:
        self._kill()
        self.failures += 1
        self.last_reason = reason
        delay = min(self.cfg.backoff_max_s, self.cfg.backoff_initial_s * 2 ** (self.failures - 1))
        log.warning("stream failure %d: %s; retrying in %.0f s", self.failures, reason, delay)
        self._on_restart()
        self._set(StreamState.BACKOFF)
        self._next_start = self._clock() + delay
        n = self.cfg.escalate_after_failures
        if n > 0 and self.failures % n == 0:
            self._on_error("stream_unrecoverable", f"{self.failures} consecutive failures; last: {reason}")
            self._escalate()

    def _escalate(self) -> None:
        cmd = self.cfg.escalation_command
        if not cmd:
            return

        def run() -> None:
            log.warning("running escalation command: %s", " ".join(cmd))
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                log.warning("escalation command exited %d %s", r.returncode, (r.stdout + r.stderr).strip())
            except (OSError, subprocess.SubprocessError) as e:
                log.error("escalation command failed: %s", e)

        threading.Thread(target=run, name="escalation", daemon=True).start()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            _terminate(proc.pid, proc)
        self._remove_pidfile()

    # A muselsl left over from a previous engine run would hold the headset.
    def _kill_orphan(self) -> None:
        try:
            pid = int(self._pidfile.read_text().strip())
        except (OSError, ValueError):
            return
        try:
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline())
        except (psutil.Error, OSError):
            self._remove_pidfile()
            return
        if "neurocinema.musestream" in cmdline or "muselsl" in cmdline:
            log.warning("stopping leftover muselsl (pid %d)", pid)
            _terminate(pid, None)
        self._remove_pidfile()

    def _write_pidfile(self, pid: int) -> None:
        try:
            self._pidfile.parent.mkdir(parents=True, exist_ok=True)
            self._pidfile.write_text(str(pid))
        except OSError as e:
            log.warning("could not write %s: %s", self._pidfile, e)

    def _remove_pidfile(self) -> None:
        try:
            self._pidfile.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("could not remove %s: %s", self._pidfile, e)


def _terminate(pid: int, proc: subprocess.Popen | None, timeout: float = 3.0) -> None:
    """Ask nicely (SIGINT lets muselsl disconnect cleanly), then force."""
    try:
        if os.name == "nt":
            if proc is not None:
                proc.terminate()
            else:
                psutil.Process(pid).terminate()
        else:
            os.kill(pid, signal.SIGINT)
        if proc is not None:
            proc.wait(timeout=timeout)
        else:
            psutil.Process(pid).wait(timeout=timeout)
    except (ProcessLookupError, psutil.NoSuchProcess):
        return
    except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
        log.warning("muselsl (pid %d) did not stop; killing it", pid)
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass
    except (OSError, psutil.Error) as e:
        log.warning("could not stop pid %d: %s", pid, e)

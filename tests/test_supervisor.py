"""The supervisor with a stand-in child process instead of muselsl."""

import os
import subprocess
import sys
import time

import psutil
import pytest

from neurocinema.config import MuseConfig
from neurocinema.supervisor import MuseStreamSupervisor, StreamState

pytestmark = pytest.mark.skipif(os.name == "nt", reason="uses POSIX signals in the stand-in child")

SLEEPER = "import time\nwhile True: time.sleep(0.1)"
EXITER = "import sys; sys.exit(3)"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Harness:
    def __init__(self, tmp_path, script=SLEEPER, **overrides):
        self.cfg = MuseConfig(pidfile=str(tmp_path / "muselsl.pid"), **overrides)
        self.clock = FakeClock()
        self.restarts = 0
        self.errors = []
        self.script = script
        self.sup = MuseStreamSupervisor(self.cfg, self._restart, self._error, clock=self.clock)
        self.sup.command = lambda: [sys.executable, "-c", self.script]

    def _restart(self):
        self.restarts += 1

    def _error(self, code, msg):
        self.errors.append(code)

    def wait_exit(self, timeout=3.0):
        proc = self.sup._proc
        if proc is not None:
            proc.wait(timeout=timeout)


def gone(pid):
    return not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_reaches_streaming_when_data_arrives(tmp_path):
    h = Harness(tmp_path)
    h.sup.start()
    assert h.sup.state == StreamState.STARTING
    h.sup.tick(receiving=True)
    assert h.sup.state == StreamState.STREAMING
    assert h.sup.pid > 0
    h.sup.stop()


def test_child_exit_backs_off_exponentially_then_retries(tmp_path):
    h = Harness(tmp_path, script=EXITER, backoff_initial_s=2, backoff_max_s=60)
    h.sup.start()
    h.wait_exit()
    h.sup.tick(receiving=False)
    assert h.sup.state == StreamState.BACKOFF
    assert h.sup.failures == 1
    assert h.restarts == 1

    h.clock.t += 1.9
    h.sup.tick(receiving=False)
    assert h.sup.state == StreamState.BACKOFF
    h.clock.t += 0.2
    h.sup.tick(receiving=False)
    assert h.sup.state == StreamState.STARTING

    h.wait_exit()
    h.sup.tick(receiving=False)
    assert h.sup.failures == 2
    assert h.sup._next_start - h.clock.t == pytest.approx(4.0)
    h.sup.stop()


def test_no_data_within_grace_is_a_failure(tmp_path):
    h = Harness(tmp_path, startup_grace_s=10)
    h.sup.start()
    pid = h.sup.pid
    h.clock.t += 11
    h.sup.tick(receiving=False)
    assert h.sup.state == StreamState.BACKOFF
    assert gone(pid)


def test_stall_while_streaming_kills_and_restarts(tmp_path):
    h = Harness(tmp_path)
    h.sup.start()
    h.sup.tick(receiving=True)
    pid = h.sup.pid
    h.sup.tick(receiving=False)
    assert h.sup.state == StreamState.BACKOFF
    assert h.sup.pid == 0
    assert gone(pid)


def test_recovery_after_failures_reports_ok(tmp_path):
    h = Harness(tmp_path, script=EXITER, backoff_initial_s=1)
    h.sup.start()
    h.wait_exit()
    h.sup.tick(receiving=False)
    h.script = SLEEPER
    h.clock.t += 1.1
    h.sup.tick(receiving=False)
    h.sup.tick(receiving=True)
    assert h.sup.state == StreamState.STREAMING
    assert h.sup.failures == 0
    assert h.errors == ["ok"]
    h.sup.stop()


def test_escalation_after_repeated_failures(tmp_path):
    marker = tmp_path / "escalated"
    h = Harness(
        tmp_path, script=EXITER, backoff_initial_s=0.01, backoff_max_s=0.01,
        escalate_after_failures=3,
        escalation_command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"],
    )
    h.sup.start()
    for _ in range(3):
        h.wait_exit()
        h.sup.tick(receiving=False)  # failure
        h.clock.t += 1
        h.sup.tick(receiving=False)  # respawn
    assert h.errors == ["stream_unrecoverable"]
    deadline = time.time() + 5
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.exists()
    h.sup.stop()


def test_manual_restart_does_not_count_as_failure(tmp_path):
    h = Harness(tmp_path)
    h.sup.start()
    h.sup.tick(receiving=True)
    h.sup.restart("test")
    assert h.sup.failures == 0
    h.sup.tick(receiving=False)
    assert h.sup.state == StreamState.STARTING
    h.sup.stop()


def test_leftover_muselsl_from_a_crashed_engine_is_stopped(tmp_path):
    # Looks like an orphaned muselsl: its command line mentions muselsl.
    orphan = subprocess.Popen([sys.executable, "-c", "import time  # muselsl stream\nwhile True: time.sleep(0.1)"])
    try:
        (tmp_path / "muselsl.pid").write_text(str(orphan.pid))
        h = Harness(tmp_path)
        h.sup.start()
        orphan.wait(timeout=5)
        assert orphan.returncode is not None
        h.sup.stop()
    finally:
        if orphan.poll() is None:
            orphan.kill()


def test_unrelated_process_in_stale_pidfile_is_left_alone(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", SLEEPER])
    try:
        (tmp_path / "muselsl.pid").write_text(str(other.pid))
        h = Harness(tmp_path)
        h.sup.start()
        time.sleep(0.2)
        assert other.poll() is None
        h.sup.stop()
    finally:
        other.kill()


def test_wrapper_exits_when_parent_dies():
    parent = subprocess.Popen([sys.executable, "-c", "pass"])
    parent.wait()
    child = subprocess.run(
        [sys.executable, "-c", f"from neurocinema.musestream import _watch_parent; _watch_parent({parent.pid})"],
        timeout=10,
    )
    assert child.returncode == 0

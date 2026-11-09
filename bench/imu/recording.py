"""The bench recording format (IMU only, no EEG). One folder per session:

    imu.csv      t,kind,x,y,z           one row per ImuSample (s, "acc" g | "gyro" deg/s)
    events.csv   t,kind,name,column,row,yaw_deg,pitch_deg,text
                 kind "phase" (name = stand, put_on, targets, gestures, free, off, end),
                 "prompt" (name = face, nod, shake; with the prompted target and
                 its ground-truth angles), or "status" (stream_lost / stream_back)
    meta.json    participant code, clocks at start, headset id, protocol settings

All times are on the source clock (LSL local_clock), exactly as the engine
sees them. `meta.json` also stores time.time() and time.monotonic() at the
start, to line up other recordings (e.g. the camera bench log).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from neurocinema.fusion import GYRO_CLIP_DPS
from neurocinema.samples import ImuSample

IMU_FIELDS = ("t", "kind", "x", "y", "z")
EVENT_FIELDS = ("t", "kind", "name", "column", "row", "yaw_deg", "pitch_deg", "text")
NOMINAL_RATE_HZ = 52.0


@dataclass
class Event:
    t: float
    kind: str
    name: str
    column: str = ""
    row: str = ""
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    text: str = ""


# --- writing -------------------------------------------------------------------------


class ImuWriter:
    def __init__(self, path: Path):
        self._f = open(path, "w", newline="")  # noqa: SIM115 - kept open, closed by close()
        self._w = csv.writer(self._f)
        self._w.writerow(IMU_FIELDS)
        self.counts = {"acc": 0, "gyro": 0}

    def write(self, s: ImuSample) -> None:
        self._w.writerow((repr(s.t), s.kind, repr(s.x), repr(s.y), repr(s.z)))
        self.counts[s.kind] += 1

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class EventWriter:
    def __init__(self, path: Path):
        self._f = open(path, "w", newline="")  # noqa: SIM115 - kept open, closed by close()
        self._w = csv.DictWriter(self._f, EVENT_FIELDS)
        self._w.writeheader()

    def write(self, e: Event) -> None:
        row = asdict(e)
        row["t"] = repr(e.t)
        for k in ("yaw_deg", "pitch_deg"):
            row[k] = "" if row[k] is None else row[k]
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# --- reading -------------------------------------------------------------------------


def read_imu(path: Path) -> list[ImuSample]:
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if tuple(header) != IMU_FIELDS:
            raise ValueError(f"{path}: expected columns {IMU_FIELDS}, got {header}")
        return [ImuSample(float(t), kind, float(x), float(y), float(z)) for t, kind, x, y, z in r]


def read_events(path: Path) -> list[Event]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Event(
                t=float(row["t"]), kind=row["kind"], name=row["name"],
                column=row["column"], row=row["row"],
                yaw_deg=float(row["yaw_deg"]) if row["yaw_deg"] else None,
                pitch_deg=float(row["pitch_deg"]) if row["pitch_deg"] else None,
                text=row["text"],
            ))
    return out


@dataclass
class Session:
    folder: Path
    samples: list[ImuSample]
    events: list[Event]
    meta: dict


def load_session(folder: Path) -> Session:
    folder = Path(folder)
    if folder.is_file():  # imu.csv given directly
        folder = folder.parent
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    events_path = folder / "events.csv"
    events = read_events(events_path) if events_path.exists() else []
    return Session(folder, read_imu(folder / "imu.csv"), events, meta)


# --- timestamps ----------------------------------------------------------------------


@dataclass
class TimingReport:
    duplicates: dict[str, int]  # samples whose time did not increase (the engine drops them)
    rate_hz: dict[str, float]  # samples / span
    gaps: dict[str, int]  # intervals longer than 0.1 s
    max_gap_s: dict[str, float]


def timing(samples: list[ImuSample]) -> TimingReport:
    rep = TimingReport({}, {}, {}, {})
    for kind in ("acc", "gyro"):
        ts = [s.t for s in samples if s.kind == kind]
        rep.duplicates[kind] = sum(1 for a, b in pairwise(ts) if b <= a)
        span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        rep.rate_hz[kind] = (len(ts) - 1) / span if span > 0 else 0.0
        diffs = [b - a for a, b in pairwise(ts)]
        rep.gaps[kind] = sum(1 for d in diffs if d > 0.1)
        rep.max_gap_s[kind] = max(diffs, default=0.0)
    return rep


def clipped(samples: list[ImuSample], limit_dps: float = GYRO_CLIP_DPS) -> list[float]:
    """Times of gyro samples at the gyro's full scale on some axis."""
    return [s.t for s in samples
            if s.kind == "gyro" and max(abs(s.x), abs(s.y), abs(s.z)) >= limit_dps]


def retime(samples: list[ImuSample], rate_hz: float = NOMINAL_RATE_HZ) -> tuple[list[ImuSample], dict[str, int]]:
    """Spread samples that share a timestamp back over 1/rate steps.

    muselsl stamps the 3 samples of a Muse IMU packet with the packet's arrival
    time, and the engine drops samples whose time does not increase. This is
    what `ble.MuselslDriver` does for the in-process source. A packet that would
    overlap the previous one (packets arriving in a burst) is spread evenly over
    the time since the previous one instead. Returns the new samples (sorted by
    time) and, per kind, how many samples were in such bursts.
    """
    out: list[ImuSample] = []
    squeezed = {"acc": 0, "gyro": 0}
    for kind in ("acc", "gyro"):
        ks = [s for s in samples if s.kind == kind]
        i = 0
        prev_t = -float("inf")
        while i < len(ks):
            j = i
            while j + 1 < len(ks) and ks[j + 1].t == ks[i].t:
                j += 1
            n = j - i + 1
            t_last = ks[j].t
            if t_last - (n - 1) / rate_hz > prev_t:
                times = [t_last - (n - 1 - k) / rate_hz for k in range(n)]
            else:
                step = (t_last - prev_t) / n if t_last > prev_t else 1e-4
                times = [prev_t + (k + 1) * step for k in range(n)]
                squeezed[kind] += n
            for k, t in enumerate(times):
                s = ks[i + k]
                out.append(ImuSample(t, s.kind, s.x, s.y, s.z))
            prev_t = times[-1]
            i = j + 1
    out.sort(key=lambda s: s.t)
    return out, squeezed

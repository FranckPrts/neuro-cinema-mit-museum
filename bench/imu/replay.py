"""Replay bench recordings through the engine with several orientation backends.

    python bench/imu/replay.py bench/imu/recordings/P01_*            # default variants
    python bench/imu/replay.py REC... --variant vqf --variant vqf:vqf_tau_acc_s=2.0
    python bench/imu/replay.py REC... --prompts prompts.csv          # one row per prompt
    python bench/imu/replay.py REC... --report docs/imu-backend-results.md   # results note

Each variant runs the full engine (fusion, gestures, targets, state machine,
re-anchoring) on the recorded samples, with a stub output, exactly as
nc-engine would have. Default variants:

  baseline       complementary filter without the idle bias (before step 1)
  complementary  the default engine
  vqf            VQF for "up" and the gyro bias
  vqf-stillness  VQF for "up", the complementary filter's bias logic

A variant is NAME or NAME:key=value,key=value with [imu] keys (or dotted keys
such as gestures.nod.peak_dps). Metrics, per variant:

  yaw / pitch   |error| at "face" prompts (median over 2.5-4.5 s after the
                prompt), before any re-anchoring: median and 95th percentile
  target        share of face prompts where the engine's target was the prompted one
  check yaw     |yaw error| facing the middle after a minute of free looking
  nods, shakes  detected in their prompt window / prompted; selections right / wrong
  false         nods or shakes outside gesture prompts (targets and free phases)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from recording import Event, Session, clipped, load_session, retime, timing  # noqa: E402

from neurocinema.config import Config, ConfigError, _merge, load_config  # noqa: E402
from neurocinema.engine import Engine  # noqa: E402

DEFAULT_VARIANTS: dict[str, dict] = {
    "baseline": {"imu": {"backend": "complementary", "idle_bias_max_age_s": 0.0}},
    "complementary": {"imu": {"backend": "complementary"}},
    "vqf": {"imu": {"backend": "vqf"}},
    "vqf-stillness": {"imu": {"backend": "vqf", "vqf_bias_from": "stillness"}},
}
TRACKING = ("active", "selected", "resetting")


# --- variants --------------------------------------------------------------------------


def parse_variant(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    base = DEFAULT_VARIANTS.get(name, {"imu": {}})
    data = {k: dict(v) for k, v in base.items()}
    for item in filter(None, (p.strip() for p in rest.split(","))):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"variant {spec!r}: expected key=value, got {item!r}")
        try:
            parsed = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            parsed = value  # a bare string
        parts = key.split(".") if "." in key else ["imu", key]
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return (spec if rest else name), data


def make_config(base_paths: list[Path], overrides: dict) -> Config:
    cfg = load_config(base_paths)
    _merge(cfg, overrides, "")
    cfg.validate()
    return cfg


# --- running the engine ------------------------------------------------------------


class Collector:
    """Stub engine output: keeps events, and the calibration outcome."""

    def __init__(self):
        self.now = 0.0
        self.engine: Engine | None = None
        self.events: list[tuple[float, str, tuple]] = []
        self.calibrations: list[tuple[float, str, float]] = []  # t, bias source, |bias|

    def send(self, address: str, *args: object) -> None:
        if not address.endswith("/event"):
            return
        name = str(args[0])
        self.events.append((self.now, name, tuple(args[1:])))
        if name == "calibrated" and self.engine is not None:
            f = self.engine.fusion
            self.calibrations.append((self.now, f.bias_source, float(np.linalg.norm(f.bias))))


@dataclass
class Trace:
    t: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    state: list[str]
    target: list[tuple[str, str] | None]
    events: list[tuple[float, str, tuple]]
    calibrations: list[tuple[float, str, float]]


def run_engine(cfg: Config, session: Session, samples) -> Trace:
    out = Collector()
    eng = Engine(cfg, out)
    out.engine = eng
    headset = session.meta.get("headset") or "recorded"
    stale = cfg.muse.stale_after_s
    ts, yaws, pitches, states, targets = [], [], [], [], []
    last_t = None
    for s in samples:
        if last_t is not None and s.t - last_t > stale and eng.connected:
            out.now = last_t + stale
            eng.on_connection(False, out.now)
        out.now = s.t
        if not eng.connected:
            eng.on_connection(True, s.t, headset=headset)
        eng.process(s)
        if s.kind == "gyro":
            eng.tick(s.t)
            f = eng.fusion
            ts.append(s.t)
            yaws.append(f.yaw_deg)
            pitches.append(f.pitch_deg)
            states.append(eng.state.value)
            cur = eng.targets.current
            targets.append(None if cur is None else (cur.column, cur.row))
        last_t = s.t
    return Trace(np.array(ts), np.array(yaws), np.array(pitches), states, targets, out.events, out.calibrations)


# --- metrics ----------------------------------------------------------------------------


@dataclass
class PromptResult:
    t: float
    name: str
    column: str
    row: str
    yaw_true: float | None
    pitch_true: float | None
    yaw: float | None = None
    pitch: float | None = None
    target: tuple[str, str] | None = None
    gestures: list[str] = field(default_factory=list)  # nod / shake / selected:col:row / nod_ignored:reason

    @property
    def yaw_err(self) -> float | None:
        return None if self.yaw is None or self.yaw_true is None else self.yaw - self.yaw_true

    @property
    def pitch_err(self) -> float | None:
        return None if self.pitch is None or self.pitch_true is None else self.pitch - self.pitch_true


@dataclass
class Result:
    variant: str
    session: str
    calibrated_after_s: float | None
    bias_source: str
    face: list[PromptResult]  # targets phase
    check: list[PromptResult]  # after free looking
    nods: list[PromptResult]
    shakes: list[PromptResult]
    false_nods: int
    false_shakes: int


def _phase_times(events: list[Event]) -> dict[str, float]:
    return {e.name: e.t for e in events if e.kind == "phase"}


def evaluate(variant: str, session: Session, trace: Trace, window: tuple[float, float]) -> Result:
    phases = _phase_times(session.events)
    prompts = [e for e in session.events if e.kind == "prompt"]
    every = float(session.meta.get("protocol", {}).get("prompt_every_s", 5.0))
    tracking = np.array([s in TRACKING for s in trace.state], dtype=bool)

    def at_prompt(e: Event) -> PromptResult:
        r = PromptResult(e.t, e.name, e.column, e.row, e.yaw_deg, e.pitch_deg)
        lo, hi = e.t + window[0], e.t + window[1]
        m = (trace.t >= lo) & (trace.t <= hi) & tracking
        if m.any():
            r.yaw = float(np.median(trace.yaw[m]))
            r.pitch = float(np.median(trace.pitch[m]))
        i = int(np.searchsorted(trace.t, hi, side="right")) - 1
        if 0 <= i < len(trace.target):
            r.target = trace.target[i]
        for t, name, args in trace.events:
            if e.t <= t < e.t + every and name in ("nod", "shake", "selected", "nod_ignored"):
                r.gestures.append(":".join([name, *map(str, args)]))
        return r

    t_targets, t_gestures, t_free = phases.get("targets"), phases.get("gestures"), phases.get("free")
    face, check, nods, shakes = [], [], [], []
    for e in prompts:
        r = at_prompt(e)
        if e.name == "face":
            (check if t_free is not None and e.t > t_free else face).append(r)
        elif e.name == "nod":
            nods.append(r)
        elif e.name == "shake":
            shakes.append(r)

    # Gestures where none was asked for: targets phase, and free looking.
    quiet = []
    if t_targets is not None and t_gestures is not None:
        quiet.append((t_targets, t_gestures))
    if t_free is not None:
        quiet.append((t_free, check[0].t if check else phases.get("off", float("inf"))))
    false_nods = sum(1 for t, n, _ in trace.events if n == "nod" and any(a <= t < b for a, b in quiet))
    false_shakes = sum(1 for t, n, _ in trace.events if n == "shake" and any(a <= t < b for a, b in quiet))

    cal_after, source = None, "-"
    t_put_on = phases.get("put_on")
    for t, src, _ in trace.calibrations:
        if t_put_on is None or t >= t_put_on:
            cal_after, source = (t - t_put_on if t_put_on is not None else None), src
            break
    return Result(variant, session.folder.name, cal_after, source, face, check, nods, shakes,
                  false_nods, false_shakes)


# --- reporting ---------------------------------------------------------------------------


def _abs(vals) -> np.ndarray:
    return np.abs(np.array([v for v in vals if v is not None], dtype=float))


def summarize(results: list[Result]) -> dict:
    face = [p for r in results for p in r.face]
    yaw, pitch = _abs(p.yaw_err for p in face), _abs(p.pitch_err for p in face)
    hits = [p.target == (p.column, p.row) for p in face if p.yaw is not None]
    nods = [p for r in results for p in r.nods]
    shakes = [p for r in results for p in r.shakes]
    sel = [next((g for g in p.gestures if g.startswith("selected")), None) for p in nods]
    want = [f"selected:{p.column}:{p.row}" for p in nods]
    check = _abs(p.yaw_err for r in results for p in r.check)
    cal = [r.calibrated_after_s for r in results if r.calibrated_after_s is not None]

    def q(a, pct):
        return float(np.percentile(a, pct)) if len(a) else float("nan")

    return {
        "calib_s": float(np.median(cal)) if cal else float("nan"),
        "bias": "/".join(sorted({r.bias_source for r in results})),
        "yaw_med": q(yaw, 50), "yaw_p95": q(yaw, 95),
        "pitch_med": q(pitch, 50), "pitch_p95": q(pitch, 95),
        "target_ok": (sum(hits) / len(hits)) if hits else float("nan"),
        "check_yaw": float(np.median(check)) if len(check) else float("nan"),
        "nods": f"{sum(any(g.startswith('nod') and not g.startswith('nod_ignored') for g in p.gestures) for p in nods)}/{len(nods)}",
        "sel_ok": f"{sum(s == w for s, w in zip(sel, want))}/{len(nods)}",
        "sel_wrong": sum(1 for s, w in zip(sel, want) if s is not None and s != w),
        "shakes": f"{sum(any(g.startswith('shake') for g in p.gestures) for p in shakes)}/{len(shakes)}",
        "false": f"{sum(r.false_nods for r in results)}/{sum(r.false_shakes for r in results)}",
        # raw counts (not printed) for the decision rule
        "target_hits": sum(hits), "target_n": len(hits), "selections_wrong": sum(1 for s, w in zip(sel, want)
                                                                                 if s is not None and s != w),
        "gesture_errors": (sum(not any(g.startswith("nod") and not g.startswith("nod_ignored") for g in p.gestures)
                               for p in nods)
                           + sum(not any(g.startswith("shake") for g in p.gestures) for p in shakes)
                           + sum(r.false_nods + r.false_shakes for r in results)),
    }


COLUMNS = [
    ("variant", "{:<16}", 16), ("calib_s", "{:>7.1f}", 7), ("bias", "{:>6}", 6),
    ("yaw_med", "{:>7.1f}", 7), ("yaw_p95", "{:>7.1f}", 7), ("pitch_med", "{:>9.1f}", 9),
    ("pitch_p95", "{:>9.1f}", 9), ("target_ok", "{:>9.0%}", 9), ("check_yaw", "{:>9.1f}", 9),
    ("nods", "{:>5}", 5), ("sel_ok", "{:>6}", 6), ("sel_wrong", "{:>9}", 9),
    ("shakes", "{:>6}", 6), ("false", "{:>9}", 9),
]
HEADERS = {"false": "false n/s", "calib_s": "calib s", "target_ok": "target"}


def _cells(name: str, s: dict) -> list[str]:
    vals = {"variant": name, **s}
    cells = []
    for k, fmt, w in COLUMNS:
        v = vals[k]
        if isinstance(v, float) and math.isnan(v):
            cells.append(f"{'-':>{w}}")
            continue
        try:
            cells.append(fmt.format(v))
        except (ValueError, TypeError):
            cells.append(f"{v!s:>{w}}")
    return cells


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    print("  " + " ".join(f"{HEADERS.get(k, k):>{w}}" if k != "variant" else f"{k:<{w}}" for k, _, w in COLUMNS))
    for name, s in rows:
        print("  " + " ".join(_cells(name, s)))


def markdown_table(rows: list[tuple[str, dict]]) -> str:
    head = [HEADERS.get(k, k) for k, _, _ in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|"]
    for name, s in rows:
        lines.append("| " + " | ".join(c.strip() for c in _cells(name, s)) + " |")
    return "\n".join(lines)


def paired_hits(results: dict[str, list[Result]], a: str, b: str) -> tuple[int, int]:
    """Face prompts on target with `a` only, and with `b` only (same prompts)."""
    hits = {name: {(r.session, round(p.t, 3)): p.target == (p.column, p.row)
                   for r in results[name] for p in r.face if p.yaw is not None} for name in (a, b)}
    common = hits[a].keys() & hits[b].keys()
    return (sum(hits[a][k] and not hits[b][k] for k in common),
            sum(hits[b][k] and not hits[a][k] for k in common))


def sign_test_p(k: int, n: int) -> float:
    """Exact two-sided binomial test of k out of n against 1/2 (McNemar's test)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def decide(results: dict[str, list[Result]], default: str = "complementary",
           candidate: str = "vqf", alpha: float = 0.05) -> tuple[bool, list[str]]:
    """The plan's rule: switch the default only if the candidate improves target
    accuracy without more gesture errors. The improvement must also be clear: on
    the prompts where only one of the two was on target, the candidate must win
    significantly (exact sign test). Returns (switch, reasons)."""
    d, c = summarize(results[default]), summarize(results[candidate])
    reasons = []
    better = (c["target_hits"] > d["target_hits"] and c["selections_wrong"] <= d["selections_wrong"]) or (
        c["selections_wrong"] < d["selections_wrong"] and c["target_hits"] >= d["target_hits"])
    reasons.append(f"target accuracy: {candidate} {c['target_hits']}/{c['target_n']} prompts on target, "
                   f"{c['selections_wrong']} wrong selections; {default} {d['target_hits']}/{d['target_n']}, "
                   f"{d['selections_wrong']} wrong -> {'better' if better else 'not better'}")
    only_c, only_d = paired_hits(results, candidate, default)
    p = sign_test_p(only_c, only_c + only_d)
    clear = better and only_c > only_d and p < alpha
    reasons.append(f"prompts on target with only one of them: {candidate} {only_c}, {default} {only_d} "
                   f"(sign test p = {p:.2g}) -> {'a clear difference' if clear else 'not a clear difference'}")
    no_worse = c["gesture_errors"] <= d["gesture_errors"]
    reasons.append(f"gesture errors (missed + false nods and shakes): {candidate} {c['gesture_errors']}, "
                   f"{default} {d['gesture_errors']} -> {'not more' if no_worse else 'more'}")
    return clear and no_worse, reasons


def per_session_comparison(results: dict[str, list[Result]], default: str, candidate: str) -> list[str]:
    lines = []
    by_session = {r.session: r for r in results[default]}
    for rc in results[candidate]:
        rd = by_session.get(rc.session)
        if rd is None:
            continue
        sd, sc = summarize([rd]), summarize([rc])
        dt = sc["target_hits"] - sd["target_hits"]
        dg = sc["gesture_errors"] - sd["gesture_errors"]
        dy = sc["yaw_p95"] - sd["yaw_p95"]
        lines.append(f"| {rc.session} | {dt:+d} | {sc['selections_wrong'] - sd['selections_wrong']:+d} | "
                     f"{dg:+d} | {dy:+.1f} |")
    return lines


def write_report(path: Path, sessions: list[Session], results: dict[str, list[Result]],
                 base: list[Path], window: tuple[float, float], retimed: bool) -> None:
    """A draft results note for docs/ (step 4 of the VQF plan), to review and edit."""
    import datetime as dt

    simulated = [s.folder.name for s in sessions if s.meta.get("simulated")]
    out = [f"# IMU orientation backends: bench results ({dt.date.today().isoformat()})", ""]
    out.append(f"Draft written by `bench/imu/replay.py --report`; review the numbers before acting on them. "
               f"{len(sessions)} session(s), {len({s.meta.get('participant', s.folder.name) for s in sessions})} "
               f"participant(s). Engine config: {', '.join(f'`{q.relative_to(ROOT) if q.is_relative_to(ROOT) else q}`' for q in base)}. "
               f"Yaw and pitch measured {window[0]:g}-{window[1]:g} s after each prompt"
               + ("; duplicate LSL timestamps re-timed." if retimed else "; duplicate timestamps kept (not re-timed)."))
    if simulated:
        out += ["", (f"> **Includes simulated sessions** ({', '.join(simulated)}): these test the tools, "
                     "they are not evidence for choosing a backend.")]

    out += ["", "## Recommendation", ""]
    if "complementary" in results and "vqf" in results:
        switch, reasons = decide(results)
        out.append(f"**{'Switch the default to `vqf`' if switch else 'Keep `complementary` as the default'}** "
                   "(rule from the plan: switch only if VQF clearly improves target accuracy without more "
                   "gesture errors).")
        out += [""] + [f"- {r}" for r in reasons]
        out.append("- a clear difference needs about 6 or more prompts won by VQF and none lost "
                   "(or 8 won to 1 lost, ...): record more sessions if the result is close.")
    else:
        out.append("Replay both `complementary` and `vqf` to apply the decision rule.")

    out += ["", "## All sessions", "", markdown_table([(n, summarize(rs)) for n, rs in results.items()]), ""]
    out.append("Errors in degrees. yaw/pitch: |error| at face prompts before re-anchoring (median, 95th "
               "percentile). target: prompts where the engine's target was the prompted one. check yaw: "
               "facing the middle after a minute of free looking. nods/shakes: detected / prompted. sel_ok / "
               "sel_wrong: selections on the right / a wrong target. false n/s: nods / shakes outside gesture prompts.")
    if "complementary" in results and "vqf" in results and len(sessions) > 1:
        out += ["", "## Per session: vqf minus complementary", "",
                "| session | on target | wrong selections | gesture errors | yaw p95 |", "|---|---:|---:|---:|---:|"]
        out += per_session_comparison(results, "complementary", "vqf")
        out += ["", "On target: more is better. The other columns: less is better."]
    if len(sessions) > 1:
        for s in sessions:
            out += ["", f"### {s.folder.name}", "",
                    markdown_table([(n, summarize([r for r in rs if r.session == s.folder.name]))
                                    for n, rs in results.items()])]
    out += ["", "## Sessions", ""]
    for s in sessions:
        out.append(f"- **{s.folder.name}**: headset {s.meta.get('headset', '?')}, "
                   f"{s.meta.get('duration_s', '?')} s, {len(s.samples)} samples"
                   + (" (simulated)" if s.meta.get("simulated") else ""))
        out += [f"  {line.strip()}" for line in stand_report(s).splitlines()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def stand_report(session: Session) -> str:
    phases = _phase_times(session.events)
    lines = []
    rep = timing(session.samples)
    lines.append(f"  rate {', '.join(f'{k} {v:.2f} Hz' for k, v in rep.rate_hz.items())}; "
                 f"duplicate timestamps {rep.duplicates}; gaps > 0.1 s {rep.gaps}, "
                 f"longest {max(rep.max_gap_s.values()):.2f} s")
    if "stand" in phases and "put_on" in phases:
        a, b = phases["stand"] + 5.0, phases["put_on"] - 5.0
        g = np.array([(s.x, s.y, s.z) for s in session.samples if s.kind == "gyro" and a <= s.t <= b])
        acc = np.array([(s.x, s.y, s.z) for s in session.samples if s.kind == "acc" and a <= s.t <= b])
        if len(g) > 10:
            mean, std = g.mean(axis=0), g.std(axis=0)
            lines.append(f"  on the stand: gyro mean (= raw bias) {np.round(mean, 2).tolist()} deg/s, "
                         f"std {np.round(std, 2).tolist()}, |acc| {np.linalg.norm(acc, axis=1).mean():.3f} g")
            if np.max(np.abs(mean)) > 4.0:
                lines.append("  ! a bias component is near or above vqf_bias_clip_dps (5): raise it for VQF")
    clip = clipped(session.samples)
    if clip:
        order = sorted(phases.items(), key=lambda kv: kv[1])
        per: dict[str, int] = {}
        for t in clip:
            name = next((n for n, start in reversed(order) if start <= t), "before start")
            per[name] = per.get(name, 0) + 1
        lines.append(f"  gyro at full scale (245 deg/s) on {len(clip)} samples: "
                     + ", ".join(f"{n} {k}" for n, k in per.items())
                     + " (yaw loses the excess; harmless in a shake, which re-zeroes yaw)")
    return "\n".join(lines)


def write_prompts(path: Path, results: list[Result]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "session", "prompt", "t", "column", "row", "yaw_true", "yaw", "pitch_true",
                    "pitch", "target", "gestures"])
        for r in results:
            for p in [*r.face, *r.check, *r.nods, *r.shakes]:
                w.writerow([r.variant, r.session, p.name, f"{p.t:.3f}", p.column, p.row, p.yaw_true,
                            None if p.yaw is None else round(p.yaw, 2), p.pitch_true,
                            None if p.pitch is None else round(p.pitch, 2),
                            "" if p.target is None else "/".join(p.target), " ".join(p.gestures)])


def replay(sessions: list[Session], variants: list[tuple[str, dict]], base_paths: list[Path],
           do_retime: bool = True, window: tuple[float, float] = (2.5, 4.5)) -> dict[str, list[Result]]:
    out: dict[str, list[Result]] = {name: [] for name, _ in variants}
    for session in sessions:
        samples = session.samples
        if do_retime:
            samples, _ = retime(samples)
        for name, overrides in variants:
            cfg = make_config(base_paths, overrides)
            trace = run_engine(cfg, session, samples)
            out[name].append(evaluate(name, session, trace, window))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path, help="session folders (or their imu.csv)")
    p.add_argument("--variant", action="append", help="NAME or NAME:key=value,... (repeatable)")
    p.add_argument("-c", "--config", type=Path, action="append",
                   help="base engine config (default config/installation.toml, then config/local.toml)")
    p.add_argument("--no-retime", action="store_true",
                   help="keep duplicate timestamps (the engine then drops 2 of 3 samples, as it did over LSL before the source re-timed them)")
    p.add_argument("--window", default="2.5,4.5", help="seconds after a prompt to measure yaw/pitch")
    p.add_argument("--prompts", type=Path, help="write one CSV row per prompt and variant")
    p.add_argument("--per-session", action="store_true", help="also print a table per session")
    p.add_argument("--report", type=Path, help="write a Markdown results note (e.g. docs/imu-backend-results.md)")
    args = p.parse_args(argv)

    base = args.config or [q for q in (ROOT / "config/installation.toml", ROOT / "config/local.toml") if q.exists()]
    specs = args.variant or [v for v in DEFAULT_VARIANTS
                             if not v.startswith("vqf") or _have_vqf()]
    try:
        variants = [parse_variant(s) for s in specs]
        for _, overrides in variants:
            make_config(base, overrides)
    except (ConfigError, ValueError) as e:
        print(f"variant error: {e}", file=sys.stderr)
        return 2
    sessions = [load_session(s) for s in args.sessions]
    w0, w1 = (float(x) for x in args.window.split(","))

    for s in sessions:
        print(f"\n{s.folder.name}: {len(s.samples)} samples, {s.meta.get('duration_s', '?')} s, "
              f"headset {s.meta.get('headset', '?')}")
        print(stand_report(s))
        if not args.no_retime:
            _, squeezed = retime(s.samples)
            if any(squeezed.values()):
                print(f"  retimed: samples in packet bursts {squeezed}")
    results = replay(sessions, variants, base, do_retime=not args.no_retime, window=(w0, w1))
    if args.per_session and len(sessions) > 1:
        for s in sessions:
            print_table(s.folder.name, [(n, summarize([r for r in rs if r.session == s.folder.name]))
                                        for n, rs in results.items()])
    print_table(f"All sessions ({len(sessions)}); errors in degrees, prompts measured "
                f"{w0:g}-{w1:g} s after the prompt", [(n, summarize(rs)) for n, rs in results.items()])
    if "complementary" in results and "vqf" in results:
        switch, reasons = decide(results)
        print(f"\n{'switch the default to vqf' if switch else 'keep complementary as the default'}:")
        for r in reasons:
            print(f"  {r}")
    if args.report:
        write_report(args.report, sessions, results, base, (w0, w1), not args.no_retime)
        print(f"\nresults note: {args.report}")
    if args.prompts:
        write_prompts(args.prompts, [r for rs in results.values() for r in rs])
        print(f"\nper-prompt rows: {args.prompts}")
    return 0


def _have_vqf() -> bool:
    import importlib.util
    return importlib.util.find_spec("vqf") is not None


if __name__ == "__main__":
    sys.exit(main())

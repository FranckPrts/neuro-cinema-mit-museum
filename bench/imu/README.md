# IMU bench: record sessions, compare orientation backends

Records the Muse accelerometer and gyroscope (**IMU only, no EEG**) while a
participant follows spoken prompts, then replays the recordings through the
engine with several orientation backends (complementary filter, VQF) and
compares them. This is a separate bench tool: `nc-engine` itself never records
anything.

- Participants are identified by a code (`P01`), never by name. Ask for their
  consent before recording.
- Recordings go to `bench/imu/recordings/`, which git ignores.

## Setup

```bash
source .venv/bin/activate
pip install -e ".[muse,vqf]"
```

## The room (or a desk)

The participant stands (or sits) on one spot, facing the middle screen. Six
targets, each needing a marker (sticky notes are fine):

| | left | middle | right |
|---|---|---|---|
| **top** | yaw −40°, pitch +15° | yaw 0°, pitch +15° | yaw +40°, pitch +15° |
| **bottom** | yaw −40°, pitch −15° | yaw 0°, pitch −15° | yaw +40°, pitch −15° |

Yaw is the angle from the middle, + = the participant's right. Pitch is
measured from the calibration point (the centre of the middle screen, at eye
height). The screen yaws come from `[targets]` in the config, or pass
`--yaw -35,0,35`. Pass `--top-pitch` and `--bottom-pitch` to change the rows. At
2 m from the participant, 40° is 1.68 m to the side and 15° is 0.54 m up or down.

## Recording (about 6 minutes per person)

1. Choose how to reach the headset (`--source`; the default is `source.kind`
   in the config). Both go through the engine's own sources, so the recording
   holds exactly what the engine would see:
   - `muse` (LSL): start the stream first, either with `nc-engine` (with
     `source.kind = "muse"`, so it launches muselsl, and `nc-monitor` alongside
     to watch) or with `muselsl stream --acc --gyro --lsltime --disable-eeg`;
   - `muse_direct` (in-process, as the museum will run it): stop `nc-engine`
     first, because a Muse accepts one Bluetooth connection. The headset must
     be in `[headsets] registry`; pass `--headset A` if there are several.
     When asked, pick the headset up to confirm it, as on the status page.
2. Put the headset on its stand, switched on.
3. `python bench/imu/record.py --participant P01` (add `--source muse_direct
   --headset A` for the in-process source), then follow the spoken prompts
   (macOS `say`; `--no-voice` to turn them off):
   1. **stand** (2 min): the headset stays still on its stand;
   2. **put on**: put it on, look at the middle of the middle screen, keep still;
   3. **targets**: "middle top", "left bottom", …, one every 5 s (18 prompts);
   4. **gestures**: "left top, and nod" (6 times), then "face the middle, and
      shake your head" (3 times);
   5. **free**: look around freely for a minute, no nods or shakes;
   6. **check**: "middle top";
   7. **off**: put the headset back on its stand.

Do a dry run first with `--quick` (about 2 minutes). Ctrl-C stops early; the
partial recording is kept and marked `aborted`. `--fake-headset` records a
simulated headset through the `muse_direct` path, to try the recorder without
a Muse (it does not follow the prompts).

If the camera bench runs at the same time (`bench/camera`, `--log`),
`meta.json` stores `time.time()` and `time.monotonic()` at the start, so the
two recordings can be lined up.

## Replay

```bash
python bench/imu/replay.py bench/imu/recordings/P0*          # all sessions, default variants
python bench/imu/replay.py REC --per-session --prompts prompts.csv
python bench/imu/replay.py REC --variant complementary --variant vqf:vqf_tau_acc_s=2.0
python bench/imu/replay.py bench/imu/recordings/P0* --report docs/imu-backend-results.md
```

Each variant runs the whole engine (fusion, gestures, targets, state machine,
re-anchoring) on the recording. The default variants are `baseline` (no idle
bias, as before), `complementary` (today's default), `vqf` and `vqf-stillness`.
The table shows:

- `calib s`, `bias`: time from "put on" to calibration, and where the bias came
  from (`worn` 2 s mean, `idle` stand mean, `vqf`);
- `yaw_med`, `yaw_p95`, `pitch_med`, `pitch_p95`: absolute error at the
  "targets" prompts in degrees, measured 2.5–4.5 s after each prompt, before
  any re-anchoring;
- `target`: share of those prompts where the engine's target was the one asked for;
- `check_yaw`: yaw error facing the middle after a minute of free looking;
- `nods`, `sel_ok`, `sel_wrong`, `shakes`: gestures detected when asked for, and
  selections that landed on the right or wrong target;
- `false n/s`: nods and shakes when none were asked for.

It first prints, per session, the sample rates, duplicate timestamps and gaps,
and the gyro mean on the stand (the raw gyro bias). If any bias component is
near `imu.vqf_bias_clip_dps` (10 °/s), it warns: raise that limit, because VQF
can't learn a bias above it (it leaves pitch a few degrees off).

Absolute errors also include how far people turn their head versus their eyes
(heads usually turn less than the gaze), so compare variants rather than
reading the numbers as accuracy.

### The results note (`--report`)

`--report PATH` writes a Markdown note for `docs/`: the recommendation, the
tables (all sessions and per session), a per-session comparison of `vqf`
against `complementary`, and each session's checks. It applies the plan's
rule: switch the default to `vqf` only if it improves target accuracy (more
prompts on target, no more wrong selections) without more gesture errors
(missed plus false nods and shakes). The improvement must also be clear: on the
prompts where only one of the two was on target, VQF must win an exact sign
test at p < 0.05, which takes about 6 prompts won and none lost, or 8 won to
1 lost. With fewer, the note says to keep `complementary` and record more
sessions. The console prints the same verdict. Review the draft before
committing it.

### Gyro clipping

The Muse gyro reads at most ±245 °/s per axis; faster motion reads as exactly
245, and the yaw loses the rest. Replay lists, per session, how many gyro
samples were at full scale and in which phase. In a shake it does not matter,
because the shake resets yaw. In fast turns it does: in simulation, turns at
up to 400 °/s raise the yaw error at prompts from 0.7° to 5° (median) and from
1.6° to 16° (95th percentile), the same for both backends. `nc-engine` logs one
line per burst ("gyro at full scale…").

### Duplicate timestamps (LSL)

`muselsl stream` gives the 3 samples of each Muse IMU packet the same
timestamp, and the engine drops samples whose time does not increase. The
engine's LSL source now spreads them back over 1/52 s steps (as the in-process
source does), so recordings made with `record.py` have none. Replay still
spreads any it finds, the same way, for older recordings; `--no-retime` replays
them as they are, which is what the engine saw over LSL before that fix (1
sample in 3).

## Without a headset

```bash
python bench/imu/simulate.py --out bench/imu/recordings/SIM01 --duplicates
python bench/imu/replay.py bench/imu/recordings/SIM01
```

The virtual head follows the protocol (with the sensor 15 cm from the neck
pivot, so the accelerometer sees head-turn accelerations). Its gyro clips at
245 °/s like the Muse. `--turn-dps 400` (fast turns) and `--shake-deg 28`
(vigorous shakes, about 440 °/s) make a participant who goes past it.

## Tests

```bash
python -m pytest bench/imu
```

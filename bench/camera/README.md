# Camera bench: pick a screen without the IMU

First attempt at using a camera instead of the Muse IMU for screen selection.
The Muse is not used here at all. MediaPipe Pose finds the visitor and
`pointing.py` turns a raised, straight arm into an eye-to-fingertip ray in the
room frame. `viewer/room_geometry.py` (with `config/room.toml`) then says which
screen the ray hits. Hold on one screen for 1.5 s to select it.

```bash
source .venv/bin/activate
pip install -r bench/camera/requirements.txt
python bench/camera/camera_bench.py
```

The first run downloads the pose model (about 9 MB) into `bench/camera/models/`.
On macOS, allow camera access for your terminal app (System Settings > Privacy &
Security > Camera) if the camera doesn't open.

## Trying it at a desk

With the laptop camera in front of you, the room maths assume that you're
standing on the visitor's spot and the camera is on the front (middle) screen.
- Point roughly at the camera: **middle** screen.
- Point well off to your right (more than about 30°): **right** screen. Same on the left.
- Drop your arm: nothing is selected after 0.5 s.

Press `h` to switch to head direction and `a` to switch back. Head direction
here comes from the body-pose model (nose against the ears). It's rough and
reads a turned head as less turned than it is. If head aiming matters, the next
step is MediaPipe's Face Landmarker.

## Options

| Flag | What it does |
|---|---|
| `--bearing 29.6` | Camera in the front-right corner (seen from the standing spot, + = visitor's right). Use `-29.6` for front-left. |
| `--tilt 15` | The camera looks down 15° (mounted high). |
| `--roi x0,y0,x1,y1` | Only look inside this box, in fractions of the frame (the standing zone). Press `r` to draw it with the mouse, and it prints the flag to reuse. |
| `--select head` | Head direction drives the selection instead of the arm. |
| `--dwell 1.5` `--hold 0.25` `--smooth 0.15` | Seconds to select, seconds before a new screen lights up, aim smoothing. |
| `--osc 127.0.0.1:7003` | Sends `/nc/cam/aim yaw pitch`, `/nc/cam/highlight left|middle|right|none` and `/nc/cam/select <screen>`. Off by default, and kept apart from the engine's `/nc/state` on purpose. |
| `--log logs/cam.csv` | One CSV row per frame (aim, head yaw, highlight, events), to compare runs. |
| `--camera 1` or `--camera clip.mp4` | Another camera, or replay a video. |
| `--model lite|full|heavy` | Speed against accuracy. `full` is the default. |

## Tests

```bash
python -m pytest bench/camera
```

These test the maths only, with no camera or MediaPipe needed.

## Known limits of this first pass

- One person only (`num_poses=1`). Use `--roi` to keep it on the standing zone.
- MediaPipe's 3D (world) landmarks are estimates from a single camera. Depth
  (toward or away from the camera) is the weakest axis, so pointing straight
  at or straight away from the camera is the least precise.
- Nothing is recorded. Frames stay in memory.

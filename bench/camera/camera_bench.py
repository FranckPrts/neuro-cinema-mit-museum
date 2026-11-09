#!/usr/bin/env python3
"""Camera bench: pick a screen by pointing (or by head direction) with a webcam.

First attempt at replacing the Muse IMU for screen selection. MediaPipe Pose
finds the body, the eye-to-fingertip ray of a raised arm is turned into the
room frame, and viewer/room_geometry.py says which wall screen it hits and
whether the ray lands in its top or bottom half. Holding on one screen half for
--dwell seconds selects it.

    python bench/camera/camera_bench.py                 # laptop webcam, camera in front
    python bench/camera/camera_bench.py --camera-config my_corner.toml  # camera placed elsewhere
    python bench/camera/camera_bench.py --select head   # compare: head direction

Where the camera is goes in bench/camera/camera.toml (position and aim point in
the room frame); the bench turns that into a bearing and tilt.

Keys: q / Esc quit, a = arm, h = head, m = mirror view, r = draw zone (ROI),
c = pitch calibration: press c, then you have --calib-delay seconds (5 by default)
to step into place and point at the middle of the middle screen (at a desk:
straight at the camera); the pitch is then averaged over half a second.
Press c again to cancel. x = clear the calibration.
Nothing is recorded; frames stay in memory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(HERE))

import room_geometry as rg  # noqa: E402
from camera_pose import CameraConfigError, load_camera  # noqa: E402
from osc_out import CamOsc  # noqa: E402
from panels import PanelMapper, screen_part  # noqa: E402
from pointing import Aim, Selector, arm_aim, head_aim  # noqa: E402

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

# Skeleton lines to draw (MediaPipe Pose indices).
BONES = [(11, 12), (11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20),
         (11, 23), (12, 24), (23, 24), (0, 7), (0, 8)]
COLOURS = {"left": (255, 160, 60), "middle": (80, 220, 80), "right": (60, 160, 255), None: (140, 140, 140)}


def model_path(variant: str) -> Path:
    p = HERE / "models" / f"pose_landmarker_{variant}.task"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = MODEL_URL.format(v=variant)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, p)
    return p


def parse_roi(s: str | None):
    if not s:
        return None
    x0, y0, x1, y1 = (float(v) for v in s.split(","))
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError("--roi wants x0,y0,x1,y1 as fractions of the frame, x0<x1, y0<y1")
    return (x0, y0, x1, y1)


def draw_map(img, cfg, aim_yaw_pitch, highlight, dwell, raw_yaw=None, cam=None, size=220, pad=12):
    """Top-down room in the bottom-right corner: screens, visitor, aim ray."""
    highlight = screen_part(highlight)
    r = cfg["room"]
    W, D = r["width_m"], r["depth_m"]
    s = (size - 2 * pad) / max(W, D)
    H_, W_ = img.shape[:2]
    ox, oy = W_ - size - 10, H_ - size - 10
    cv2.rectangle(img, (ox, oy), (ox + size, oy + size), (25, 25, 25), -1)

    def px(x, z):  # room x (right) / z (+ = behind) -> pixel; front wall at top
        return int(ox + pad + (x + W / 2) * s), int(oy + pad + (z + D - r["visitor_inset_m"]) * s)
    walls = rg.walls(cfg)
    names = {"front": "middle", "left": "left", "right": "right"}
    for name, w in walls.items():
        a = (w["origin"][0], w["origin"][2])
        b = (a[0] + w["u"][0] * w["width"], a[1] + w["u"][2] * w["width"])
        on = name in cfg["screens"]["walls"]
        scr = names.get(name)
        col = COLOURS.get(scr, (90, 90, 90)) if on else (70, 70, 70)
        th = 6 if (on and scr == highlight) else 2
        cv2.line(img, px(*a), px(*b), col, th)
    eye = px(0.0, 0.0)
    if cam is not None:  # camera: yellow triangle, dotted line to what it looks at
        c, a = px(cam.position[0], cam.position[2]), px(cam.aim[0], cam.aim[2])
        for i in range(0, 20):
            if i % 2 == 0:
                p0 = (c[0] + (a[0] - c[0]) * i // 20, c[1] + (a[1] - c[1]) * i // 20)
                p1 = (c[0] + (a[0] - c[0]) * (i + 1) // 20, c[1] + (a[1] - c[1]) * (i + 1) // 20)
                cv2.line(img, p0, p1, (0, 200, 255), 1)
        tri = np.array([[c[0], c[1] - 7], [c[0] - 6, c[1] + 5], [c[0] + 6, c[1] + 5]], np.int32)
        cv2.fillPoly(img, [tri], (0, 200, 255))
    cv2.circle(img, eye, 5, (255, 255, 255), -1)
    if raw_yaw is not None:  # unsmoothed aim, thin grey
        p = rg.hit(cfg, raw_yaw, 0.0)["point"]
        cv2.line(img, eye, px(p[0], p[2]), (120, 120, 120), 1)
    if aim_yaw_pitch is not None:
        h = rg.hit(cfg, aim_yaw_pitch[0], 0.0)
        p = h["point"]
        cv2.line(img, eye, px(p[0], p[2]), (255, 255, 255), 2)
    if highlight and dwell > 0:
        cv2.ellipse(img, eye, (18, 18), -90, 0, int(360 * dwell), COLOURS[highlight], 3)


def draw_side(img, cfg, mapper, aim_yaw_pitch, highlight, raw_pitch=None, size=220, pad=14, gap=10):
    """Side view of the wall being aimed at, above the top-down map: eye, aim ray, screen halves.

    Horizontal = distance from the eye along the aim's heading; vertical = height.
    """
    r = cfg["room"]
    H_, W_ = img.shape[:2]
    ox, oy = W_ - size - 10, H_ - 2 * size - 10 - gap
    cv2.rectangle(img, (ox, oy), (ox + size, oy + size), (25, 25, 25), -1)
    yaw = aim_yaw_pitch[0] if aim_yaw_pitch else 0.0
    flat = rg.hit(cfg, yaw, 0.0)          # which wall this heading reaches, and how far
    wall, p = flat["surface"], flat["point"]
    dist = math.hypot(p[0], p[2])
    ceil = r["ceiling_m"]
    s = (size - 2 * pad) / max(dist, ceil)

    def px(d, h):
        return int(ox + pad + d * s), int(oy + size - pad - h * s)
    cv2.line(img, px(0, 0), px(dist, 0), (90, 90, 90), 1)        # floor
    cv2.line(img, px(0, ceil), px(dist, ceil), (60, 60, 60), 1)  # ceiling
    screen = {"front": "middle", "left": "left", "right": "right"}.get(wall)
    if screen and wall in cfg["screens"]["walls"]:
        inset = cfg["screens"]["inset_m"]
        mid = mapper.midline_height(wall)
        col = COLOURS[screen]
        for row, lo, hi in (("bottom", inset, mid), ("top", mid, ceil - inset)):
            on = highlight == f"{screen}/{row}"
            cv2.line(img, px(dist, lo), px(dist, hi), col, 7 if on else 2)
            cv2.putText(img, row, (px(dist, (lo + hi) / 2)[0] - 46, px(dist, (lo + hi) / 2)[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.line(img, px(dist - 0.15, mid), px(dist, mid), (255, 255, 255), 1)
        label = f"{screen} screen, side view"
    else:
        cv2.line(img, px(dist, 0), px(dist, ceil), (70, 70, 70), 2)
        label = f"{wall} wall, side view"
    cv2.putText(img, label, (ox + 8, oy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    eye_h = r["eye_height_m"]
    eye = px(0, eye_h)
    cv2.circle(img, eye, 5, (255, 255, 255), -1)

    def ray(pitch, colour, th):
        h = rg.hit(cfg, yaw, pitch)
        q = h["point"]
        cv2.line(img, eye, px(math.hypot(q[0], q[2]), q[1]), colour, th)
    if raw_pitch is not None:
        ray(raw_pitch, (120, 120, 120), 1)
    if aim_yaw_pitch is not None:
        ray(aim_yaw_pitch[1], (255, 255, 255), 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index, or a video file to replay")
    ap.add_argument("--select", choices=("arm", "head"), default="arm", help="what drives the selection")
    ap.add_argument("--camera-config", type=Path, default=HERE / "camera.toml",
                    help="camera position and aim point in the room (default: bench/camera/camera.toml)")
    ap.add_argument("--bearing", type=float, default=None,
                    help="override: camera direction seen from the standing spot, deg (+ = visitor's right)")
    ap.add_argument("--tilt", type=float, default=None, help="override: how far the camera looks down, deg")
    ap.add_argument("--roi", type=parse_roi, default=None,
                    help="only look inside this box: x0,y0,x1,y1 as fractions of the frame (the standing zone)")
    ap.add_argument("--model", choices=("lite", "full", "heavy"), default="full")
    ap.add_argument("--room", type=Path, default=REPO / "config" / "room.toml")
    ap.add_argument("--dwell", type=float, default=1.5, help="seconds on one screen to select it")
    ap.add_argument("--hold", type=float, default=0.25, help="seconds before a new screen lights up")
    ap.add_argument("--smooth", type=float, default=0.15, help="aim smoothing time constant, s")
    ap.add_argument("--split", type=float, default=0.5,
                    help="top/bottom divider, as a fraction of the screen height from its bottom edge")
    ap.add_argument("--band", type=float, default=3.0,
                    help="dead band around the divider, deg of pitch (stops top/bottom flicker)")
    ap.add_argument("--no-halves", action="store_true", help="screens only, no top/bottom")
    ap.add_argument("--pitch-offset", type=float, default=0.0,
                    help="deg added to the arm's pitch (the c key sets this live and prints it)")
    ap.add_argument("--calib-delay", type=float, default=5.0,
                    help="seconds between pressing c and the calibration sample, to step into place")
    ap.add_argument("--osc", default=None,
                    help="host:port to send /nc/cam/* to, in the engine's message format (off by default)")
    ap.add_argument("--log", type=Path, default=None, help="write one CSV row per frame here")
    ap.add_argument("--no-mirror", action="store_true", help="show the image unmirrored")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)

    cfg = rg.load_room(args.room)
    try:
        cam = load_camera(args.camera_config)
    except CameraConfigError as e:
        print(f"camera config: {e}", file=sys.stderr)
        return 2
    bearing = cam.bearing_deg if args.bearing is None else args.bearing
    tilt = cam.tilt_deg if args.tilt is None else args.tilt
    src_note = "" if args.bearing is None and args.tilt is None else "  (overridden on the command line)"
    print(f"camera at x={cam.position[0]:.2f} y={cam.position[1]:.2f} z={cam.position[2]:.2f} m, "
          f"aimed at x={cam.aim[0]:.2f} y={cam.aim[1]:.2f} z={cam.aim[2]:.2f} m "
          f"({cam.distance_m:.2f} m away) -> bearing {bearing:+.1f} deg, tilt {tilt:+.1f} deg{src_note}")
    mapper = PanelMapper(cfg, split=args.split, band_deg=args.band, halves=not args.no_halves)
    selector = Selector(mapper, smooth_s=args.smooth, hold_s=args.hold, dwell_s=args.dwell)
    pitch_offset = args.pitch_offset
    CALIB_MEASURE_S = 0.5
    calib_start: float | None = None   # time c was pressed; None = not calibrating
    calib_samples: list[float] = []    # raw arm pitch while measuring
    notice, notice_until = "", 0.0     # short message after a calibration

    osc = None
    if args.osc:
        from pythonosc.udp_client import SimpleUDPClient
        host, port = args.osc.rsplit(":", 1)
        client = SimpleUDPClient(host, int(port))
        osc = CamOsc(lambda address, *a: client.send_message(address, list(a)), t0=0.0)

    log_f = log_w = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t", "fps", "person", "arm_yaw", "arm_pitch", "arm_src", "head_yaw",
                        "aim_yaw", "aim_pitch", "pitch_from_divider", "highlight", "event"])

    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera!r}. On macOS, allow camera access for your terminal "
              "in System Settings > Privacy & Security > Camera.", file=sys.stderr)
        return 1

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(args.model))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))

    mode, mirror, roi = args.select, not args.no_mirror, args.roi
    win = "Neurocinema camera bench"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t0, last, fps, last_ts = time.monotonic(), time.monotonic(), 0.0, -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            t = now - t0
            fps = 0.9 * fps + 0.1 / max(1e-3, now - last) if fps else 1 / max(1e-3, now - last)
            last = now
            H, W = frame.shape[:2]

            x0, y0, x1, y1 = (0, 0, W, H)
            if roi:
                x0, y0, x1, y1 = int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)
            crop = np.ascontiguousarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), ts)

            arm = head = None
            if res.pose_world_landmarks:
                world, img_lm = res.pose_world_landmarks[0], res.pose_landmarks[0]
                arm = arm_aim(world, bearing, tilt)
                if arm and pitch_offset:
                    arm = Aim(arm.yaw, arm.pitch + pitch_offset, arm.source, arm.quality)
                head = head_aim(world, bearing, tilt)
                pts = [(int(x0 + l.x * (x1 - x0)), int(y0 + l.y * (y1 - y0))) for l in img_lm]
                for a, b in BONES:
                    cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
                if arm:
                    side = 19 if arm.source.startswith("left") else 20
                    eye = ((pts[2][0] + pts[5][0]) // 2, (pts[2][1] + pts[5][1]) // 2)
                    cv2.line(frame, eye, pts[side], (0, 255, 255), 3)
            aim = arm if mode == "arm" else head
            mapper.halves = mode == "arm" and not args.no_halves  # head pitch isn't measured
            events = selector.update(t, aim)
            if roi:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)
            if mirror:
                frame = cv2.flip(frame, 1)

            sm = selector.aim()
            if osc:
                osc.calib = (*cam.position, *cam.aim, bearing, tilt, cfg["room"]["eye_height_m"],
                             args.split, pitch_offset)
                osc.frame(t, sm, events, selector.highlight, selector.dwell(t), selector._selected)
            for kind, scr in events:
                print(f"{t:8.2f}s  {kind:9s} {scr}")

            # Overlay.
            hl = selector.highlight
            lines = [
                f"select by: {mode.upper()}   ({'a' if mode == 'head' else 'h'} to switch)   {fps:4.1f} fps",
                f"arm : {f'{arm.yaw:+6.1f} yaw {arm.pitch:+6.1f} pitch  ({arm.source})' if arm else 'not pointing'}",
                f"head: {f'{head.yaw:+6.1f} yaw' if head else 'not found'}",
                f"aim : {f'{sm[0]:+6.1f} yaw {sm[1]:+6.1f} pitch' if sm else '-'}",
            ]
            d = mapper.last.get("delta") if sm else None
            if d is not None:
                lines.append(f"half: {mapper.last['row']}, {d:+5.1f} deg from the divider (dead band +/-{args.band / 2:.1f})")
            elif mapper.halves:
                lines.append("half: -")
            else:
                lines.append("half: off" + (" (head mode has no pitch)" if mode == "head" else ""))
            lines.append(f"camera: bearing {bearing:+.1f}, tilt {tilt:+.1f}  ({args.camera_config.name if src_note == '' else 'command line'})")
            if pitch_offset:
                lines.append(f"pitch calibration: {pitch_offset:+.1f} deg  (x clears)")
            box = frame[0:26 * len(lines) + 16, 0:min(W, 760)]
            box[:] = (box * 0.45).astype(box.dtype)  # darken behind the text
            for i, txt in enumerate(lines):
                cv2.putText(frame, txt, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                            cv2.LINE_AA)
            if hl and "/" in hl:
                label = "{} SCREEN - {}".format(*(part.upper() for part in hl.split("/")))
            else:
                label = f"{hl.upper()} SCREEN" if hl else "no screen"
            if selector._selected:
                label += "  - SELECTED"
            cv2.putText(frame, label, (12, H - 24), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(frame, label, (12, H - 24), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOURS[screen_part(hl)], 2)
            raw = aim.yaw if aim else None
            draw_map(frame, cfg, sm, hl, selector.dwell(t), raw, cam if src_note == "" else None)
            draw_side(frame, cfg, mapper, sm, hl, aim.pitch if (aim and mode == "arm") else None)

            # Calibration countdown, then a short measurement.
            if calib_start is not None:
                left = args.calib_delay - (t - calib_start)
                if left > 0:
                    big, small = str(math.ceil(left)), "Step into place and point at the middle screen"
                elif left > -CALIB_MEASURE_S:
                    big, small = "HOLD", "Measuring..."
                    if arm:
                        calib_samples.append(arm.pitch - pitch_offset)
                else:
                    if calib_samples:
                        target = mapper.midline_pitch("front", rg.hit(cfg, 0.0, 0.0)["point"])
                        pitch_offset = target - float(np.median(calib_samples))
                        notice = f"Calibrated: pitch offset {pitch_offset:+.1f} deg"
                        print(f"pitch calibrated: --pitch-offset {pitch_offset:.1f}")
                    else:
                        notice = "No pointing arm seen. Press c to try again."
                        print("pitch calibration failed: no pointing arm seen")
                    notice_until, calib_start = t + 3.0, None
                if calib_start is not None:
                    shade = frame[H // 2 - 110:H // 2 + 70, :]
                    shade[:] = (shade * 0.4).astype(shade.dtype)
                    (bw, _), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 4, 8)
                    cv2.putText(frame, big, ((W - bw) // 2, H // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 4,
                                (255, 255, 255), 8, cv2.LINE_AA)
                    (sw, _), _ = cv2.getTextSize(small, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.putText(frame, small, ((W - sw) // 2, H // 2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2, cv2.LINE_AA)
            elif t < notice_until:
                (nw, _), _ = cv2.getTextSize(notice, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(frame, notice, ((W - nw) // 2, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6)
                cv2.putText(frame, notice, ((W - nw) // 2, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)

            if log_w:
                log_w.writerow([f"{t:.3f}", f"{fps:.1f}", int(bool(res.pose_world_landmarks)),
                                f"{arm.yaw:.2f}" if arm else "", f"{arm.pitch:.2f}" if arm else "",
                                arm.source if arm else "", f"{head.yaw:.2f}" if head else "",
                                f"{sm[0]:.2f}" if sm else "", f"{sm[1]:.2f}" if sm else "",
                                f"{d:.2f}" if d is not None else "", hl or "", ";".join(f"{k}:{s}" for k, s in events)])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                mode = "arm"
            elif key == ord("h"):
                mode = "head"
            elif key == ord("c"):
                if calib_start is None:
                    calib_start, calib_samples = t, []
                    print(f"calibration in {args.calib_delay:g} s: step into place and point at the middle screen")
                else:
                    calib_start, notice, notice_until = None, "Calibration cancelled", t + 2.0
            elif key == ord("x"):
                pitch_offset = 0.0
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("r"):
                x, y, w, h = cv2.selectROI(win, frame, showCrosshair=False)  # frame is as displayed
                if w > 0 and h > 0:
                    if mirror:
                        x = W - x - w
                    roi = (x / W, y / H, (x + w) / W, (y + h) / H)
                    print(f"zone set: --roi {roi[0]:.3f},{roi[1]:.3f},{roi[2]:.3f},{roi[3]:.3f}")
                else:
                    roi = None
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        if osc:
            osc.close()
        if log_f:
            log_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

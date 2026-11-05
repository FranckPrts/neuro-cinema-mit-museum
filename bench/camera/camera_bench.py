#!/usr/bin/env python3
"""Camera bench: pick a screen by pointing (or by head direction) with a webcam.

First attempt at replacing the Muse IMU for screen selection. MediaPipe Pose
finds the body, the eye-to-fingertip ray of a raised arm is turned into the
room frame, and viewer/room_geometry.py says which wall screen it hits.
Holding on one screen for --dwell seconds selects it.

    python bench/camera/camera_bench.py                 # laptop webcam, camera in front
    python bench/camera/camera_bench.py --bearing 29.6  # camera in the front-right corner
    python bench/camera/camera_bench.py --select head   # compare: head direction

Keys: q / Esc quit, a = arm, h = head, m = mirror view, r = draw zone (ROI).
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
from pointing import Selector, arm_aim, head_aim  # noqa: E402

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


def screen_namer(cfg):
    """yaw/pitch -> 'left' | 'middle' | 'right' | None, using the room geometry."""
    names = {"front": "middle", "left": "left", "right": "right"}

    def screen_of(yaw: float, pitch: float):
        h = rg.hit(cfg, yaw, pitch)
        return names.get(h["surface"]) if h["on_screen"] else None
    return screen_of


def parse_roi(s: str | None):
    if not s:
        return None
    x0, y0, x1, y1 = (float(v) for v in s.split(","))
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError("--roi wants x0,y0,x1,y1 as fractions of the frame, x0<x1, y0<y1")
    return (x0, y0, x1, y1)


def draw_map(img, cfg, aim_yaw_pitch, highlight, dwell, raw_yaw=None, size=220, pad=12):
    """Top-down room in the bottom-right corner: screens, visitor, aim ray."""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index, or a video file to replay")
    ap.add_argument("--select", choices=("arm", "head"), default="arm", help="what drives the selection")
    ap.add_argument("--bearing", type=float, default=0.0,
                    help="camera direction seen from the standing spot, deg (+ = visitor's right; "
                         "0 = in front, ~29.6 = front-right corner)")
    ap.add_argument("--tilt", type=float, default=0.0, help="how far the camera looks down, deg")
    ap.add_argument("--roi", type=parse_roi, default=None,
                    help="only look inside this box: x0,y0,x1,y1 as fractions of the frame (the standing zone)")
    ap.add_argument("--model", choices=("lite", "full", "heavy"), default="full")
    ap.add_argument("--room", type=Path, default=REPO / "config" / "room.toml")
    ap.add_argument("--dwell", type=float, default=1.5, help="seconds on one screen to select it")
    ap.add_argument("--hold", type=float, default=0.25, help="seconds before a new screen lights up")
    ap.add_argument("--smooth", type=float, default=0.15, help="aim smoothing time constant, s")
    ap.add_argument("--osc", default=None, help="host:port to send /nc/cam/* to (off by default)")
    ap.add_argument("--log", type=Path, default=None, help="write one CSV row per frame here")
    ap.add_argument("--no-mirror", action="store_true", help="show the image unmirrored")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)

    cfg = rg.load_room(args.room)
    selector = Selector(screen_namer(cfg), smooth_s=args.smooth, hold_s=args.hold, dwell_s=args.dwell)

    osc = None
    if args.osc:
        from pythonosc.udp_client import SimpleUDPClient
        host, port = args.osc.rsplit(":", 1)
        osc = SimpleUDPClient(host, int(port))

    log_f = log_w = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t", "fps", "person", "arm_yaw", "arm_pitch", "arm_src", "head_yaw",
                        "aim_yaw", "aim_pitch", "highlight", "event"])

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
                arm = arm_aim(world, args.bearing, args.tilt)
                head = head_aim(world, args.bearing, args.tilt)
                pts = [(int(x0 + l.x * (x1 - x0)), int(y0 + l.y * (y1 - y0))) for l in img_lm]
                for a, b in BONES:
                    cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
                if arm:
                    side = 19 if arm.source.startswith("left") else 20
                    eye = ((pts[2][0] + pts[5][0]) // 2, (pts[2][1] + pts[5][1]) // 2)
                    cv2.line(frame, eye, pts[side], (0, 255, 255), 3)
            aim = arm if mode == "arm" else head
            events = selector.update(t, aim)
            if roi:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)
            if mirror:
                frame = cv2.flip(frame, 1)

            sm = selector.aim()
            if osc:
                if sm:
                    osc.send_message("/nc/cam/aim", [float(sm[0]), float(sm[1])])
                for kind, scr in events:
                    osc.send_message(f"/nc/cam/{kind}", scr or "none")
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
            box = frame[0:26 * len(lines) + 16, 0:620]
            box[:] = (box * 0.45).astype(box.dtype)  # darken behind the text
            for i, txt in enumerate(lines):
                cv2.putText(frame, txt, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                            cv2.LINE_AA)
            label = f"{hl.upper()} SCREEN" if hl else "no screen"
            if selector._selected:
                label += "  - SELECTED"
            cv2.putText(frame, label, (12, H - 24), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(frame, label, (12, H - 24), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOURS[hl], 2)
            raw = aim.yaw if aim else None
            draw_map(frame, cfg, sm, hl, selector.dwell(t), raw)
            cv2.imshow(win, frame)

            if log_w:
                log_w.writerow([f"{t:.3f}", f"{fps:.1f}", int(bool(res.pose_world_landmarks)),
                                f"{arm.yaw:.2f}" if arm else "", f"{arm.pitch:.2f}" if arm else "",
                                arm.source if arm else "", f"{head.yaw:.2f}" if head else "",
                                f"{sm[0]:.2f}" if sm else "", f"{sm[1]:.2f}" if sm else "",
                                hl or "", ";".join(f"{k}:{s}" for k, s in events)])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                mode = "arm"
            elif key == ord("h"):
                mode = "head"
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
        if log_f:
            log_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

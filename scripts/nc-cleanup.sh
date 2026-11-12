#!/usr/bin/env bash
# Stop everything this repo starts (engine, camera bench, viewer, monitor,
# muselsl) so the OSC ports are free again, then show who still holds them.
# Leaves TouchDesigner and anything else alone. Used by the VS Code "Try: ..." tasks.
set -u
cd "$(dirname "$0")/.."

PATTERNS=(
  "nc-engine"
  "python -m neurocinema"
  "nc-monitor"
  "viewer/nc_viewer.py"
  "bench/camera/camera_bench.py"
  "muselsl stream"
  "muselsl view"
)

found=0
for p in "${PATTERNS[@]}"; do
  if pgrep -f "$p" >/dev/null; then
    echo "stopping: $p"
    pkill -TERM -f "$p"
    found=1
  fi
done

if [ "$found" = 1 ]; then
  for _ in 1 2 3 4 5 6; do  # up to 3 s for a clean exit (muselsl releases the headset)
    alive=0
    for p in "${PATTERNS[@]}"; do pgrep -f "$p" >/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.5
  done
  for p in "${PATTERNS[@]}"; do
    if pgrep -f "$p" >/dev/null; then
      echo "force-stopping: $p"
      pkill -KILL -f "$p"
    fi
  done
  rm -f run/muselsl.pid
else
  echo "nothing of ours was running"
fi

# 7000 TouchDesigner, 7001 engine commands, 7002 monitor, 7003 viewer, 7004 camera -> engine,
# 8080 viewer page, 8081 status page.
busy=$(lsof +c 0 -nP -iUDP:7000-7004 -iTCP:8080-8081 -sTCP:LISTEN 2>/dev/null | tail -n +2)
if [ -n "$busy" ]; then
  echo "ports still in use (e.g. TouchDesigner on 7000):"
  echo "$busy" | awk '{printf "  %-16s pid %-7s %s\n", $1, $2, $9}'
else
  echo "ports 7000-7004, 8080, 8081 are free"
fi

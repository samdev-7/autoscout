#!/bin/bash
# Wait for the in-flight 960 epoch to finish (so its checkpoint is written),
# then stop that run and warm-start a 640 continuation from it.
cd "$(dirname "$0")/.." && cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
C=runs/detect/out/runs/fullB_960/results.csv
N0=$(wc -l < "$C" 2>/dev/null || echo 1)
echo "[handoff] waiting for 960 epoch boundary (rows now: $((N0-1)))"
for i in $(seq 1 120); do
  sleep 30
  N=$(wc -l < "$C" 2>/dev/null || echo 1)
  if [ "$N" -gt "$N0" ]; then echo "[handoff] epoch completed (rows: $((N-1)))"; break; fi
done
sleep 20                      # let last.pt/best.pt finish writing
pkill -f fullrun.py; sleep 5
echo "[handoff] 960 run stopped. checkpoints:"
ls -1 runs/detect/out/runs/fullB_960/weights/
tail -3 "$C" | cut -d, -f1,6,7,8,9
echo "[handoff] launching 640 warm-start"
.venv/bin/python -u src/cont640.py 2>&1 | grep -v "^objc\[" | tail -120

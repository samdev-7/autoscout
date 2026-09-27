#!/bin/zsh
cd "$(dirname "$0")/.." && cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
for C in 0.15 0.20 0.25; do
  W=runs/detect/out/runs/fullB_960/weights/best.pt O=out/d960_$C.npz IMGSZ=960 CONF=$C \
    .venv/bin/python src/detect_match.py 2>&1 | grep mean
done

#!/bin/bash
cd "$(dirname "$0")/.." && cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
caffeinate -i -w $$ &
.venv/bin/python -u src/fullrun.py 2>&1 | grep -v "^objc\[" | tail -200
echo "===== FULL RUN EXITED ====="; cat out/STATUS.md 2>/dev/null | head -30

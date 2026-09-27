#!/bin/bash
# Overnight watcher: survives training failure, always leaves a status file.
cd "$(dirname "$0")/.." && cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ST=out/OVERNIGHT_STATUS.txt
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a out/watch.log; }
: > out/watch.log
log "watcher started; training pid: $(pgrep -f train_ext.py | head -1)"

# keep the Mac awake while training runs, else everything suspends overnight
TPID=$(pgrep -f train_ext.py | head -1)
[ -n "$TPID" ] && { caffeinate -i -w "$TPID" & log "caffeinate holding on pid $TPID"; }

HB=0
while pgrep -f train_ext.py > /dev/null; do
  sleep 120; HB=$((HB+1))
  if [ $((HB % 15)) -eq 0 ]; then
    log "still training: $(grep -aoE '[0-9]+/60 ' out/train.log | tail -1) $(grep -aoE '[0-9.]+s/it' out/train.log | tail -1)"
  fi
done
log "training process exited"

{
  echo "=== OVERNIGHT RUN $(date) ==="
  echo "--- last epoch rows ---"
  grep -aE "^ +all " out/train.log | tail -5
  grep -aiE "WALL|epochs completed|early stopping|Optimizer stripped" out/train.log | tail -5
} > "$ST"

for attempt in 1 2; do
  log "evaluation attempt $attempt"
  if .venv/bin/python src/evaluate.py > out/eval_run.log 2>&1; then
    log "evaluation OK"; { echo "--- EVALUATION ---"; grep -v "^objc\[" out/eval_run.log; } >> "$ST"
    break
  else
    log "evaluation FAILED (exit $?)"
    { echo "--- EVALUATION FAILED attempt $attempt ---"; tail -25 out/eval_run.log; } >> "$ST"
    sleep 20
  fi
done
log "watcher done"
echo "=================== OVERNIGHT SUMMARY ==================="
cat "$ST"

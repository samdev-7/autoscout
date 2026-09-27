#!/bin/bash
cd "$(dirname "$0")/.." && cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
: > out/rate.log
while pgrep -f train_ext.py > /dev/null; do
  R=$(grep -aoE '[0-9.]+s/it' out/train.log | tail -1)
  E=$(grep -aoE '[0-9]+/60 ' out/train.log | tail -1 | tr -d ' ')
  T=$(pmset -g therm 2>/dev/null | grep -oE 'CPU_Speed_Limit *= *[0-9]+' | grep -oE '[0-9]+$')
  echo "$(date '+%H:%M:%S') epoch=$E rate=${R:-?} cpu_speed_limit=${T:-na}" >> out/rate.log
  sleep 120
done

#!/bin/zsh
cd "$(dirname "$0")/.." && cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
for C in 0.10 0.15 0.20 0.25; do
  [ -f out/d960_$C.npz ] || continue
  .venv/bin/python -c "
import numpy as np,sys; sys.path.insert(0,'src'); import geom
d=np.load('out/d960_$C.npz',allow_pickle=True)['dets']
np.save('out/xy960_$C.npy',geom.box_to_field(d[:,3:7]))" 2>/dev/null
  DETS=out/d960_$C.npz OUT=out/app960_$C.npz .venv/bin/python src/appear.py >/dev/null 2>&1
  DETS=out/d960_$C.npz XY=out/xy960_$C.npy APP=out/app960_$C.npz OUT=out/tracks_$C.npy \
    .venv/bin/python src/run_tracks.py >/dev/null 2>&1
  R=$(TRK=out/tracks_$C.npy .venv/bin/python src/eval_tracks.py 2>/dev/null | grep "1.0 m" | awk '{print $4, $5}')
  N=$(.venv/bin/python -c "import numpy as np;print(f\"{len(np.load('out/d960_$C.npz',allow_pickle=True)['dets'])/4921:.2f}\")")
  echo "conf $C  $N det/frame  ->  recall/precision $R"
done

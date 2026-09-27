"""Extract band-cropped frames at a fixed step for the tracing UI.

Step is a compromise: dense enough that a robot cannot be confused between
consecutive samples (at 5 Hz a robot moves under a metre), sparse enough that
tracing a whole match is not thousands of clicks.
"""
import av, cv2, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import BAND0, BAND1

STEP = int(os.environ.get("STEP", "6"))          # frames between samples (5 Hz)
T0 = float(os.environ.get("T0", "8"))
T1 = float(os.environ.get("T1", "172"))

os.makedirs("out/trace_frames", exist_ok=True)
for f in os.listdir("out/trace_frames"):
    os.remove(f"out/trace_frames/{f}")
c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
man, n = [], 0
for fr in c.decode(s):
    if fr.time is None or fr.time < T0:
        continue
    if fr.time > T1:
        break
    if n % STEP == 0:
        im = cv2.cvtColor(fr.to_ndarray(format="rgb24")[BAND0:BAND1],
                          cv2.COLOR_RGB2BGR)
        name = f"t{n:05d}.jpg"
        cv2.imwrite(f"out/trace_frames/{name}", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        man.append({"file": name, "f": n, "t": round(float(fr.time), 3)})
    n += 1
c.close()
json.dump({"step": STEP, "band": [BAND0, BAND1], "w": 1920, "h": BAND1 - BAND0,
           "frames": man}, open("out/trace_manifest.json", "w"))
sz = sum(os.path.getsize(f"out/trace_frames/{m['file']}") for m in man) / 2**20
print(f"{len(man)} frames at step {STEP} ({30/STEP:.1f} Hz), {sz:.0f} MiB, "
      f"t={man[0]['t']}..{man[-1]['t']}s")

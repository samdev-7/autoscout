"""Extract time-aligned frames from all three panels for cross-view pairing.

Same broadcast frame, so the three panels are synchronised by construction --
no clock alignment needed, which is the awkward part of a real multi-camera rig.
"""
import av, cv2, os, json

PAN = {"wide": (0, 490, 0, 1920), "red_station": (540, 1080, 0, 960),
       "blue_station": (540, 1080, 960, 1920)}
STEP = int(os.environ.get("STEP", "30"))       # 1 Hz
T0, T1 = 7.25, 172.0
for k in PAN:
    os.makedirs(f"out/pair/{k}", exist_ok=True)
    for f in os.listdir(f"out/pair/{k}"):
        os.remove(f"out/pair/{k}/{f}")
c = av.open("data/match1_qual.mp4"); s = c.streams.video[0]
s.codec_context.options = {"hwaccel": "videotoolbox"}; s.thread_type = "AUTO"
man, n = [], 0
for f in c.decode(s):
    if f.time is None or f.time < T0:
        continue
    if f.time > T1:
        break
    if n % STEP == 0:
        im = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
        name = f"p{len(man):04d}.jpg"
        for k, (y0, y1, x0, x1) in PAN.items():
            cv2.imwrite(f"out/pair/{k}/{name}", im[y0:y1, x0:x1],
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        man.append({"file": name, "t": round(float(f.time), 2)})
    n += 1
c.close()
json.dump({"frames": man, "step": STEP}, open("out/pair_manifest.json", "w"))
sz = sum(os.path.getsize(f"out/pair/{k}/{m['file']}") for k in PAN for m in man) / 2**20
print(f"{len(man)} time-aligned triples ({30/STEP:.0f} Hz), {sz:.0f} MiB, "
      f"t={man[0]['t']}..{man[-1]['t']}s")

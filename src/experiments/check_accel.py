"""Confirm the accelerator is real: device present, work lands on it, and it's faster."""
import time, torch

def bench(dev, n, size=4096):
    x = torch.randn(size, size, device=dev)
    sync = (torch.cuda.synchronize if dev.startswith("cuda")
            else torch.mps.synchronize if dev == "mps" else (lambda: None))
    for _ in range(3): y = x @ x          # warmup
    sync(); t0 = time.time()
    for _ in range(n): y = x @ x
    sync()
    return (time.time() - t0) / n

print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    dev = "cuda"
    print(f"CUDA build {torch.version.cuda} | devices {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
elif torch.backends.mps.is_available():
    dev = "mps"; print("MPS available (Apple GPU), built:", torch.backends.mps.is_built())
else:
    raise SystemExit("NO ACCELERATOR — this would train on CPU")

g = bench(dev, 50); c = bench("cpu", 3)
# Apple CPUs have AMX matrix units, so fp32 GEMM on CPU is strong: expect a modest
# MPS ratio. On CUDA a small ratio really would mean something is wrong.
floor = 5.0 if dev.startswith("cuda") else 1.3
print(f"\n4096x4096 matmul   {dev}: {g*1e3:7.1f} ms   cpu: {c*1e3:7.1f} ms   "
      f"speedup {c/g:5.1f}x   (expect >{floor:.1f}x on {dev})")
print("  OK" if c/g >= floor else "  WARNING: below expected — check the accelerator is really used")

# tensors must physically live on the device
t = torch.ones(8, device=dev)
print(f"tensor device: {t.device}   (must not say cpu)")
if dev == "cuda":
    print(f"GPU mem allocated: {torch.cuda.memory_allocated()/2**20:.0f} MiB")
elif dev == "mps":
    print(f"MPS mem allocated: {torch.mps.current_allocated_memory()/2**20:.0f} MiB")

# mixed precision must actually engage
with torch.autocast(dev if dev != "mps" else "mps", dtype=torch.float16):
    out = torch.randn(256, 256, device=dev) @ torch.randn(256, 256, device=dev)
print(f"autocast fp16 output dtype: {out.dtype}   (float16 => AMP active)")

# and the real thing: a YOLO forward pass on the device
try:
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    m = YOLO("yolo11s.pt")
    im = torch.zeros(1, 3, 640, 640)
    t0 = time.time(); m.predict(im, device=dev, verbose=False); a = time.time() - t0
    t0 = time.time(); m.predict(im, device=dev, verbose=False); b = time.time() - t0
    print(f"YOLO forward on {dev}: {b*1e3:.1f} ms (warm)")
except Exception as e:
    print("YOLO check skipped:", type(e).__name__)

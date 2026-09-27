"""Isolate MPS compute ceiling from data-loading. Checkpoint params ship frozen."""
import time, torch
from ultralytics import YOLO
def tensors(o,out=None):
    out=[] if out is None else out
    if torch.is_tensor(o):
        if o.is_floating_point(): out.append(o)
    elif isinstance(o,dict):
        for v in o.values(): tensors(v,out)
    elif isinstance(o,(list,tuple)):
        for v in o: tensors(v,out)
    return out
dev="mps"
m=YOLO("yolo11s.pt").model.to(dev).train()
for p in m.parameters(): p.requires_grad_(True)      # <- checkpoint ships frozen
opt=torch.optim.AdamW(m.parameters(),lr=1e-6)
def bench(bs,imgsz,det,amp,n=10):
    torch.use_deterministic_algorithms(det,warn_only=True)
    x=torch.randn(bs,3,imgsz,imgsz,device=dev); t0=None
    for i in range(n):
        if i==3: torch.mps.synchronize(); t0=time.time()
        with torch.autocast("mps",dtype=torch.float16,enabled=amp):
            y=m(x)
        loss=sum(t.float().pow(2).mean() for t in tensors(y))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize(); return bs*(n-3)/(time.time()-t0)
print(f"{'config':50}{'img/s':>9}")
res={}
for det in (True,False):
    for amp in (False,True):
        try:
            r=bench(16,640,det,amp); res[(det,amp)]=r
            print(f"  batch16 640  deterministic={str(det):5} amp={str(amp):5}{r:9.1f}")
        except Exception as e: print(f"  batch16 640  det={det} amp={amp}  FAIL {type(e).__name__}: {e}"[:110])
for bs in (24,32):
    try: print(f"  batch{bs} 640  deterministic=False amp=False {bench(bs,640,False,False):11.1f}")
    except Exception as e: print(f"  batch{bs}  FAIL {type(e).__name__}")
print(f"\n  observed IN TRAINING: 10.7 img/s (batch16, 1.5 s/it, workers=0)")
b=res.get((True,False))
if b:
    print(f"  pure-GPU at the SAME config: {b:.1f} img/s")
    print(f"  => data loading costs {100*(1-10.7/b):.0f}% of throughput" if b>10.7 else
          "  => GPU-bound, dataloader is not the problem")
    f=res.get((False,False))
    if f: print(f"  deterministic=False gives {f/b:.2f}x")

"""Calibrate a finer hash: must separate 'same frame re-encoded' from
'different frame, same static-camera match'. The 8x8 hash could not."""
import glob,os,cv2,numpy as np,random,collections
random.seed(5)
def dh(path,s):
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    r=cv2.resize(im,(s+1,s),interpolation=cv2.INTER_AREA)
    return np.packbits((r[:,1:]>r[:,:-1]).flatten()).view(np.uint64)
pc=lambda a,b:int(np.bitwise_count(a^b).sum())

frc=sorted(glob.glob("data/rf/frcrobots_v5/train/images/2024mibel_qm11*"))
print(f"same-match frames available: {len(frc)}")
for s,bits in ((8,64),(16,256),(24,576)):
    im=cv2.imread(frc[0]); cv2.imwrite("/tmp/_r.jpg",im,[cv2.IMWRITE_JPEG_QUALITY,45])
    same=pc(dh(frc[0],s),dh("/tmp/_r.jpg",s))
    within=[pc(dh(a,s),dh(b,s)) for a,b in
            [(random.choice(frc),random.choice(frc)) for _ in range(40)]]
    other=sorted(glob.glob("data/rf/grain/*/images/*.*"))
    cross=[pc(dh(random.choice(frc),s),dh(random.choice(other),s)) for _ in range(40)]
    print(f"  {s}x{s} ({bits} bits): re-encoded={same:4d}   same-match-diff-frame "
          f"median={np.median(within):5.0f} min={min(within):4d}   unrelated median={np.median(cross):5.0f}")

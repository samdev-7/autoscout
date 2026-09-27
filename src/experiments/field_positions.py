"""Plot every labelled robot position on the rectified field."""
import json, numpy as np, cv2
LEN,WID=16.541,8.069
FBW,FBH=1848,898
box=cv2.imread("out/field_box.png")
pos=json.load(open("out/label_field_pos.json"))
f2d=lambda X,Y:(int(X/LEN*FBW), int(FBH-Y/WID*FBH))

# density heat map per alliance
heat={c:np.zeros((FBH,FBW),np.float32) for c in "RB"}
for p in pos:
    x,y=f2d(p["X"],p["Y"])
    if 0<=x<FBW and 0<=y<FBH: heat[p["c"]][y,x]+=1
for c in heat: heat[c]=cv2.GaussianBlur(heat[c],(0,0),34)

vis=(box*0.42).astype(np.uint8)
for c,col in (("R",(60,60,255)),("B",(255,150,60))):
    h=heat[c]; h=h/h.max() if h.max()>0 else h
    lay=np.zeros_like(vis,np.float32)
    for i in range(3): lay[...,i]=h*col[i]
    vis=np.clip(vis.astype(np.float32)+lay*1.5,0,255).astype(np.uint8)
for p in pos:
    x,y=f2d(p["X"],p["Y"])
    cv2.circle(vis,(x,y),7,(40,40,255) if p["c"]=="R" else (255,170,60),-1)
    cv2.circle(vis,(x,y),7,(20,20,20),1)
cv2.rectangle(vis,(0,0),(FBW-1,FBH-1),(255,255,255),3)
cv2.putText(vis,f"{len(pos)} labelled robot positions, 83 frames (t=8-172s)",
            (18,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
cv2.imwrite("out/field_positions.jpg",cv2.resize(vis,(1400,680)),[cv2.IMWRITE_JPEG_QUALITY,92])

R=np.array([[p["X"],p["Y"]] for p in pos if p["c"]=="R"])
B=np.array([[p["X"],p["Y"]] for p in pos if p["c"]=="B"])
print(f"red  {len(R):3d} positions   mean X {R[:,0].mean():5.2f}  mean Y {R[:,1].mean():4.2f}")
print(f"blue {len(B):3d} positions   mean X {B[:,0].mean():5.2f}  mean Y {B[:,1].mean():4.2f}")
half=LEN/2
print(f"\ntime share by field half (X<{half:.1f} vs X>{half:.1f}):")
for n,A in (("red",R),("blue",B)):
    lo=(A[:,0]<half).mean()
    print(f"  {n:4}: {100*lo:4.0f}% low-X   {100*(1-lo):4.0f}% high-X")
print("\nwrote out/field_positions.jpg")

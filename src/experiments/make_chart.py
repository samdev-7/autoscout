"""Regenerate the training chart from train.log (validated dataviz palette)."""
import csv, json, subprocess
rows=[r for r in csv.reader(open("out/metrics.csv")) if r]
data=[{"ep":int(r[0]),"R":float(r[1]),"P":float(r[2]),"m50":float(r[3]),"m5095":float(r[4])} for r in rows]
DATA=json.dumps(data)
best50=max(data,key=lambda d:d["m50"]); bestR=max(data,key=lambda d:d["R"])
SUB=(f"YOLO11s on external data only (3,485 imgs). {len(data)} epochs. "
     f"Best mAP50 {best50['m50']:.3f} (ep {best50['ep']}); best recall {bestR['R']:.3f} (ep {bestR['ep']}).")
tpl=open("out/_chart_tpl.html").read() if False else None
html = """<meta charset="utf-8">
<title>autoscout — training progress</title>
<style>
.viz-root{color-scheme:light;
  --surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#78776f;
  --grid:#e6e5e0;--axis:#c9c8c2;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
  background:var(--surface-1);color:var(--text-primary);
  font:13px/1.5 ui-sans-serif,-apple-system,system-ui,sans-serif;padding:18px 20px 22px;margin:0;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
  --text-muted:#8e8d83;--grid:#2e2e2b;--axis:#3d3d39;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8e8d83;
  --grid:#2e2e2b;--axis:#3d3d39;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;}
h1{font-size:15px;margin:0 0 2px;font-weight:600}
p.sub{margin:0 0 14px;color:var(--text-secondary);font-size:12px}
#legend{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 10px}
.lg{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-secondary)}
.sw{width:14px;height:3px;border-radius:2px;flex:none}
svg{display:block;width:100%;height:auto;overflow:visible}
table{border-collapse:collapse;font:11px/1.4 ui-monospace,Menlo,monospace;margin-top:16px;width:100%}
th,td{border-bottom:1px solid var(--grid);padding:3px 8px;text-align:right;color:var(--text-secondary)}
th{color:var(--text-muted);font-weight:600}
td:first-child,th:first-child{text-align:left}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;background:var(--surface-1);
  border:1px solid var(--axis);border-radius:6px;padding:7px 9px;
  font:11px/1.5 ui-monospace,Menlo,monospace;color:var(--text-primary);
  box-shadow:0 4px 14px rgba(0,0,0,.18);z-index:9;white-space:nowrap}
</style><body class="viz-root">
<h1>Training progress — validated on 421 held-out boxes</h1>
<p class="sub">__SUB__ Nothing from this match was trained on: every point is cross-event transfer.</p>
<div id="legend"></div>
<svg id="c" viewBox="0 0 760 340" role="img" aria-label="Training metrics by epoch"></svg>
<div id="tip"></div>
<table id="tbl"><caption style="caption-side:top;text-align:left;color:var(--text-muted);
 font:11px ui-monospace,monospace;padding-bottom:5px">Table view — same data</caption></table>
<script>
const DATA=__DATA__;
const S=[{k:"R",lab:"Recall",v:"--s1"},{k:"P",lab:"Precision",v:"--s2"},
         {k:"m50",lab:"mAP50",v:"--s3"},{k:"m5095",lab:"mAP50-95",v:"--s4"}];
const W=760,H=340,M={t:14,r:96,b:34,l:44},IW=W-M.l-M.r,IH=H-M.t-M.b;
const maxEp=Math.max(...DATA.map(d=>d.ep));
const X=e=>M.l+(maxEp<=1?0:(e-1)/(maxEp-1))*IW, Y=v=>M.t+(1-v)*IH;
let s="";
for(let g=0;g<=10;g+=2){const v=g/10;
 s+=`<line x1="${M.l}" y1="${Y(v)}" x2="${M.l+IW}" y2="${Y(v)}" stroke="var(--grid)"/>`;
 s+=`<text x="${M.l-8}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--text-muted)">${v.toFixed(1)}</text>`;}
s+=`<line x1="${M.l}" y1="${Y(0.85)}" x2="${M.l+IW}" y2="${Y(0.85)}" stroke="var(--text-muted)" stroke-dasharray="5 4" opacity=".7"/>`;
s+=`<text x="${M.l+IW-4}" y="${Y(0.85)-6}" text-anchor="end" font-size="10.5" fill="var(--text-muted)">recall target 0.85</text>`;
for(const d of DATA) if(d.ep%3===1||d.ep===maxEp)
 s+=`<text x="${X(d.ep)}" y="${M.t+IH+18}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${d.ep}</text>`;
s+=`<text x="${M.l+IW/2}" y="${H-2}" text-anchor="middle" font-size="11" fill="var(--text-muted)">epoch</text>`;
s+=`<line x1="${M.l}" y1="${M.t+IH}" x2="${M.l+IW}" y2="${M.t+IH}" stroke="var(--axis)"/>`;
for(const q of S){
 s+=`<polyline points="${DATA.map(d=>X(d.ep)+','+Y(d[q.k])).join(' ')}" fill="none" stroke="var(${q.v})" stroke-width="2" stroke-linejoin="round"/>`;
 for(const d of DATA) s+=`<circle cx="${X(d.ep)}" cy="${Y(d[q.k])}" r="3.4" fill="var(${q.v})" stroke="var(--surface-1)" stroke-width="1.6"/>`;
 const L=DATA[DATA.length-1];
 s+=`<text x="${X(L.ep)+12}" y="${Y(L[q.k])+4}" font-size="11.5" font-weight="600" fill="var(--text-secondary)">${q.lab}</text>`;}
s+=`<line id="ch" x1="0" y1="${M.t}" x2="0" y2="${M.t+IH}" stroke="var(--axis)" opacity="0"/>`;
document.getElementById('c').innerHTML=s;
document.getElementById('legend').innerHTML=S.map(q=>`<span class="lg"><span class="sw" style="background:var(${q.v})"></span>${q.lab}</span>`).join('');
const tip=document.getElementById('tip'),ch=document.getElementById('ch'),c=document.getElementById('c');
c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect(),sx=(e.clientX-r.left)*W/r.width;
 if(sx<M.l-6||sx>M.l+IW+6){tip.style.opacity=0;ch.setAttribute('opacity',0);return;}
 let bd=1e9,b=DATA[0]; for(const d of DATA){const q=Math.abs(X(d.ep)-sx); if(q<bd){bd=q;b=d;}}
 ch.setAttribute('x1',X(b.ep));ch.setAttribute('x2',X(b.ep));ch.setAttribute('opacity',.85);
 tip.innerHTML=`<b>epoch ${b.ep}</b><br>`+S.map(q=>`<span style="color:var(${q.v})">&#9632;</span> ${q.lab} ${b[q.k].toFixed(3)}`).join('<br>');
 tip.style.opacity=1;tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';});
c.addEventListener('mouseleave',()=>{tip.style.opacity=0;ch.setAttribute('opacity',0);});
document.getElementById('tbl').innerHTML='<tr><th>epoch</th>'+S.map(q=>`<th>${q.lab}</th>`).join('')+'</tr>'+
 DATA.map(d=>`<tr><td>${d.ep}</td>`+S.map(q=>`<td>${d[q.k].toFixed(3)}</td>`).join('')+'</tr>').join('');
</script>"""
open("out/training_chart.html","w").write(html.replace("__DATA__",DATA).replace("__SUB__",SUB))
print(f"wrote chart: {len(data)} epochs; best mAP50 {best50['m50']:.3f} ep{best50['ep']}, "
      f"best recall {bestR['R']:.3f} ep{bestR['ep']}")

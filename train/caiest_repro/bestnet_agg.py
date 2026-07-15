"""bestnet_agg.py — aggregate the BESTNET campaign: gate cells (vs aug_s0) + TLE + val ->
BESTNET_RESULTS.json + BESTNET_WRITEUP.md. beats_augs0 iff placement 95% CI lower bound > 2.500
(calibrated: aug_s0-vs-aug_s0 must read 2.500). tle_safe iff worst_ms_over_300 < 1000."""
import json,glob,math,os
HERE=os.path.dirname(os.path.abspath(__file__))
GD=os.path.join(HERE,"ckpt","best","gates"); TD=os.path.join(HERE,"ckpt","best","tle"); VD=os.path.join(HERE,"ckpt","best","val")
TCRIT={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,30:2.042,40:2.021}
def tcrit(dfn):
  if dfn in TCRIT: return TCRIT[dfn]
  for k in sorted(TCRIT):
    if k>=dfn: return TCRIT[k]
  return 1.96
# tag -> (label, cand_basename, channels, blocks, params_M)
NETS={
 "raw384":("full_384x40_s0 (RAW converged)","full_384x40_s0.pkl",384,40,113.56),
 "raw192":("big192x40_s0 (RAW)","big192x40_s0_fused.pkl",192,40,30.30),
 "enh384_s0":("enh 384x40 s0 (e11 recipe)","enh_384x40_s0.pkl",384,40,113.56),
 "enh384_s1":("enh 384x40 s1 (e11 recipe)","enh_384x40_s1.pkl",384,40,113.56),
 "enh192_s0":("enh 192x40 s0 (e11 recipe)","enh_192x40_s0.pkl",192,40,30.30),
 "enh192_s1":("enh 192x40 s1 (e11 recipe)","enh_192x40_s1.pkl",192,40,30.30),
}
def gate_agg(tag):
  cells=sorted(glob.glob(os.path.join(GD,tag+"_s*.json")))
  vals=[];games=0;first=[];fourth=[];secs=0
  for c in cells:
    d=json.load(open(c)); vals.append(d["placement_pts"]); games+=d["games"]
    first.append(d["first_pct"]); fourth.append(d["fourth_pct"]); secs+=d.get("seconds",0)
  n=len(vals)
  if n==0: return None
  mean=sum(vals)/n
  if n>1:
    sd=math.sqrt(sum((v-mean)**2 for v in vals)/(n-1)); se=sd/math.sqrt(n); ci=tcrit(n-1)*se
  else: sd=se=ci=0.0
  lo,hi=mean-ci,mean+ci
  return dict(n_blocks=n,total_games=games,placement_mean=round(mean,4),placement_sd=round(sd,4),
    placement_se=round(se,4),ci95_lo=round(lo,4),ci95_hi=round(hi,4),margin_lo=round(lo-2.5,4),
    first_pct=round(sum(first)/n,2),fourth_pct=round(sum(fourth)/n,2),gate_seconds=round(secs,1),
    block_placements=vals)
def load(d,f):
  p=os.path.join(d,f)
  return json.load(open(p)) if os.path.exists(p) else {}
calib=gate_agg("calib")
nets={}
for tag,(label,cb,ch,blk,pm) in NETS.items():
  g=gate_agg(tag); tle=load(TD,tag+".json"); val=load(VD,tag+".json")
  entry={"label":label,"cand":cb,"channels":ch,"blocks":blk,"params_M":pm,
         "val_acc":val.get("val_acc"),
         "mean_ms_per_move":tle.get("mean_ms_per_move"),"worst_ms_over_300":tle.get("worst_ms_over_300"),
         "budget_ms":1000}
  entry["tle_safe"]=(tle.get("worst_ms_over_300") is not None and tle["worst_ms_over_300"]<1000)
  if g:
    entry.update(g)
    entry["beats_augs0"]=bool(g["ci95_lo"]>2.5)
    entry["gate_verdict"]=("BEATS_AUGS0" if g["ci95_lo"]>2.5 else ("WORSE" if g["ci95_hi"]<2.5 else "TIED_NOT_SEPARATED"))
  else:
    entry["beats_augs0"]=None; entry["gate_verdict"]="PENDING"
  nets[tag]=entry
# winner: beats aug_s0 (CI-separated) AND tle_safe
elig={k:v for k,v in nets.items() if v.get("beats_augs0") and v.get("tle_safe")}
if elig:
  best=max(elig,key=lambda k:nets[k]["ci95_lo"])
  overall=(f"NEW BEST DEPLOYABLE: {best} ({nets[best][label]}) CI-separated above aug_s0 "
    f"(margin_lo=+{nets[best][margin_lo]}) AND TLE-clean (worst {nets[best][worst_ms_over_300]}ms < 1000ms).")
else:
  done=[k for k,v in nets.items() if v.get("gate_verdict") not in (None,"PENDING")]
  overall=("NO net is BOTH CI-separated-better than aug_s0 AND TLE-clean among completed gates "
    f"({sorted(done)}) -> aug_s0 STAYS the best deployable entry.")
out=dict(experiment="BESTNET: enhanced big nets + raw big nets vs aug_s0 (current best deployable 128x40)",
  reference="aug_s0 = ckpt/aug/aug_128x40_s0.pkl (128x40); calibrated gate, 2.500=tied; BEAT iff 95% CI lo>2.500",
  tle_budget_ms=1000, calibration_augs0_vs_augs0=calib, nets=nets, overall_verdict=overall)
json.dump(out,open(os.path.join(HERE,"BESTNET_RESULTS.json"),"w"),indent=2)
# writeup
L=["# BESTNET — best deployable base net (enhanced/raw big nets vs aug_s0)\n"]
L.append("Reference = **aug_s0** (`ckpt/aug/aug_128x40_s0.pkl`, 128x40, the current best deployable). ")
L.append("Gate = `e11_gate.py` lam=0 calibrated duplicate placement gate; 2.500 = tied with aug_s0; ")
L.append("**BEAT iff 95% CI lower bound > 2.500**. TLE = per-move CPU single-thread latency; budget ~1000 ms/move.\n")
if calib: L.append(f"Calibration (aug_s0 vs aug_s0): **{calib[placement_mean]}** over {calib[n_blocks]} block(s) — must read 2.500.\n")
L.append("| net | params | val_acc | per-move ms (mean/worst) | TLE-safe | placement vs aug_s0 | 95% CI | margin_lo | beats aug_s0 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for tag,v in nets.items():
  ms=f"{v.get(mean_ms_per_move)}/{v.get(worst_ms_over_300)}" if v.get(mean_ms_per_move) else "-"
  pl=v.get("placement_mean","-"); ci=f"[{v.get(ci95_lo)}, {v.get(ci95_hi)}]" if v.get("ci95_lo") is not None else "-"
  ml=(f"{v[margin_lo]:+}" if v.get("margin_lo") is not None else "-")
  L.append(f"| {tag} ({v[label]}) | {v[params_M]}M | {v.get(val_acc)} | {ms} | {v.get(tle_safe)} | {pl} | {ci} | {ml} | {v.get(beats_augs0)} |")
L.append("\n## VERDICT\n"); L.append(overall+"\n")
open(os.path.join(HERE,"BESTNET_WRITEUP.md"),"w").write("\n".join(L))
print("OVERALL:",overall)

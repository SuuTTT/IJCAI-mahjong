"""Aggregate high-power BN128 confirm cells -> BN128S1_CONFIRM.json. Honest t-based 95% CI.
Every number read from saved per-cell JSON. Rule: beats distill ONLY if CI lower bound > 2.500."""
import json, glob, math, os
D = "ckpt/bn128s1"
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
 11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086}
def agg(prefix):
    cells = sorted(glob.glob(os.path.join(D, prefix+"_s*.json")))
    blocks=[]; vals=[]; games=0
    for c in cells:
        d=json.load(open(c))
        vals.append(d["placement_pts"]); games+=d["games"]
        blocks.append({"seed0":d["seed0"],"placement_pts":d["placement_pts"],
                       "first_pct":d["first_pct"],"fourth_pct":d["fourth_pct"],
                       "games":d["games"],"seeds":d["seeds"]})
    n=len(vals)
    if n==0: return None
    mean=sum(vals)/n
    if n>1:
        var=sum((v-mean)**2 for v in vals)/(n-1); std=math.sqrt(var)
        se=std/math.sqrt(n); t=TCRIT.get(n-1,2.0); ci=t*se
    else:
        std=0.0; ci=0.0
    return dict(prefix=prefix,n_blocks=n,total_games=games,mean=round(mean,4),std=round(std,4),
                ci95_halfwidth=round(ci,4),ci95_lo=round(mean-ci,4),ci95_hi=round(mean+ci,4),
                beats_distill=bool(mean-ci>2.500),blocks=blocks)
cand=agg("full_128x40_s1")
verdict=None
if cand:
    if cand["ci95_lo"]>2.500:
        verdict=("YES_CONFIRMED","full_128x40_s1 CI-SEPARATED above 2.500 -> REAL same-size base-policy upgrade. Safe to deploy (same size as distill, no TLE risk).")
    elif cand["ci95_hi"]<2.500:
        verdict=("NO_WORSE","full_128x40_s1 CI below 2.500 -> worse than distill. Keep distill.")
    else:
        verdict=("NO_NOISE","full_128x40_s1 CI still includes 2.500 -> edge not confirmed at this power. Distill stays unless 384 qualifies.")
out=dict(experiment="BN128S1 high-power confirmatory gate: full_128x40_s1 (resbn_fused 128x40, SAME size as distill) vs deployed distill cnn_lad_chunjiandu",
  gate="e8_gate.py lam=0 calibrated duplicate-format placement gate; 2.500=tied; rule: CI lower bound > 2.500 to beat",
  candidate=cand,verdict_code=(verdict[0] if verdict else "NO_DATA"),
  verdict=(verdict[1] if verdict else "no candidate cells found"))
json.dump(out,open("BN128S1_CONFIRM.json","w"),indent=2)
print(json.dumps({"cand_mean":(cand or {}).get("mean"),"cand_ci":[(cand or {}).get("ci95_lo"),(cand or {}).get("ci95_hi")],
  "cand_n":(cand or {}).get("n_blocks"),"verdict":(verdict[0] if verdict else None)},indent=2))

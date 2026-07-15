"""Aggregate high-power BN384 confirm cells -> BN384_CONFIRM.json. Honest t-based 95% CI.
Reads every number from saved per-cell JSON. Rule: a net beats distill ONLY if CI lower bound > 2.500."""
import json, glob, math, os
D = "ckpt/bn384"
# two-sided 95% t critical by dof (1..30)
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
 11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
 21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}

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
    lo=mean-ci; hi=mean+ci
    return dict(prefix=prefix, n_blocks=n, total_games=games,
                mean=round(mean,4), std=round(std,4),
                ci95_halfwidth=round(ci,4), ci95_lo=round(lo,4), ci95_hi=round(hi,4),
                beats_distill=bool(lo>2.500), blocks=blocks)

cand=agg("full_384x40_s0")
calib=agg("calib_distill")
big192=agg("big192x40_s0_fused")
calib_ok = calib is not None and abs(calib["mean"]-2.5)<0.003
verdict=None
if cand:
    if cand["ci95_lo"]>2.500:
        verdict=("YES_CONFIRMED",
          "full_384x40_s0 CI-SEPARATED above 2.500 -> REAL base-policy upgrade. Deploy full_384x40_s0 (raw policy net) under moyu Botzone account like distill_cs2; value model not needed.")
    elif cand["ci95_hi"]<2.500:
        verdict=("NO_WORSE","full_384x40_s0 CI below 2.500 -> worse than distill. Keep distill.")
    else:
        verdict=("NO_NOISE","full_384x40_s0 CI still includes 2.500 -> edge was noise. Distill stays.")
out=dict(
  experiment="BN384 high-power confirmatory gate: full_384x40_s0 (resbn_fused 384x40) vs deployed distill cnn_lad_chunjiandu",
  gate="e8_gate.py lam=0 calibrated duplicate-format placement gate; 2.500=tied with distill; rule: CI lower bound > 2.500 to beat",
  calibration_check=dict(calib=calib, calib_ok=bool(calib_ok)),
  candidate=cand,
  big192_optional=big192,
  verdict_code=(verdict[0] if verdict else "NO_DATA"),
  verdict=(verdict[1] if verdict else "no candidate cells found"))
json.dump(out, open("BN384_CONFIRM.json","w"), indent=2)
print(json.dumps({"calib_mean":(calib or {}).get("mean"),"calib_ok":calib_ok,
  "cand_mean":(cand or {}).get("mean"),"cand_ci":[(cand or {}).get("ci95_lo"),(cand or {}).get("ci95_hi")],
  "cand_n":(cand or {}).get("n_blocks"),"verdict":(verdict[0] if verdict else None),
  "big192_mean":(big192 or {}).get("mean"),"big192_ci":[(big192 or {}).get("ci95_lo"),(big192 or {}).get("ci95_hi")] if big192 else None}, indent=2))

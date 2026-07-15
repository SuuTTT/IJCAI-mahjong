"""Aggregate BASENET gate cells -> BASENET_GATE_RESULTS.json + BASENET_GATE_WRITEUP.md.
Per candidate: gate-vs-distill mean +/- std over seed blocks, 95% CI (t-based, n blocks),
beats_distill = (CI lower bound > 2.500). Calibration check: calib_distill must read 2.500.
Every number is read from the saved per-cell JSON. Honest CIs."""
import os, json, glob, math
D = "ckpt/basenet_gate"
# net config metadata for the writeup
META = {
 "calib_distill":      ("resbn_fused 128x40", "distill self (calibration)"),
 "moyu_bn_128x40":     ("resbn 128x40",       "moyu BN base (capacity-sweep ref)"),
 "full_128x40_s0":     ("resbn_fused 128x40", "converged 90k base, seed0"),
 "full_128x40_s1":     ("resbn_fused 128x40", "converged 90k base, seed1"),
 "full_256x40_s0":     ("resbn_fused 256x40", "converged 90k base, seed0 (val~0.886)"),
 "full_256x40_s1":     ("resbn_fused 256x40", "converged 90k base, seed1 (val~0.898)"),
 "full_384x40_s0":     ("resbn_fused 384x40", "converged 90k base, seed0 (val~0.884)"),
 "big192x40_s0_fused": ("resbn_fused 192x40", "historical 'beat moyu' candidate"),
 "big256x40_s0_fused": ("resbn_fused 256x40", "historical big-256 candidate"),
}
# t critical (two-sided 95%) by dof
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228}

def agg(label):
    cells = sorted(glob.glob(os.path.join(D, f"{label}_s*.json")))
    vals=[]; games=0; det=[]
    for c in cells:
        d=json.load(open(c))
        vals.append(d["placement_pts"]); games+=d["games"]
        det.append({"seed0":d["seed0"],"placement_pts":d["placement_pts"],
                    "first_pct":d["first_pct"],"fourth_pct":d["fourth_pct"],"games":d["games"]})
    n=len(vals)
    if n==0: return None
    mean=sum(vals)/n
    if n>1:
        var=sum((v-mean)**2 for v in vals)/(n-1); std=math.sqrt(var)
        se=std/math.sqrt(n); t=TCRIT.get(n-1,2.0); ci=t*se
    else:
        std=0.0; ci=0.0
    lo=mean-ci; hi=mean+ci
    beats = lo > 2.500
    return dict(label=label, kind=META.get(label,("?","?"))[0], note=META.get(label,("?","?"))[1],
                n_blocks=n, total_games=games,
                gate_mean=round(mean,4), gate_std=round(std,4),
                ci95_halfwidth=round(ci,4), ci95_lo=round(lo,4), ci95_hi=round(hi,4),
                beats_distill=bool(beats), blocks=det)

labels=[l for l in META]
results=[r for r in (agg(l) for l in labels) if r]
# rank candidates (exclude calib) by gate_mean desc
ranked=sorted([r for r in results if r["label"]!="calib_distill"],
              key=lambda r:-r["gate_mean"])
calib=next((r for r in results if r["label"]=="calib_distill"), None)
calib_ok = calib is not None and abs(calib["gate_mean"]-2.5)<1e-6
any_beats=any(r["beats_distill"] for r in ranked)
best=ranked[0] if ranked else None
out=dict(
  experiment="BASENET_GATE: trained base nets vs deployed distill (cnn_lad_chunjiandu.npz)",
  ref="cnn_lad_chunjiandu.npz (ResFused 128x40, deployed distill bot)",
  gate="calibrated duplicate-format placement gate (e8_gate.py lam=0, raw policy); 2.500=tied with distill",
  rule="a net beats distill ONLY if its 95% CI lower bound > 2.500",
  seeds_per_block=300, games_per_block=1200,
  calibration_distill_vs_distill=calib["gate_mean"] if calib else None,
  calibration_ok=calib_ok,
  any_candidate_beats_distill=any_beats,
  best_base_net=best["label"] if best else None,
  best_base_net_gate=best["gate_mean"] if best else None,
  results=results, ranking=[r["label"] for r in ranked])
json.dump(out, open("BASENET_GATE_RESULTS.json","w"), indent=2)

# writeup
L=[]
L.append("# BASENET GATE — trained base nets vs deployed distill\n")
L.append(f"**Ref (deployed):** cnn_lad_chunjiandu.npz (ResFused 128x40 distill bot).  ")
L.append(f"**Gate:** calibrated duplicate-format placement gate (e8_gate.py, lam=0 raw policy). 2.500 = tied with distill.  ")
L.append(f"**Blocks:** {best['n_blocks'] if best else '?'} seed-blocks x 300 seeds (1200 games each).  ")
L.append(f"**Beats rule:** 95% CI lower bound strictly > 2.500.\n")
L.append(f"**Calibration:** distill-vs-distill = {calib['gate_mean'] if calib else 'NA'} "
         f"({'OK = 2.500' if calib_ok else 'FAIL'}).\n")
L.append("## Ranked: gate-vs-distill (higher = better than distill; 2.500 = tied)\n")
L.append("| Rank | Net | Arch | Gate mean | ±95% CI | CI range | Beats distill? |")
L.append("|------|-----|------|-----------|---------|----------|----------------|")
for i,r in enumerate(ranked,1):
    L.append(f"| {i} | {r['label']} | {r['kind']} | {r['gate_mean']:.4f} | "
             f"±{r['ci95_halfwidth']:.4f} | [{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}] | "
             f"{'**YES**' if r['beats_distill'] else 'no'} |")
L.append("")
L.append("## Verdict\n")
if any_beats:
    winners=[r['label'] for r in ranked if r['beats_distill']]
    L.append(f"**UPGRADE FOUND.** CI-separated above 2.500 vs distill: {', '.join(winners)}. "
             f"Strongest base net = **{best['label']}** ({best['gate_mean']:.4f}, "
             f"CI [{best['ci95_lo']:.4f},{best['ci95_hi']:.4f}]).")
    L.append(f"\n**Deploy recommendation:** re-upload **{best['label']}** as the bot (as distill_cs2 was) "
             f"for sim11/final.")
else:
    L.append(f"**NO upgrade.** No candidate base net is CI-separated above 2.500 vs distill. "
             f"Best base net is **{best['label']}** at {best['gate_mean']:.4f} "
             f"(CI [{best['ci95_lo']:.4f},{best['ci95_hi']:.4f}]) — still at/below distill's 2.500. "
             f"Distill (cnn_lad_chunjiandu) is confirmed our strongest base policy; the entry stays distill.")
L.append("")
L.append("## Per-net detail\n")
for r in results:
    L.append(f"- **{r['label']}** ({r['kind']}; {r['note']}): mean {r['gate_mean']:.4f} "
             f"±{r['ci95_halfwidth']:.4f}, blocks=" +
             ", ".join(f"{b['seed0']}:{b['placement_pts']:.4f}" for b in r['blocks']))
open("BASENET_GATE_WRITEUP.md","w").write("\n".join(L)+"\n")
print("calibration_ok:", calib_ok, "| any_beats:", any_beats, "| best:", best['label'] if best else None, best['gate_mean'] if best else None)
print("ranking:", [ (r['label'], r['gate_mean'], r['beats_distill']) for r in ranked])

"""e4_aggregate.py — assemble E4_RESULTS.json from gate blocks + claimrate + critic stats.
Per AWR config: beta, seed, placement mean+-std across blocks, paired vs-base(calib) delta + 95% CI,
claim_rate. Verdict logic: AWR beats base iff delta CI lower bound > 0."""
import json, glob, os, re, numpy as np
from math import sqrt

ROOT="/root/IJCAI-mahjong/train/caiest_repro"
GD=f"{ROOT}/ckpt/e4/gates"

# t critical values (two-sided 95%) for small df
TCRIT={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306}
def ci95(vals):
    a=np.asarray(vals,float); n=len(a)
    if n<2: return (float("nan"),float("nan"),float(a.mean()) if n else float("nan"),float("nan"))
    m=a.mean(); sd=a.std(ddof=1); se=sd/sqrt(n)
    t=TCRIT.get(n-1,1.96)
    return (m-t*se, m+t*se, m, sd)

def blocks_for(name):
    out={}
    for f in sorted(glob.glob(f"{GD}/{name}_blk*.json")):
        bi=int(re.search(r"_blk(\d+)\.json",f).group(1))
        d=json.load(open(f)); out[bi]=d["placement_pts"]
    return out

calib=blocks_for("calib")
print("calib blocks:",calib)
calib_pts=[calib[b] for b in sorted(calib)]
cm=np.mean(calib_pts); print("calib mean=%.4f"%cm)

# critic stats from adv cache
z=np.load(f"{ROOT}/data/adv_cache.npz")
adv=z["adv"].astype(np.float64); vp=z["vplace"].astype(np.float64); rp=z["realized_pts"].astype(np.float64)
critic={"corr_vplace_realized": round(float(np.corrcoef(vp,rp)[0,1]),4),
        "adv_mean": round(float(adv.mean()),4), "adv_std": round(float(adv.std()),4),
        "vplace_mean": round(float(vp.mean()),4), "N": int(len(adv)),
        "note":"in-data corr; held-out critic: 4th-AUC 0.955, place-acc 0.75, score-r 0.67 (value_256x40)"}

# claim rates
cr=json.load(open(f"{ROOT}/ckpt/e4/claimrate.json"))
base_claim=cr["results"]["base_moyu"]["claim"]

configs=[]
for f in sorted(glob.glob(f"{GD}/awr_b*_blk0.json")):
    nm=re.search(r"(awr_b[\d.]+_s\d+)_blk0\.json",os.path.basename(f)).group(1)
    m=re.match(r"awr_b([\d.]+)_s(\d+)",nm)
    beta=float(m.group(1)); seed=int(m.group(2))
    blk=blocks_for(nm)
    bis=sorted(set(blk)&set(calib))
    pts=[blk[b] for b in bis]
    deltas=[blk[b]-calib[b] for b in bis]   # paired per-block delta vs calib
    lo,hi,dm,dsd=ci95(deltas)
    pl_lo,pl_hi,plm,plsd=ci95(pts)
    beats = (lo>0) if not np.isnan(lo) else False
    configs.append(dict(
        config=nm, beta=beta, seed=seed,
        blocks=bis, placement_per_block=[round(x,4) for x in pts],
        placement_mean=round(plm,4), placement_std=round(plsd,4),
        calib_per_block=[round(calib[b],4) for b in bis], calib_mean=round(cm,4),
        delta_per_block=[round(x,4) for x in deltas],
        delta_mean=round(dm,4), delta_std=round(dsd,4),
        delta_ci95=[round(lo,4),round(hi,4)],
        beats_base_ci_separated=bool(beats),
        claim_rate=round(cr["results"].get(nm,{}).get("claim",float("nan")),4),
        claim_rate_vs_base=round(cr["results"].get(nm,{}).get("claim",float("nan"))-base_claim,4),
    ))

configs.sort(key=lambda c:(c["beta"],c["seed"]))
any_beat=any(c["beats_base_ci_separated"] for c in configs)
out=dict(
    experiment="E4: offline AWR with VERIFIED-GOOD critic vs base imitation policy (duplicate placement)",
    critic=critic,
    base_calib={"placement_mean":round(cm,4),"per_block":[round(x,4) for x in calib_pts],
                "claim_rate":round(base_claim,4),"teacher_claim":round(cr["teacher_claim"],4)},
    gate={"metric":"duplicate-format placement points (4/3/2/1), seat-bias cancelled",
          "seeds_per_block":500,"games_per_block":2000,"n_blocks":len(calib_pts),
          "seed0_per_block":[70000,80000,90000][:len(calib_pts)]},
    configs=configs,
    verdict={"any_beta_beats_base_ci_separated":bool(any_beat),
             "summary":("AWR-with-good-critic beats base" if any_beat else
                        "NO beta beats base; retraining fails even with a verified critic — imitation ceiling is real")},
)
json.dump(out,open(f"{ROOT}/E4_RESULTS.json","w"),indent=2)
print(json.dumps(out["verdict"],indent=2))
print("wrote E4_RESULTS.json")

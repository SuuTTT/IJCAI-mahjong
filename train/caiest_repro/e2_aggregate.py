import os, json, glob, math
GD = "/root/IJCAI-mahjong/train/caiest_repro/ckpt/e2/gates"
OUT = "/root/IJCAI-mahjong/train/caiest_repro/E2_RESULTS.json"

def mean(x): return sum(x)/len(x)
def std(x):
    if len(x)<2: return 0.0
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/(len(x)-1))

cells = {}  # (ct,rt) -> list of block dicts
for f in sorted(glob.glob(os.path.join(GD,"ct*_rt*_s*.json"))):
    d = json.load(open(f))
    ct = int(d["claim_tau"]); rt = int(d["ref_tau"])
    cells.setdefault((ct,rt), []).append(d)

cell_out = []
agg = {}  # (ct,rt)-> (mean,std)
for (ct,rt), blocks in sorted(cells.items()):
    pps = [b["placement_pts"] for b in blocks]
    seeds = [b["seed0"] for b in blocks]
    ng = sum(b["games"] for b in blocks)
    rcr_raw = mean([b.get("ref_claim_rate_raw",0.0) for b in blocks])
    rcr_kept = mean([b.get("ref_claim_rate_kept",0.0) for b in blocks])
    m, sd = mean(pps), std(pps)
    agg[(ct,rt)] = (m, sd, pps)
    cell_out.append(dict(cand_tau=ct, ref_tau=rt,
        placement_mean=round(m,4), placement_std=round(sd,4),
        placement_blocks=[round(p,4) for p in pps],
        n_games=ng, seeds=seeds,
        ref_claim_rate_raw=round(rcr_raw,4), ref_claim_rate_kept=round(rcr_kept,4)))

# correction benefit per ref_tau = placement(ct=2) - placement(ct=0)
benefit = []
for rt in sorted({rt for (_,rt) in agg}):
    if (0,rt) in agg and (2,rt) in agg:
        m0,s0,p0 = agg[(0,rt)]; m2,s2,p2 = agg[(2,rt)]
        b = m2 - m0
        # std of difference via paired blocks if same seed0 ordering, else combine
        diffs = [p2[i]-p0[i] for i in range(min(len(p0),len(p2)))]
        bsd = std(diffs) if len(diffs)>=2 else math.sqrt(s0**2+s2**2)
        benefit.append(dict(ref_tau=rt,
            placement_ct0=round(m0,4), placement_ct2=round(m2,4),
            correction_benefit=round(b,4), benefit_std=round(bsd,4),
            ref_claim_rate_kept_ct0=round(mean([d["ref_claim_rate_kept"] for d in cells[(0,rt)]]),4)))

res = dict(
    description="E2: is the tau=2 claim-suppression PLACEMENT benefit opponent-dependent? "
                "Candidate=moyu_bn_128x40; matrix cand_tau{0,2} x ref_tau{0,1,2,3}; 3 wall-seed blocks/cell.",
    candidate="moyu_bn_128x40.pkl", reference="moyu_bn_128x40.pkl",
    seeds_per_block=300, blocks_per_cell=3, games_per_cell="3x300x4",
    cells=cell_out, correction_benefit_by_ref_tau=benefit)
json.dump(res, open(OUT,"w"), indent=2)
print(json.dumps(res, indent=2))

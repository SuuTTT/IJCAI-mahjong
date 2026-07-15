"""
e6_aggregate.py -- pool per-game arrays across wall-seed blocks per cell, compute
DUPLICATE vs SINGLE-GAME metrics, and tau=0 vs tau=2 deltas with significance, per field.

Cells: cand_tau {0,2} x ref_tau {0,2}. For each (ct,rt) we pool pg_raw / pg_rank across
all blocks (N = blocks*seeds*4 games). Then per field rt in {0,2} we compare ct=2 vs ct=0:
  - dup_placement (duplicate metric)        : delta + cross-block paired t (df small)
  - sg_score_mean / std / mean_placement     : Welch t on the pooled per-game arrays
  - sg_first_rate / sg_fourth_rate           : two-proportion z-test
Variance ratio (F) for score_std change. Effect sizes (Cohen's d) for the score mean.
"""
import os, json, glob, math
import numpy as np

GD = "/root/IJCAI-mahjong/train/caiest_repro/ckpt/e6/gates"
OUT = "/root/IJCAI-mahjong/train/caiest_repro/E6_RESULTS.json"


def mean(x): return sum(x)/len(x)
def std(x):
    if len(x) < 2: return 0.0
    m = mean(x); return math.sqrt(sum((v-m)**2 for v in x)/(len(x)-1))


def welch_t(a, b):
    """Welch's t-statistic for mean(a)-mean(b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0, 0.0, 0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return 0.0, ma-mb, na+nb-2
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    num = (va/na + vb/nb)**2
    den = (va/na)**2/(na-1) + (vb/nb)**2/(nb-1)
    df = num/den if den > 0 else na+nb-2
    return t, ma-mb, df


def prop_z(p1, n1, p0, n0):
    """Two-proportion z-test for p1 - p0 (rate at ct=2 minus ct=0)."""
    if n1 == 0 or n0 == 0: return 0.0, 0.0
    x1, x0 = p1*n1, p0*n0
    pbar = (x1 + x0) / (n1 + n0)
    se = math.sqrt(pbar*(1-pbar)*(1/n1 + 1/n0))
    if se == 0: return 0.0, p1-p0
    return (p1-p0)/se, p1-p0


def f_test(a, b):
    """F = var(b)/var(a): does ct=2 (a) reduce variance vs ct=0 (b)? F<1 => ct2 lower var."""
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0: return float("inf"), 1.0
    F = vb / va
    return F, math.sqrt(vb)-math.sqrt(va)  # also raw std delta (ct2-ct0) = -(this)... return F + std-delta sep


# load
cells = {}   # (ct,rt) -> dict(summaries=[...], pg_raw=concat, pg_rank=concat)
for f in sorted(glob.glob(os.path.join(GD, "ct*_rt*_s*.npz"))):
    z = np.load(f, allow_pickle=True)
    s = json.loads(str(z["summary"]))
    ct = int(s["claim_tau"]); rt = int(s["ref_tau"])
    c = cells.setdefault((ct, rt), dict(summaries=[], pg_raw=[], pg_rank=[]))
    c["summaries"].append(s)
    c["pg_raw"].append(z["pg_raw"]); c["pg_rank"].append(z["pg_rank"])

cell_out = {}
for (ct, rt), c in sorted(cells.items()):
    raw = np.concatenate(c["pg_raw"]); rank = np.concatenate(c["pg_rank"])
    S = c["summaries"]
    dup_blocks = [s["dup_placement_pts"] for s in S]
    rec = dict(
        cand_tau=ct, ref_tau=rt, n_blocks=len(S), n_games=int(raw.size),
        seeds=[s["seed0"] for s in S],
        ref_claim_rate_kept=round(mean([s["ref_claim_rate_kept"] for s in S]), 4),
        # duplicate
        dup_placement_mean=round(mean(dup_blocks), 4),
        dup_placement_std=round(std(dup_blocks), 4),
        dup_placement_blocks=[round(x, 4) for x in dup_blocks],
        # single-game (pooled per-game)
        sg_first_rate=round(float(np.mean(rank <= 1.5)), 4),
        sg_fourth_rate=round(float(np.mean(rank >= 3.5)), 4),
        sg_score_mean=round(float(raw.mean()), 4),
        sg_score_std=round(float(raw.std(ddof=1)), 4),
        sg_mean_placement=round(float(rank.mean()), 4),
    )
    cell_out[(ct, rt)] = (rec, raw, rank, dup_blocks)

# tau comparison per field
comparisons = []
for rt in sorted({rt for (_, rt) in cell_out}):
    if (0, rt) not in cell_out or (2, rt) not in cell_out:
        continue
    rec0, raw0, rank0, dup0 = cell_out[(0, rt)]
    rec2, raw2, rank2, dup2 = cell_out[(2, rt)]
    n0, n2 = raw0.size, raw2.size

    # duplicate placement: paired cross-block t (paired by seed0 order)
    k = min(len(dup0), len(dup2))
    diffs = [dup2[i]-dup0[i] for i in range(k)]
    dup_delta = mean(diffs); dup_sd = std(diffs)
    dup_t = dup_delta/(dup_sd/math.sqrt(k)) if (dup_sd > 0 and k > 1) else 0.0

    # single-game score mean: Welch t (ct2 - ct0)
    t_mean, d_mean, df_mean = welch_t(raw2, raw0)
    pooled_sd = math.sqrt((raw0.var(ddof=1)*(n0-1)+raw2.var(ddof=1)*(n2-1))/(n0+n2-2))
    cohen_d = d_mean/pooled_sd if pooled_sd > 0 else 0.0

    # single-game mean placement: Welch t
    t_plc, d_plc, _ = welch_t(rank2, rank0)

    # variance: F-test (F = var0/var2; F>1 => tau2 LOWER variance = good)
    var0, var2 = raw0.var(ddof=1), raw2.var(ddof=1)
    F = var0/var2 if var2 > 0 else float("inf")
    std_delta = math.sqrt(var2) - math.sqrt(var0)   # ct2 - ct0 (negative = tau2 lowers std)

    # rates: two-proportion z (ct2 - ct0)
    f1_0, f1_2 = rec0["sg_first_rate"], rec2["sg_first_rate"]
    f4_0, f4_2 = rec0["sg_fourth_rate"], rec2["sg_fourth_rate"]
    z_first, d_first = prop_z(f1_2, n2, f1_0, n0)
    z_fourth, d_fourth = prop_z(f4_2, n2, f4_0, n0)

    comparisons.append(dict(
        ref_tau=rt,
        field=("over-claim (homogeneous)" if rt == 0 else "selective (~real top-10)"),
        ref_claim_rate_kept_ct0=rec0["ref_claim_rate_kept"],
        n_games_per_cell=n0,
        # DUPLICATE (expect null)
        dup_placement_ct0=rec0["dup_placement_mean"], dup_placement_ct2=rec2["dup_placement_mean"],
        dup_delta=round(dup_delta, 4), dup_t=round(dup_t, 3), dup_df=k-1,
        # SINGLE-GAME 1st-rate (win proxy)
        sg_first_rate_ct0=f1_0, sg_first_rate_ct2=f1_2,
        sg_first_delta=round(d_first, 4), sg_first_z=round(z_first, 3),
        # SINGLE-GAME 4th-rate (last avoidance)  -- negative delta = improvement
        sg_fourth_rate_ct0=f4_0, sg_fourth_rate_ct2=f4_2,
        sg_fourth_delta=round(d_fourth, 4), sg_fourth_z=round(z_fourth, 3),
        # SINGLE-GAME raw score mean
        sg_score_mean_ct0=rec0["sg_score_mean"], sg_score_mean_ct2=rec2["sg_score_mean"],
        sg_score_mean_delta=round(d_mean, 4), sg_score_mean_t=round(t_mean, 3),
        sg_score_mean_cohend=round(cohen_d, 4),
        # SINGLE-GAME raw score std/variance (variance reduction = good)
        sg_score_std_ct0=rec0["sg_score_std"], sg_score_std_ct2=rec2["sg_score_std"],
        sg_score_std_delta=round(std_delta, 4), variance_ratio_F_var0_over_var2=round(F, 4),
        # SINGLE-GAME mean placement
        sg_mean_placement_ct0=rec0["sg_mean_placement"], sg_mean_placement_ct2=rec2["sg_mean_placement"],
        sg_mean_placement_delta=round(d_plc, 4), sg_mean_placement_t=round(t_plc, 3),
    ))

# calibration sanity: cand==ref tau=0 single-game mean placement should be ~2.5
calib = None
if (0, 0) in cell_out:
    rec0, raw0, rank0, _ = cell_out[(0, 0)]
    calib = dict(cell="ct0_rt0 (cand==ref==moyu, tau=0)",
                 sg_mean_placement=rec0["sg_mean_placement"],
                 sg_first_rate=rec0["sg_first_rate"], sg_fourth_rate=rec0["sg_fourth_rate"],
                 sg_score_mean=rec0["sg_score_mean"], n_games=int(raw0.size),
                 note="single-game mean placement should be ~2.5 for self-play calibration")

res = dict(
    description="E6: is the tau=2 claim-suppression correction SCORING-FORMAT-DEPENDENT? "
                "Candidate=moyu_bn_128x40; cells cand_tau{0,2} x ref_tau{0,2}; "
                "per-game single-game metrics vs duplicate placement. >=3 wall-seed blocks/cell.",
    candidate="moyu_bn_128x40.pkl", reference="moyu_bn_128x40.pkl",
    metric_defs=dict(
        dup_placement="E1/E2 duplicate metric: 5-avg_rank summed over 24 perms (here per-block mean).",
        sg_first_rate="fraction of individual games candidate ranks 1 (avg-rank<=1.5).",
        sg_fourth_rate="fraction of individual games candidate ranks 4 (avg-rank>=3.5).",
        sg_score_mean="mean raw MCR score per individual game.",
        sg_score_std="std of raw MCR score per individual game (variance proxy).",
        sg_mean_placement="mean single-game avg-rank (calibrates ~2.5 self-play).",
        signif="dup: paired cross-block t (df=blocks-1). sg_score_mean/placement: Welch t on per-game arrays. "
               "rates: two-proportion z. variance: F=var(ct0)/var(ct2) (F>1 => tau2 lower variance). "
               "directions: lower sg_fourth_rate, lower sg_score_std, higher sg_first_rate = tau2 helps.",
    ),
    calibration=calib,
    cells=[cell_out[k][0] for k in sorted(cell_out)],
    tau_comparisons_by_field=comparisons,
)
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))

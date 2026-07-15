#!/usr/bin/env python3
"""Aggregate calibrated-gate blocks for a temporal candidate vs aug_s0.
Each block json (ckpt/archx/gates/<tag>_s*.json) = 2000 games (500 seeds x 4 seats),
placement_pts is the block mean. Calibrated line: aug_s0-vs-aug_s0 = 2.500 (tie); a WIN
needs the 95% CI lower bound of the block-mean distribution > 2.500.
"""
import os, sys, json, glob, math
import numpy as np

GATES = "ckpt/archx/gates"
CALIB = 2.500  # aug_s0 vs aug_s0

# Student-t 0.975 quantiles for small df (fallback 1.96 for large)
T975 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
        11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,
        20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,29:2.045,39:2.023,49:2.010}
def tq(df):
    if df in T975: return T975[df]
    if df<=0: return float("inf")
    ks=sorted(T975);
    for k in ks:
        if k>=df: return T975[k]
    return 1.960

def summarize(tag):
    files = sorted(glob.glob(os.path.join(GATES, f"{tag}_s*.json")))
    blocks=[]; details=[]
    for f in files:
        d=json.load(open(f))
        if d.get("ref") not in ("aug_128x40_s0.pkl",):  # only blocks gated vs aug_s0
            continue
        blocks.append(d["placement_pts"]); details.append((os.path.basename(f), d["placement_pts"], d.get("seed0")))
    n=len(blocks)
    if n==0: return None
    x=np.array(blocks,float)
    mean=float(x.mean()); sd=float(x.std(ddof=1)) if n>1 else 0.0
    se=sd/math.sqrt(n) if n>1 else 0.0
    t=tq(n-1); half=t*se
    ci_lo=mean-half; ci_hi=mean+half
    # block bootstrap CI (10k resamples of the n block means)
    rng=np.random.RandomState(0)
    bs=np.array([rng.choice(x,n,replace=True).mean() for _ in range(10000)])
    blo=float(np.percentile(bs,2.5)); bhi=float(np.percentile(bs,97.5))
    beats = ci_lo>CALIB and blo>CALIB
    return dict(tag=tag, n_blocks=n, total_games=n*2000, placement_mean=round(mean,4),
                placement_sd=round(sd,4), se=round(se,4),
                ci95_lo=round(ci_lo,4), ci95_hi=round(ci_hi,4),
                boot_lo=round(blo,4), boot_hi=round(bhi,4),
                margin_lo=round(ci_lo-CALIB,4), calib_line=CALIB,
                beats_augs0=bool(beats),
                verdict=("BEATS_AUGS0" if beats else ("WORSE" if ci_hi<CALIB else "TIED_NOT_SEPARATED")),
                blocks=details)

if __name__=="__main__":
    tags=sys.argv[1:] or ["temporal_s0"]
    out={t:summarize(t) for t in tags}
    print(json.dumps(out, indent=2))

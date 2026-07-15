"""e6_match_assemble.py — E6-P3 assemble arms from precomputed eval hands + verdict.

Arms (per match, hand indices 0..M-1; all policies deterministic so arms are
EXACTLY paired on the same hand outcomes):
  always_kd   : kdens3 every hand
  always_aug  : aug_s0 every hand
  oracle_rule : hand 0 kd; hands 1+ aug iff TRUE field == weak (the switch-rule
                ceiling: told the true class, same protocol as the switcher)
  oracle_best : hand 0 kd; hands 1+ the Phase-1-best fixed policy for the field
                (weak/mixed/strong -> aug_s0, champion -> kdens3)
  switcher    : STICKY. Hand 0 kd. After each completed hand h (1..M-1) while
                unswitched, estimator_h on cumulative features of hands 1..h
                (all kd-played while unswitched -> matches training dist);
                first pred==weak -> aug_s0 permanently.

  python3 e6_match_assemble.py --evaldata data/e6_match_eval.npz \
      --est results/E6_CROSSHAND_EST.json --ckptdir ckpt/e6 \
      --out results/E6_CROSSHAND.json
"""
import os, sys, json, argparse, time, math, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from e6_match_common import (FIELD_ORDER, CLS_OF_FIELD, CLASSES, rows_to_cumfeat)

COL = dict(winner=0, zimo=1, ron=2, dealin=3, fan=4, plays=5, rank=6)
SCORE0 = 16          # scores cols 16..19
ORACLE_BEST_AUG = {"weak": True, "mixed": True, "strong": True, "champion": False}


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    n = len(a); m = float(a.mean())
    se = float(a.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    return dict(mean=round(m, 4), se=round(se, 4),
                ci95=[round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)], n=n)


def _paired(x, y):
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    s = _stats(d)
    s["significant"] = bool(abs(s["mean"]) > 1.96 * s["se"]) if s["se"] > 0 else False
    return s


def arm_metrics(rows_sel):
    """rows_sel: (N, M, NROW) chosen rows per match/hand."""
    tot = rows_sel[:, :, SCORE0].sum(axis=1)                    # (N,) seat-0 match total
    seat_tot = rows_sel[:, :, SCORE0:SCORE0 + 4].sum(axis=1)    # (N,4)
    c = seat_tot[:, 0:1]
    mrank = ((seat_tot > c).sum(axis=1)
             + ((seat_tot == c).sum(axis=1) + 1) / 2.0)
    return dict(total_score=_stats(tot), match_rank=_stats(mrank),
                hand_rank_mean=round(float(rows_sel[:, :, COL["rank"]].mean()), 4),
                hand_win_rate=round(float((rows_sel[:, :, COL["winner"]] == 0).mean()), 4),
                hand_dealin_rate=round(float(rows_sel[:, :, COL["dealin"]].mean()), 4)), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldata", required=True)
    ap.add_argument("--est", required=True)
    ap.add_argument("--ckptdir", default="ckpt/e6")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d = np.load(a.evaldata, allow_pickle=True)
    Xkd, Xaug = d["X_kd"], d["X_aug"]            # (4, N, M, NROW)
    F, N, M, _ = Xkd.shape
    est_json = json.load(open(a.est))
    models = {}
    for h in range(1, M):
        with open(os.path.join(a.ckptdir, f"e6_match_est_h{h}.pkl"), "rb") as f:
            models[h] = pickle.load(f)

    switcher_meta = dict(
        rule=("sticky: hand0 kd; after hand h (while unswitched, so features are "
              "all-kd-played = estimator training dist) predict with model_h on "
              "cumulative features of hands 1..h; first pred==weak -> aug_s0 "
              "for all remaining hands"),
        oracle_rule="hand0 kd; hands 1+ aug iff true field==weak",
        oracle_best="hand0 kd; hands 1+ Phase-1 best fixed policy "
                    f"(aug on {[k for k, v in ORACLE_BEST_AUG.items() if v]}, kd on champion)")
    table = {}
    totals = {}
    for fi, field in enumerate(FIELD_ORDER):
        t0 = time.time()
        kd, aug = Xkd[fi], Xaug[fi]
        cum = np.stack([rows_to_cumfeat(kd[mi]) for mi in range(N)])  # (N, M, NFEAT)
        # sticky switch hand: first h in 1..M-1 with pred==weak on kd-only features
        switch_hand = np.full(N, M, dtype=np.int64)
        preds_h1 = None
        for h in range(1, M):
            live = switch_hand == M
            if not live.any():
                break
            pr = models[h].predict(cum[live, h - 1])
            if h == 1:
                pr_all = models[1].predict(cum[:, 0])
                preds_h1 = {CLASSES[c]: round(float((pr_all == c).mean()), 4)
                            for c in range(3)}
            idx = np.flatnonzero(live)[pr == 0]
            switch_hand[idx] = h
        hidx = np.arange(M)[None, :]
        sel_sw = np.where((hidx >= switch_hand[:, None])[:, :, None], aug, kd)
        arms = {
            "always_kd": kd,
            "always_aug": aug,
            "oracle_rule": (np.concatenate([kd[:, :1], aug[:, 1:]], axis=1)
                            if field == "weak" else kd),
            "oracle_best": (np.concatenate([kd[:, :1], aug[:, 1:]], axis=1)
                            if ORACLE_BEST_AUG[field] else kd),
            "switcher": sel_sw,
        }
        cell = {}
        tots = {}
        for name, rows in arms.items():
            cell[name], tots[name] = arm_metrics(rows)
        sw = switch_hand[switch_hand < M]
        cell["switcher"]["switch_frac"] = round(float((switch_hand < M).mean()), 4)
        cell["switcher"]["mean_switch_hand"] = (round(float(sw.mean()), 4)
                                                if len(sw) else None)
        cell["switcher"]["switch_hand_hist"] = {
            str(h): int((switch_hand == h).sum()) for h in range(1, M)}
        cell["switcher"]["pred_dist_h1"] = preds_h1
        cell["paired_diffs_total_score"] = {
            "switcher_vs_always_kd": _paired(tots["switcher"], tots["always_kd"]),
            "switcher_vs_always_aug": _paired(tots["switcher"], tots["always_aug"]),
            "switcher_vs_oracle_rule": _paired(tots["switcher"], tots["oracle_rule"]),
            "oracle_rule_vs_always_kd": _paired(tots["oracle_rule"], tots["always_kd"]),
        }
        cell["seconds"] = round(time.time() - t0, 1)
        table[field] = cell
        totals[field] = tots
        print(f"FIELD {field}: total_score kd={cell['always_kd']['total_score']['mean']} "
              f"aug={cell['always_aug']['total_score']['mean']} "
              f"oracle={cell['oracle_rule']['total_score']['mean']} "
              f"switcher={cell['switcher']['total_score']['mean']} "
              f"switch_frac={cell['switcher']['switch_frac']}", flush=True)

    # ---- verdict ----
    pd_weak = table["weak"]["paired_diffs_total_score"]
    gap = pd_weak["oracle_rule_vs_always_kd"]["mean"]
    got = pd_weak["switcher_vs_always_kd"]["mean"]
    capture = round(got / gap, 4) if gap > 0 else None
    elsewhere = {}
    ok_elsewhere = True
    for f in ("mixed", "strong", "champion"):
        pdd = table[f]["paired_diffs_total_score"]["switcher_vs_always_kd"]
        bad = bool(pdd["mean"] < 0 and pdd["significant"])
        elsewhere[f] = dict(diff=pdd["mean"], ci95=pdd["ci95"],
                            significantly_worse=bad)
        ok_elsewhere &= not bad
    verdict = dict(
        weak_oracle_gap_paired=gap, weak_switcher_gain_paired=got,
        weak_gap_capture_frac=capture,
        weak_gain_significant=bool(pd_weak["switcher_vs_always_kd"]["mean"] > 0
                                   and pd_weak["switcher_vs_always_kd"]["significant"]),
        matches_kdens3_elsewhere=elsewhere,
        captures_most_oracle_gap_weak=bool(capture is not None and capture >= 0.5),
        no_significant_loss_elsewhere=bool(ok_elsewhere),
        overall=bool(capture is not None and capture >= 0.5 and ok_elsewhere))
    out = dict(meta=dict(design="E6-P3 cross-hand field estimation + match switching",
                         evaldata=os.path.abspath(a.evaldata),
                         n_matches_per_field=int(N), hands_per_match=int(M),
                         eval_seed0=int(d["seed0"]),
                         switcher=switcher_meta,
                         finished=time.strftime("%F %T")),
               estimator=est_json, switcher_table=table, verdict=verdict)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    print("SAVED", a.out, flush=True)


if __name__ == "__main__":
    main()

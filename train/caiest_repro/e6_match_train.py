"""e6_match_train.py — E6-P3 cross-hand field estimator.

HistGradientBoostingClassifier per horizon h in 1..hands-1: cumulative-mean
features over hands 1..h -> 3-class field type (weak / mixed / strong+champ,
same classes as P2). Split BY MATCH (match_idx %% 5 == 0 -> test), balanced
sample weights. THE NUMBER: balanced accuracy by h — how fast does cross-hand
identifiability rise above the 0.333 chance floor (P2 within-hand: 0.34-0.36)?

  python3 e6_match_train.py --data data/e6_match_train.npz \
      --out results/E6_CROSSHAND_EST.json --ckptdir ckpt/e6
"""
import os, sys, json, argparse, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from e6_match_common import (FIELD_ORDER, CLS_OF_FIELD, CLASSES, FEAT_NAMES,
                             rows_to_cumfeat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckptdir", default="ckpt/e6")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.ckptdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d = np.load(a.data, allow_pickle=True)
    X = d["X"]                     # (4, N, M, NROW)
    F, N, M, _ = X.shape
    cum = np.stack([np.stack([rows_to_cumfeat(X[fi, mi]) for mi in range(N)])
                    for fi in range(F)])          # (4, N, M, NFEAT)
    y3 = np.array([CLS_OF_FIELD[f] for f in FIELD_ORDER])
    mids = np.arange(N)
    test_m = (mids % 5 == 0)
    out = {"design": ("HistGB on cumulative per-hand outcome features after h "
                      "hands -> 3-class field; split by match (idx%5==0 test); "
                      "balanced sample weights; kdens3 seat-0 (pre-switch dist)"),
           "data": os.path.abspath(a.data), "classes": CLASSES,
           "feat_names": FEAT_NAMES, "n_matches_per_field": int(N),
           "hands_per_match": int(M), "by_h": {}, "bal_acc_by_h": {}}
    for h in range(1, M):          # estimate after hands 1..M-1 (switch decision points)
        t0 = time.time()
        Xtr = np.concatenate([cum[fi, ~test_m, h - 1] for fi in range(F)])
        Xte = np.concatenate([cum[fi, test_m, h - 1] for fi in range(F)])
        ytr = np.concatenate([np.full((~test_m).sum(), y3[fi]) for fi in range(F)])
        yte = np.concatenate([np.full(test_m.sum(), y3[fi]) for fi in range(F)])
        fte = np.concatenate([np.full(test_m.sum(), fi) for fi in range(F)])
        cw = {c: len(ytr) / (3.0 * (ytr == c).sum()) for c in range(3)}
        sw = np.array([cw[c] for c in ytr])
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, early_stopping=True,
            validation_fraction=0.15, random_state=a.seed)
        clf.fit(Xtr, ytr, sample_weight=sw)
        pred = clf.predict(Xte)
        bal = balanced_accuracy_score(yte, pred)
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
        pw_by_field = {FIELD_ORDER[fi]: round(float((pred[fte == fi] == 0).mean()), 4)
                       for fi in range(F)}
        ck = os.path.join(a.ckptdir, f"e6_match_est_h{h}.pkl")
        with open(ck, "wb") as f:
            pickle.dump(clf, f)
        out["by_h"][str(h)] = dict(
            n_train=int(len(ytr)), n_test=int(len(yte)),
            bal_acc=round(float(bal), 4),
            acc=round(float((pred == yte).mean()), 4),
            recalls={CLASSES[c]: round(float(cm[c, c] / max(1, cm[c].sum())), 4)
                     for c in range(3)},
            weak_precision=round(float(cm[0, 0] / max(1, cm[:, 0].sum())), 4),
            confusion_rows_true_cols_pred=cm.tolist(),
            pred_weak_rate_by_true_field=pw_by_field,
            n_iter=int(clf.n_iter_), seconds=round(time.time() - t0, 1), ckpt=ck)
        out["bal_acc_by_h"][str(h)] = round(float(bal), 4)
        print(f"h={h}: bal_acc={bal:.4f} pred_weak_by_field={pw_by_field} "
              f"({time.time()-t0:.1f}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print("CURVE", json.dumps(out["bal_acc_by_h"]), flush=True)


if __name__ == "__main__":
    main()

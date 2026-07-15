"""e6_estimator_train.py — E6-P2(a) train per-turn field-type classifiers.

For each snapshot turn t in {4,8,12}: small CNN (e6_switch_common.EstNet) on
seat-0 obs -> {weak, mixed, strong/champion}. Held-out split is BY GAME SEED
(seed % 5 == 0 -> test, ~20%). Class weights = inverse frequency (strong+champ
is a merged, doubled class). Reports per-turn held-out accuracy, balanced
accuracy, confusion, and per-original-field P(pred=weak) — the false-switch
rate on champion fields is the switcher's key risk.

Chooses T = smallest turn whose balanced_acc >= max_balanced_acc - 0.02
(earliest identifiable turn leaves the most game to exploit).

  python3 e6_estimator_train.py --data data/e6_est_snaps.npz \
      --out results/E6_ESTIMATOR.json --ckptdir ckpt/e6
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from e6_switch_common import EstNet, SNAP_TURNS, CLASSES, FIELD_ORDER

torch.set_num_threads(8)


def train_one(X, y, field, seed, device, epochs=15, bs=256, lr=1e-3):
    test = (seed % 5 == 0)
    Xtr, ytr = X[~test], y[~test]
    Xte, yte, fte = X[test], y[test], field[test]
    cnt = np.bincount(ytr, minlength=3).astype(np.float64)
    w = cnt.sum() / (3.0 * np.maximum(cnt, 1))
    net = EstNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr)
    n = len(ytr_t)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            xb = torch.from_numpy(Xte[i:i + 1024]).float().to(device)
            preds.append(net(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = float((pred == yte).mean())
    conf = np.zeros((3, 3), dtype=int)
    for a, b in zip(yte, pred):
        conf[a, b] += 1
    recalls = [float(conf[c, c] / max(1, conf[c].sum())) for c in range(3)]
    bal = float(np.mean(recalls))
    weak_prec = float(conf[0, 0] / max(1, conf[:, 0].sum()))
    pw_by_field = {}
    for fi, fname in enumerate(FIELD_ORDER):
        m = (fte == fi)
        pw_by_field[fname] = round(float((pred[m] == 0).mean()), 4) if m.any() else None
    rep = dict(n_train=int((~test).sum()), n_test=int(test.sum()),
               train_class_counts=[int(c) for c in cnt],
               acc=round(acc, 4), balanced_acc=round(bal, 4),
               recalls={CLASSES[c]: round(recalls[c], 4) for c in range(3)},
               weak_precision=round(weak_prec, 4),
               confusion_rows_true_cols_pred=conf.tolist(),
               pred_weak_rate_by_true_field=pw_by_field)
    return net, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckptdir", default="ckpt/e6")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.ckptdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    z = np.load(a.data)
    report = dict(design="CNN obs(38,4,9)->3-class field type at seat-0 discard-turn "
                         "snapshots; split by game seed (seed%5==0 test); class-weighted CE",
                  classes=CLASSES, data=os.path.abspath(a.data),
                  device=device, epochs=a.epochs, turns={})
    for t in SNAP_TURNS:
        t0 = time.time()
        X, y = z[f"X{t}"], z[f"y{t}"]
        field, seed = z[f"field{t}"], z[f"seed{t}"]
        net, rep = train_one(X, y, field, seed, device, epochs=a.epochs)
        rep["seconds"] = round(time.time() - t0, 1)
        ck = os.path.join(a.ckptdir, f"e6_est_t{t}.pt")
        torch.save(net.cpu().state_dict(), ck)
        rep["ckpt"] = os.path.abspath(ck)
        report["turns"][str(t)] = rep
        print(f"TURN {t}: acc={rep['acc']} bal={rep['balanced_acc']} "
              f"recalls={rep['recalls']} pred_weak_by_field={rep['pred_weak_rate_by_true_field']}",
              flush=True)
    bals = {t: report["turns"][str(t)]["balanced_acc"] for t in SNAP_TURNS}
    best = max(bals.values())
    chosen = min(t for t in SNAP_TURNS if bals[t] >= best - 0.02)
    report["chosen_T"] = int(chosen)
    report["chosen_rule"] = "smallest turn with balanced_acc >= max-0.02"
    report["balanced_acc_by_turn"] = {str(k): v for k, v in bals.items()}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print("CHOSEN T =", chosen, "| balanced acc by turn:", bals, flush=True)


if __name__ == "__main__":
    main()

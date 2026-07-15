"""
oppdealin_eval.py -- THE VERDICT: does the off-policy v2 beat the on-policy model on the
full-candidate (all legal tiles) off-policy held-out set?

Loads the off-policy per-candidate held-out pairs (same game-disjoint split as the trainer),
then evaluates BOTH:
  - v2  (ckpt/dealin_pc_v2/*.pt)  -- trained off-policy, all candidates
  - on-policy (ckpt/dealin_pc/*.pt) -- trained only on kdens3's chosen discards
on the SAME held-out pairs. Reports per-model and ensemble AUROC + PR-AUC. v2 should win,
especially on the off-distribution tiles kdens3 rarely picks.

  CUDA_VISIBLE_DEVICES=3 python3 oppdealin_eval.py --data data/oppdealin/full \
       --v2 ckpt/dealin_pc_v2 --onpolicy ckpt/dealin_pc
"""
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from oppdealin_train import DealInFused, load_pairs, VAL_SPLIT_SEED


def load_fused(path, dev):
    sd = torch.load(path, map_location="cpu")
    ch = sd["stem.weight"].shape[0]
    blocks = len({k.split(".")[1] for k in sd if k.startswith("body.")})
    net = DealInFused(ch, blocks); net.load_state_dict(sd); net.eval().to(dev)
    return net


@torch.no_grad()
def predict(net, obs_gpu, dec_t, tile_t, idxs, dev, bs=8192):
    ps = []
    for i in range(0, len(idxs), bs):
        bi = torch.from_numpy(idxs[i:i + bs]).to(dev)
        di = dec_t[bi]; ti = tile_t[bi]; n = len(bi)
        o = obs_gpu[di].float()
        cand = torch.zeros(n, 1, 4, 9, device=dev)
        cand[torch.arange(n, device=dev), 0, torch.div(ti, 9, rounding_mode='floor'), ti % 9] = 1.0
        x = torch.cat([o, cand], dim=1)
        ps.append(torch.sigmoid(net(x)).cpu().numpy())
    return np.concatenate(ps)


def ens_pred(paths, obs_gpu, dec_t, tile_t, idxs, dev):
    acc = None
    per = {}
    for p in sorted(paths):
        pr = predict(load_fused(p, dev), obs_gpu, dec_t, tile_t, idxs, dev)
        per[os.path.basename(p)] = pr
        acc = pr if acc is None else acc + pr
    return acc / len(paths), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/oppdealin/full")
    ap.add_argument("--v2", default="ckpt/dealin_pc_v2")
    ap.add_argument("--onpolicy", default="ckpt/dealin_pc")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out", default="results/oppdealin_verdict.json")
    a = ap.parse_args()
    dev = "cuda"

    obs, dec_idx, tile, label, pgame, game = load_pairs(a.data)
    rng = np.random.RandomState(VAL_SPLIT_SEED)
    ug = np.unique(game); rng.shuffle(ug)
    n_val = int(len(ug) * a.val_frac); val_games = set(ug[:n_val].tolist())
    is_val = np.array([g in val_games for g in pgame])
    va_idx = np.flatnonzero(is_val)
    yv = label[va_idx]
    print(f"[eval] held-out candidate_pairs={len(va_idx)} pos={int(yv.sum())} "
          f"base_rate={yv.mean():.4f} val_games={n_val}", flush=True)

    obs_gpu = torch.from_numpy(obs).to(dev)
    dec_t = torch.from_numpy(dec_idx).to(dev); tile_t = torch.from_numpy(tile).to(dev)

    v2_paths = glob.glob(os.path.join(a.v2, "*.pt"))
    op_paths = glob.glob(os.path.join(a.onpolicy, "*.pt"))
    assert v2_paths and op_paths, f"missing ckpts v2={v2_paths} op={op_paths}"

    def report(tag, paths):
        ens, per = ens_pred(paths, obs_gpu, dec_t, tile_t, va_idx, dev)
        rows = {}
        for name, pr in per.items():
            rows[name] = dict(auroc=round(float(roc_auc_score(yv, pr)), 4),
                              prauc=round(float(average_precision_score(yv, pr)), 4))
        ens_au = round(float(roc_auc_score(yv, ens)), 4)
        ens_pr = round(float(average_precision_score(yv, ens)), 4)
        print(f"[eval] {tag}: ENSEMBLE AUROC={ens_au} PR-AUC={ens_pr}", flush=True)
        for name, r in rows.items():
            print(f"[eval]   {tag} {name}: AUROC={r['auroc']} PR-AUC={r['prauc']}", flush=True)
        return dict(ensemble_auroc=ens_au, ensemble_prauc=ens_pr, per_seed=rows,
                    paths=sorted(paths))

    res = dict(data=a.data, held_out_pairs=int(len(va_idx)),
               held_out_pos=int(yv.sum()), base_rate=round(float(yv.mean()), 6),
               val_games=int(n_val), val_split_seed=VAL_SPLIT_SEED)
    res["v2_offpolicy"] = report("v2(off-policy)", v2_paths)
    res["onpolicy"] = report("on-policy", op_paths)
    res["verdict_auroc_gain"] = round(res["v2_offpolicy"]["ensemble_auroc"]
                                      - res["onpolicy"]["ensemble_auroc"], 4)
    res["verdict_prauc_gain"] = round(res["v2_offpolicy"]["ensemble_prauc"]
                                      - res["onpolicy"]["ensemble_prauc"], 4)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n[eval] VERDICT on OFF-POLICY full-candidate held-out set "
          f"({len(va_idx)} pairs, base_rate={yv.mean():.4f}):", flush=True)
    print(f"[eval]   v2 (off-policy) : AUROC={res['v2_offpolicy']['ensemble_auroc']} "
          f"PR-AUC={res['v2_offpolicy']['ensemble_prauc']}", flush=True)
    print(f"[eval]   on-policy       : AUROC={res['onpolicy']['ensemble_auroc']} "
          f"PR-AUC={res['onpolicy']['ensemble_prauc']}", flush=True)
    print(f"[eval]   GAIN            : AUROC {res['verdict_auroc_gain']:+.4f} "
          f"PR-AUC {res['verdict_prauc_gain']:+.4f}  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

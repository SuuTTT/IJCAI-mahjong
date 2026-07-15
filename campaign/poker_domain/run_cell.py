"""Run one experiment cell (label-noise eps, seed):
  - load the behavioral-cloning dataset (self-play under the coherent reference)
  - corrupt a fraction eps of action labels -> uniformly random LEGAL action
  - train N teacher nets (imitation, CE), form the teacher-ensemble
  - distill M student nets from teacher-ensemble soft labels (KD + CE)
  - build each arm's behavioral strategy over ALL 288 infosets (mask+renorm)
  - EVAL: exact exploitability of teacher_ens / student_ens / singles / reference
  - write CELL.json with per-arm exploitability and the gap.

Metric: exploitability (chips/hand, lower = closer to the coherent equilibrium).
gap_student_ens_minus_teacher_ens = expl(student_ens) - expl(teacher_ens)
  (negative => student-ensemble is LESS exploitable => better)
advantage_student = expl(teacher_ens) - expl(student_ens)  (positive => student wins)
Prediction: advantage_student grows with eps.
"""
import argparse, json, os, time, random
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
import leduc as L
from model import PolicyMLP, encode_keys, NACT
import exploit as E
from cfr import json_to_strat


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def corrupt_labels(y, legal, eps, seed):
    if eps <= 0:
        return y.copy(), 0
    rng = np.random.RandomState(seed)
    y = y.copy()
    idx = np.where(rng.rand(len(y)) < eps)[0]
    for i in idx:
        opts = np.where(legal[i] == 1)[0]
        y[i] = opts[rng.randint(len(opts))]
    return y, len(idx)


def train_net(Xt, yt, device, seed, epochs, lr, bs, hidden,
              soft=None, alpha=0.0, temp=2.0):
    set_seed(seed)
    net = PolicyMLP(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        net.train()
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logits = net(Xt[b])
            ce = F.cross_entropy(logits, yt[b])
            if soft is not None:
                kd = F.kl_div(F.log_softmax(logits / temp, dim=1), soft[b],
                              reduction="batchmean") * (temp * temp)
                loss = alpha * kd + (1 - alpha) * ce
            else:
                loss = ce
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def ensemble_soft(nets, Xt, temp, bs=8192):
    n = Xt.shape[0]
    out = torch.zeros(n, NACT, device=Xt.device)
    for i in range(0, n, bs):
        acc = None
        for net in nets:
            p = F.softmax(net(Xt[i:i + bs]) / temp, dim=1)
            acc = p if acc is None else acc + p
        out[i:i + bs] = acc / len(nets)
    return out


@torch.no_grad()
def build_strategy(nets, keys, legal_lists, device):
    """Behavioral strategy over infosets: mean softmax over nets, mask to legal,
    renormalize. Returns dict key -> [3] probs."""
    X = torch.from_numpy(encode_keys(keys)).to(device)
    acc = None
    for net in nets:
        p = F.softmax(net(X), dim=1)
        acc = p if acc is None else acc + p
    probs = (acc / len(nets)).cpu().numpy()
    strat = {}
    for i, k in enumerate(keys):
        legal = legal_lists[k]
        v = np.zeros(3)
        for a in legal:
            v[a] = probs[i, a]
        tot = v.sum()
        if tot <= 1e-12:
            for a in legal:
                v[a] = 1.0 / len(legal)
        else:
            v = v / tot
        strat[k] = v.tolist()
    return strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", default="reference_strategy.json")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.data)
    feats, y0, legal = d["feats"], d["y"], d["legal"]
    y, n_corrupt = corrupt_labels(y0, legal, a.eps, a.seed + 777)
    Xt = torch.from_numpy(feats).to(device)
    yt = torch.from_numpy(y).long().to(device)
    n = len(y)
    print(f"[cell eps={a.eps} seed={a.seed}] samples={n} corrupt={n_corrupt} "
          f"device={device}", flush=True)

    infosets = L.enumerate_infosets()      # key -> legal actions
    keys = list(infosets.keys())

    with open(a.ref) as f:
        refj = json.load(f)
    ref_strat = json_to_strat(refj["strategy"])
    ref_expl = E.exploitability(ref_strat)

    # ---- N teachers ----
    teachers = []
    for i in range(a.N):
        net = train_net(Xt, yt, device, seed=a.seed * 100 + i,
                        epochs=a.epochs, lr=a.lr, bs=a.bs, hidden=a.hidden)
        teachers.append(net)

    # ---- distill M students from teacher-ensemble soft labels ----
    soft = ensemble_soft(teachers, Xt, a.temp)
    students = []
    for j in range(a.M):
        net = train_net(Xt, yt, device, seed=a.seed * 100 + 50 + j,
                        epochs=a.epochs, lr=a.lr, bs=a.bs, hidden=a.hidden,
                        soft=soft, alpha=a.alpha, temp=a.temp)
        students.append(net)

    # ---- build strategies & eval exploitability ----
    arms = {
        "teacher_ens": teachers,
        "student_ens": students,
        "teacher_single": teachers[:1],
        "student_single": students[:1],
    }
    expl = {}
    for name, nets in arms.items():
        strat = build_strategy(nets, keys, infosets, device)
        expl[name] = E.exploitability(strat)
        print(f"[cell eps={a.eps} seed={a.seed}] {name} exploitability="
              f"{expl[name]:.5f}", flush=True)

    gap = expl["student_ens"] - expl["teacher_ens"]
    advantage = expl["teacher_ens"] - expl["student_ens"]

    out = {
        "domain": "leduc_poker", "metric": "exploitability_lower_better",
        "eps": a.eps, "seed": a.seed,
        "N_teachers": a.N, "M_students": a.M, "alpha": a.alpha, "kd_temp": a.temp,
        "hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "bs": a.bs,
        "n_samples": int(n), "n_corrupt_labels": int(n_corrupt),
        "n_infosets": len(keys),
        "reference_exploitability": round(ref_expl, 6),
        "exploitability": {k: round(v, 6) for k, v in expl.items()},
        "gap_student_ens_minus_teacher_ens": round(gap, 6),
        "advantage_student_ens_over_teacher_ens": round(advantage, 6),
        "wall_seconds": round(time.time() - t0, 1),
    }
    outfile = os.path.join(a.outdir, "CELL.json")
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    open(os.path.join(a.outdir, "CELL.DONE"), "w").close()
    print(f"[cell eps={a.eps} seed={a.seed}] DONE advantage={advantage:+.5f} "
          f"teacher_ens={expl['teacher_ens']:.5f} "
          f"student_ens={expl['student_ens']:.5f} ref={ref_expl:.5f} "
          f"secs={out['wall_seconds']} -> {outfile}", flush=True)


if __name__ == "__main__":
    main()

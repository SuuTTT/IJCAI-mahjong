"""Run one experiment cell (teacher depth D, label-noise eps):
  - load imitation data for depth D
  - optionally corrupt fraction eps of labels with a random legal move
  - train N teacher nets (imitation), form teacher-ensemble
  - distill M student nets from the teacher-ensemble soft labels (KD, alpha)
  - evaluate teacher-ens, student-ens, and single nets vs minimax ladder D=1,3,5
  - write CELL json with per-arm winrates, the student_ens - teacher_ens GAP, CIs

Prediction: GAP grows as teacher depth D increases (stronger/more coherent target).
"""
import argparse, json, os, time, random
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(2)  # host-side threads for GPU training; avoid oversubscription
import torch.nn.functional as F
import othello as oth
from model import PolicyCNN, encode_planes, NSQ
import neteval


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def corrupt_labels(me, opp, y, eps, seed):
    """Replace a fraction eps of labels with a uniformly random LEGAL move."""
    if eps <= 0:
        return y.copy(), 0
    rng = np.random.RandomState(seed)
    y = y.copy()
    idx = np.where(rng.rand(len(y)) < eps)[0]
    for i in idx:
        moves = oth.legal_move_list(int(me[i]), int(opp[i]))
        if moves:
            y[i] = rng.choice(moves)
    return y, len(idx)


def train_net(Xt, yt, device, seed, epochs, lr, bs, ch,
              soft=None, alpha=0.0, temp=2.0):
    set_seed(seed)
    net = PolicyCNN(ch=ch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        net.train()
        for i in range(0, n, bs):
            bidx = perm[i:i + bs]
            xb = Xt[bidx]; yb = yt[bidx]
            logits = net(xb)
            ce = F.cross_entropy(logits, yb)
            if soft is not None:
                sb = soft[bidx]
                kd = F.kl_div(F.log_softmax(logits / temp, dim=1), sb,
                              reduction="batchmean") * (temp * temp)
                loss = alpha * kd + (1 - alpha) * ce
            else:
                loss = ce
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def train_acc(net, Xt, yt, bs=4096):
    correct = 0
    for i in range(0, Xt.shape[0], bs):
        pred = net(Xt[i:i + bs]).argmax(1)
        correct += (pred == yt[i:i + bs]).sum().item()
    return correct / Xt.shape[0]


@torch.no_grad()
def ensemble_soft(nets, Xt, temp, bs=4096):
    """Mean softmax(logits/temp) over nets -> soft targets [n,36]."""
    n = Xt.shape[0]
    out = torch.zeros(n, NSQ, device=Xt.device)
    for i in range(0, n, bs):
        acc = None
        for net in nets:
            p = F.softmax(net(Xt[i:i + bs]) / temp, dim=1)
            acc = p if acc is None else acc + p
        out[i:i + bs] = acc / len(nets)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=6)     # teachers
    ap.add_argument("--M", type=int, default=3)     # students
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--n_pairs", type=int, default=60)
    ap.add_argument("--ladder", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--eval_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.data)
    me, opp, y0 = d["me"], d["opp"], d["y"]
    y, n_corrupt = corrupt_labels(me, opp, y0, a.eps, a.seed + 777)
    X = encode_planes(me, opp)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).long().to(device)
    n = len(y)
    print(f"[cell D={a.depth} eps={a.eps}] states={n} corrupt={n_corrupt} "
          f"device={device}", flush=True)

    # ---- train N teachers ----
    teachers = []
    for i in range(a.N):
        net = train_net(Xt, yt, device, seed=a.seed * 100 + i,
                        epochs=a.epochs, lr=a.lr, bs=a.bs, ch=a.ch)
        p = os.path.join(a.outdir, f"teacher_{i}.pt")
        torch.save(net.state_dict(), p)
        teachers.append(net)
    tacc = [round(train_acc(t, Xt, yt), 4) for t in teachers]
    print(f"[cell D={a.depth} eps={a.eps}] teacher train-acc {tacc}", flush=True)

    # ---- teacher-ensemble soft labels -> distill M students ----
    soft = ensemble_soft(teachers, Xt, a.temp)
    students = []
    for j in range(a.M):
        net = train_net(Xt, yt, device, seed=a.seed * 100 + 50 + j,
                        epochs=a.epochs, lr=a.lr, bs=a.bs, ch=a.ch,
                        soft=soft, alpha=a.alpha, temp=a.temp)
        p = os.path.join(a.outdir, f"student_{j}.pt")
        torch.save(net.state_dict(), p)
        students.append(net)
    sacc = [round(train_acc(s, Xt, yt), 4) for s in students]
    print(f"[cell D={a.depth} eps={a.eps}] student train-acc {sacc}", flush=True)

    # free GPU before CPU eval
    del teachers, students, Xt, yt, soft
    torch.cuda.empty_cache() if device == "cuda" else None

    teacher_paths = [os.path.join(a.outdir, f"teacher_{i}.pt") for i in range(a.N)]
    student_paths = [os.path.join(a.outdir, f"student_{j}.pt") for j in range(a.M)]
    arms = {
        "teacher_ens": teacher_paths,
        "student_ens": student_paths,
        "teacher_single": teacher_paths[:1],
        "student_single": student_paths[:1],
    }

    # ---- eval vs ladder ----
    results = {}
    for arm, paths in arms.items():
        results[arm] = {}
        for od in a.ladder:
            r = neteval.eval_arm(paths, a.ch, od, a.n_pairs,
                                 seed=a.seed + od, workers=a.eval_workers)
            results[arm][f"D{od}"] = r
            print(f"[cell D={a.depth} eps={a.eps}] {arm} vs D{od}: "
                  f"wr={r['winrate']:.3f} {r['ci95']}", flush=True)

    # overall winrate per arm (pooled across ladder)
    def pooled(arm):
        pts = sum(results[arm][f"D{od}"]["points"] for od in a.ladder)
        ng = sum(results[arm][f"D{od}"]["n_games"] for od in a.ladder)
        return pts / ng, pts, ng
    overall = {}
    for arm in arms:
        wr, pts, ng = pooled(arm)
        overall[arm] = {"winrate": round(wr, 4),
                        "ci95": neteval.wilson_ci(pts, ng), "n_games": ng}

    gap_overall = overall["student_ens"]["winrate"] - overall["teacher_ens"]["winrate"]
    gap_by_opp = {f"D{od}": round(results["student_ens"][f"D{od}"]["winrate"]
                                  - results["teacher_ens"][f"D{od}"]["winrate"], 4)
                  for od in a.ladder}

    out = {
        "board": "6x6", "teacher_depth": a.depth, "eps": a.eps,
        "N_teachers": a.N, "M_students": a.M, "alpha": a.alpha, "kd_temp": a.temp,
        "channels": a.ch, "epochs": a.epochs, "lr": a.lr, "bs": a.bs,
        "n_states": n, "n_corrupt_labels": int(n_corrupt),
        "n_pairs_per_opponent": a.n_pairs, "ladder": a.ladder,
        "seed": a.seed, "teacher_train_acc": tacc, "student_train_acc": sacc,
        "per_opponent": results, "overall": overall,
        "gap_student_ens_minus_teacher_ens": round(gap_overall, 4),
        "gap_by_opponent": gap_by_opp,
        "wall_seconds": round(time.time() - t0, 1),
    }
    outfile = os.path.join(a.outdir, "CELL.json")
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[cell D={a.depth} eps={a.eps}] DONE gap={gap_overall:+.4f} "
          f"teacher_ens={overall['teacher_ens']['winrate']:.3f} "
          f"student_ens={overall['student_ens']['winrate']:.3f} "
          f"secs={out['wall_seconds']} -> {outfile}", flush=True)


if __name__ == "__main__":
    main()

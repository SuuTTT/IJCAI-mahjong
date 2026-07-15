#!/usr/bin/env python3
"""
Controlled-coherence synthetic domain for the distill-then-ensemble theory.

Ground-truth policy g: R^d -> {0..K-1} built as the argmax of a balanced blend
of C independent random 2-layer (1 hidden layer) tanh MLP sub-policies with
FIXED seeds. Coherence knob = C:
  C=1  -> single coherent sub-policy (effective teacher width = H), easily learned.
  C>1  -> balanced blend of C sub-policies; the effective target is a 2-layer MLP
          of width C*H, so its decision surface is more fragmented / higher-
          complexity and LESS learnable by the fixed-size student => less coherent.
We verify empirically that clean-test learnability decreases with C (the coherence
sanity check).

Observation noise eps: training labels = g(x), flipped to a uniformly random OTHER
class with probability eps. Clean test labels = g(x) exactly.

Per (C, eps) cell, for each of `seeds` seeds:
  - train N teacher MLPs (3x256) on the noisy labels (distinct inits/noise draws)
  - teacher-ensemble = argmax of mean softmax over the N teachers
  - distill M students via KD from the mean-teacher soft targets (T, alpha) + noisy CE
  - student-ensemble = argmax of mean softmax over the M students
  - single teacher / single student = mean over the individual models
All accuracies on the CLEAN test set. gap = student_ens_acc - teacher_ens_acc.
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- global config (recorded in every JSON) ----  HARDER REGIME (env-tunable)
D = int(os.environ.get("D", "64"))            # input dim
K = int(os.environ.get("K", "20"))            # classes
FREQ = float(os.environ.get("FREQ", "2.0"))   # ground-truth input-frequency scale (harder as it grows)
H_SUB = int(os.environ.get("H_SUB", "32"))    # hidden width of each ground-truth sub-policy
N_TRAIN = 200_000
N_TEST = 20_000
HIDDEN = 256    # student/teacher hidden width
N_LAYERS = 3    # hidden layers -> "3x256"
EPOCHS = 40
BATCH = 4096
LR = 2e-3
WD = 0.0
KD_T = 3.0      # distillation temperature
KD_ALPHA = 0.7  # weight on soft KD term
N_TEACHERS = 6
M_STUDENTS = 3
GT_WORLD_SEED = 20260713  # fixes the ground-truth functions per C

torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- ground truth ----------------
def make_subpolicy(rng, d, k, h, device):
    # smooth 1-hidden-layer tanh MLP with fixed random weights, NO output bias
    # (output bias would dominate the small zero-mean logits and collapse the
    #  argmax onto a single class -> we balance classes explicitly below).
    W1 = torch.tensor(rng.normal(0, FREQ / math.sqrt(d), size=(d, h)), dtype=torch.float32, device=device)
    b1 = torch.tensor(rng.normal(0, 1.0, size=(h,)), dtype=torch.float32, device=device)
    W2 = torch.tensor(rng.normal(0, 2.0 / math.sqrt(h), size=(h, k)), dtype=torch.float32, device=device)
    return (W1, b1, W2)


def build_g(C, device):
    """Return a function X->logits summing C fixed sub-policies (balanced classes).

    A per-class bias is fit once (fixed seed) so the argmax label distribution is
    approximately uniform over K -> chance = 1/K, accuracy is informative.
    """
    subs = []
    for c in range(C):
        rng = np.random.default_rng([GT_WORLD_SEED, C, c])
        subs.append(make_subpolicy(rng, D, K, H_SUB, device))

    def raw(X):
        # SUM (not average) of sub-policies: variance grows with C, keeping the
        # argmax well-separated; the effective target is a width-(C*H) 2-layer MLP.
        acc = None
        for (W1, b1, W2) in subs:
            f = torch.tanh(X @ W1 + b1) @ W2
            acc = f if acc is None else acc + f
        return acc

    # fit a scale + per-class bias once (fixed sample) so argmax classes ~uniform.
    rng = np.random.default_rng([GT_WORLD_SEED, C, 424242])
    Xb = torch.tensor(rng.standard_normal((60000, D)), dtype=torch.float32, device=device)
    with torch.no_grad():
        L = raw(Xb)
        scale = L.std().clamp_min(1e-6)
        Ln = L / scale
        bias = torch.zeros(K, device=device)
        target = math.log(1.0 / K)
        for _ in range(400):
            p = torch.softmax(Ln + bias, dim=1).mean(0).clamp_min(1e-8)  # smooth
            bias = bias + 0.5 * (target - torch.log(p))

    def g_logits(X):
        return raw(X) / scale + bias
    return g_logits


def sample_world(C, device):
    """Fixed per C: X_train, X_test, clean labels for both."""
    rng = np.random.default_rng([GT_WORLD_SEED, C, 999])
    Xtr = torch.tensor(rng.standard_normal((N_TRAIN, D)), dtype=torch.float32, device=device)
    Xte = torch.tensor(rng.standard_normal((N_TEST, D)), dtype=torch.float32, device=device)
    g = build_g(C, device)
    with torch.no_grad():
        ytr = g(Xtr).argmax(1)
        yte = g(Xte).argmax(1)
    return Xtr, ytr, Xte, yte


def make_noisy(y_clean, eps, seed, device):
    """Flip each label to a uniform OTHER class w.p. eps."""
    n = y_clean.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    flip = (torch.rand(n, generator=g) < eps)
    # random offset in 1..K-1, added mod K -> guaranteed a different class
    offset = torch.randint(1, K, (n,), generator=g)
    yn = y_clean.clone().cpu()
    yn[flip] = (yn[flip] + offset[flip]) % K
    return yn.to(device)


# ---------------- model ----------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(D, HIDDEN), nn.ReLU()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        layers += [nn.Linear(HIDDEN, K)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_teacher(Xtr, ytr_noisy, seed, device):
    torch.manual_seed(seed)
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = Xtr.shape[0]
    steps = EPOCHS * math.ceil(n / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device=device).manual_seed(seed + 12345)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], ytr_noisy[idx]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def soft_targets(models, X, T, device, bs=16384):
    """mean softmax (at temperature T) over models; returns log-prob-free probs."""
    n = X.shape[0]
    out = torch.zeros(n, K, device=device)
    for i in range(0, n, bs):
        xb = X[i:i + bs]
        acc = torch.zeros(xb.shape[0], K, device=device)
        for m in models:
            acc += F.softmax(m(xb) / T, dim=1)
        out[i:i + bs] = acc / len(models)
    return out


def train_student(Xtr, ytr_noisy, teacher_soft, seed, device):
    """KD: alpha * T^2 * KL(teacher||student soft) + (1-alpha)*CE(student, noisy hard)."""
    torch.manual_seed(seed)
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = Xtr.shape[0]
    steps = EPOCHS * math.ceil(n / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device=device).manual_seed(seed + 999)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb, tb = Xtr[idx], ytr_noisy[idx], teacher_soft[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            log_p = F.log_softmax(logits / KD_T, dim=1)
            kd = F.kl_div(log_p, tb, reduction="batchmean") * (KD_T * KD_T)
            ce = F.cross_entropy(logits, yb)
            loss = KD_ALPHA * kd + (1 - KD_ALPHA) * ce
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def acc_single(models, X, y, device, bs=16384):
    """mean per-model top-1 accuracy."""
    accs = []
    for m in models:
        correct = 0
        for i in range(0, X.shape[0], bs):
            correct += (m(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item()
        accs.append(correct / X.shape[0])
    return accs


@torch.no_grad()
def acc_ensemble(models, X, y, device, bs=16384):
    """top-1 of mean softmax (T=1)."""
    correct = 0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs]
        acc = torch.zeros(xb.shape[0], K, device=device)
        for m in models:
            acc += F.softmax(m(xb), dim=1)
        correct += (acc.argmax(1) == y[i:i+bs]).sum().item()
    return correct / X.shape[0]


def run_seed(Xtr, Xte, yte, ytr_clean, C, eps, seed, device):
    ytr_noisy = make_noisy(ytr_clean, eps, seed * 7 + 1, device)
    # teachers
    teachers = [train_teacher(Xtr, ytr_noisy, seed * 100 + t, device) for t in range(N_TEACHERS)]
    teacher_single = acc_single(teachers, Xte, yte, device)
    teacher_ens = acc_ensemble(teachers, Xte, yte, device)
    # distill: mean-teacher soft targets at temperature KD_T over the training set
    tsoft = soft_targets(teachers, Xtr, KD_T, device)
    students = [train_student(Xtr, ytr_noisy, tsoft, seed * 200 + s, device) for s in range(M_STUDENTS)]
    student_single = acc_single(students, Xte, yte, device)
    student_ens = acc_ensemble(students, Xte, yte, device)
    return dict(
        teacher_single=teacher_single, teacher_ens=teacher_ens,
        student_single=student_single, student_ens=student_ens,
        gap=student_ens - teacher_ens,
    )


def ci95(vals):
    a = np.array(vals, dtype=float)
    m = a.mean()
    if len(a) < 2:
        return m, 0.0
    sd = a.std(ddof=1)
    return m, 1.96 * sd / math.sqrt(len(a))


def run_cell(C, eps, seeds, gpu, outdir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_num_threads(2)
    device = get_device()
    t0 = time.time()
    Xtr, ytr_clean, Xte, yte = sample_world(C, device)
    per_seed = []
    for s in seeds:
        r = run_seed(Xtr, Xte, yte, ytr_clean, C, eps, s, device)
        per_seed.append(r)
        print(f"[C={C} eps={eps} seed={s}] tE={r['teacher_ens']:.4f} sE={r['student_ens']:.4f} gap={r['gap']:+.4f}", flush=True)
    def agg(key):
        m, ci = ci95([p[key] for p in per_seed])
        return dict(mean=m, ci95=ci)
    # single-model means: average the per-model lists first
    tsingle = [np.mean(p["teacher_single"]) for p in per_seed]
    ssingle = [np.mean(p["student_single"]) for p in per_seed]
    tm, tci = ci95(tsingle)
    sm, sci = ci95(ssingle)
    out = dict(
        C=C, eps=eps, seeds=list(seeds),
        N_teachers=N_TEACHERS, M_students=M_STUDENTS,
        n_train=N_TRAIN, n_test=N_TEST, d=D, K=K, H_sub=H_SUB, freq=FREQ,
        hidden=HIDDEN, n_layers=N_LAYERS, epochs=EPOCHS, batch=BATCH, lr=LR,
        kd_T=KD_T, kd_alpha=KD_ALPHA, gt_world_seed=GT_WORLD_SEED,
        teacher_ens=agg("teacher_ens"),
        student_ens=agg("student_ens"),
        teacher_single=dict(mean=tm, ci95=tci),
        student_single=dict(mean=sm, ci95=sci),
        gap=agg("gap"),
        per_seed=per_seed,
        seconds=time.time() - t0,
    )
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, f"cell_C{C}_eps{eps:.2f}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {fp}  gap={out['gap']['mean']:+.4f}+/-{out['gap']['ci95']:.4f}  ({out['seconds']:.1f}s)", flush=True)
    return out


def calibrate(gpu):
    """Quick coherence sanity: single teacher clean-fit acc at eps=0 across C."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_num_threads(2)
    device = get_device()
    for C in [1, 2, 4, 8]:
        Xtr, ytr_clean, Xte, yte = sample_world(C, device)
        # class balance of the ground truth
        counts = torch.bincount(yte, minlength=K).cpu().numpy()
        t = train_teacher(Xtr, ytr_clean, 0, device)
        a = acc_single([t], Xte, yte, device)[0]
        print(f"[D={D} K={K} FREQ={FREQ} H={H_SUB}] C={C}: clean 1-teacher test acc={a:.4f}  chance={1.0/K:.3f}  mincnt={counts.min()} maxcnt={counts.max()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["calibrate", "cell"])
    ap.add_argument("--C", type=int)
    ap.add_argument("--eps", type=float)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="/root/synth_coherence/cells")
    args = ap.parse_args()
    if args.mode == "calibrate":
        calibrate(args.gpu)
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
        run_cell(args.C, args.eps, seeds, args.gpu, args.outdir)

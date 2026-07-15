#!/usr/bin/env python3
"""E1 (PLAN_domain_extension.md): CIFAR-10N real-label-noise test of the
distill-then-ensemble noise-threshold claim (3rd domain after games).

One process = ONE noise level on ONE GPU:
  6 teachers (ResNet-18, distinct seeds, identical recipe, CE on noisy labels)
  -> (a) 2 disjoint 3-teacher mean-softmax ensembles, COMPOSITION-AVERAGED
  -> (b) 3 KD students (soft targets = 6-teacher mean softmax, alpha=0.7,
         0.3 * label-smoothed CE on the noisy label) -> 3-student ensemble
  -> (c) single student (mean of 3), (d) single teacher (mean of 6)
Eval: CLEAN test-set accuracy only. Identical recipe across all conditions.

No torchvision / no DataLoader: manual CIFAR pickle load + GPU-side
augmentation (pad4 random-crop + hflip) => zero CPU data workers
(shared box: heavy CPU gates running; do not starve them).
"""
import argparse, json, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- data ----------------

def load_cifar10(data_dir):
    base = os.path.join(data_dir, 'cifar-10-batches-py')
    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(base, f'data_batch_{i}'), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        xs.append(d[b'data'])
        ys.extend(d[b'labels'])
    xtr = np.concatenate(xs).reshape(-1, 3, 32, 32)
    ytr = np.array(ys, dtype=np.int64)
    with open(os.path.join(base, 'test_batch'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    xte = d[b'data'].reshape(-1, 3, 32, 32)
    yte = np.array(d[b'labels'], dtype=np.int64)
    return xtr, ytr, xte, yte


def load_cifar100(data_dir):
    base = os.path.join(data_dir, 'cifar-100-python')
    with open(os.path.join(base, 'train'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    xtr = d[b'data'].reshape(-1, 3, 32, 32)
    ytr = np.array(d[b'fine_labels'], dtype=np.int64)
    with open(os.path.join(base, 'test'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    xte = d[b'data'].reshape(-1, 3, 32, 32)
    yte = np.array(d[b'fine_labels'], dtype=np.int64)
    return xtr, ytr, xte, yte


def get_labels(noise, ytr_clean, data_dir, seed=12345, dataset='c10'):
    """Return (labels, source). CIFAR-N human labels if present and
    order-verified against the standard batch order, else synthetic uniform."""
    if noise == 'clean':
        return ytr_clean.copy(), 'clean'
    nc = 10 if dataset == 'c10' else 100
    fname = 'CIFAR-10_human.pt' if dataset == 'c10' else 'CIFAR-100_human.pt'
    npath = os.path.join(data_dir, fname)
    if os.path.exists(npath):
        h = torch.load(npath, map_location='cpu', weights_only=False)
        cl = np.asarray(h['clean_label'])
        if cl.shape == ytr_clean.shape and (cl == ytr_clean).all():
            key = ({'aggre': 'aggre_label', 'rand1': 'random_label1',
                    'rand2': 'random_label2', 'rand3': 'random_label3',
                    'worst': 'worse_label'} if dataset == 'c10'
                   else {'noisy': 'noisy_label', 'mix25': 'noisy_label',
                         'mix50': 'noisy_label', 'mix75': 'noisy_label'})[noise]
            y = np.asarray(h[key]).astype(np.int64)
            if noise.startswith('mix'):
                # keep fraction p of the REAL human-noisy labels, revert rest to clean
                p = int(noise[3:]) / 100.0
                keep = np.random.RandomState(777).rand(len(y)) < p
                y = np.where(keep, y, ytr_clean)
                print(f'mix {noise}: kept {keep.sum()} noisy; actual noise rate '
                      f'{(y != ytr_clean).mean():.4f}', flush=True)
            return y, f'cifar{nc}n'
        print(f'WARN: {fname} clean_label mismatch vs train order -> synthetic fallback', flush=True)
    rate = {'aggre': 0.10, 'rand1': 0.20, 'rand2': 0.18, 'rand3': 0.18, 'worst': 0.40, 'noisy': 0.40, 'mix25': 0.10, 'mix50': 0.20, 'mix75': 0.30}[noise]
    rng = np.random.RandomState(seed)
    y = ytr_clean.copy()
    flip = rng.rand(len(y)) < rate
    # uniform over the OTHER classes
    y[flip] = (y[flip] + 1 + rng.randint(0, nc - 1, size=int(flip.sum()))) % nc
    return y, f'synthetic_uniform_{rate:.2f}'

# ---------------- model (CIFAR ResNet-18) ----------------

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        cfg = [(64, 1), (64, 1), (128, 2), (128, 1),
               (256, 2), (256, 1), (512, 2), (512, 1)]
        layers, cin = [], 64
        for cout, stride in cfg:
            layers.append(BasicBlock(cin, cout, stride))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, nc)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

# ---------------- GPU-side augmentation ----------------

STATS = {
    'c10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'c100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def augment(x_u8, gen, mean, std):
    """x_u8: (B,3,32,32) uint8 on GPU. pad-4 random crop + hflip + normalize."""
    B, dev = x_u8.size(0), x_u8.device
    xp = F.pad(x_u8, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=dev, generator=gen)
    j = torch.randint(0, 9, (B,), device=dev, generator=gen)
    win = xp.unfold(2, 32, 1).unfold(3, 32, 1)            # (B,3,9,9,32,32) view
    x = win[torch.arange(B, device=dev), :, i, j]          # (B,3,32,32)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    x = x.float().div_(255.0).sub_(mean).div_(std)
    return x

# ---------------- train / eval ----------------

def train_model(seed, xtr_gpu, y_gpu, epochs, dev, mean, std, nc=10,
                teachers=None, alpha=0.7, bs=128, base_lr=0.1, log_tag=''):
    torch.manual_seed(seed)
    model = ResNet18(nc).to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    N = xtr_gpu.size(0)
    steps_per_ep = (N + bs - 1) // bs
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep)
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(N, device=dev, generator=gen)
        tot_loss, nb = 0.0, 0
        for k in range(0, N, bs):
            idx = perm[k:k + bs]
            xb = augment(xtr_gpu[idx], gen, mean, std)
            yb = y_gpu[idx]
            with torch.amp.autocast('cuda'):
                logits = model(xb)
                if teachers is None:
                    loss = F.cross_entropy(logits.float(), yb)
                else:
                    with torch.no_grad():
                        tp = torch.stack([F.softmax(t(xb).float(), dim=1)
                                          for t in teachers]).mean(0)
                    soft = -(tp * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
                    hard = F.cross_entropy(logits.float(), yb, label_smoothing=0.1)
                    loss = alpha * soft + (1.0 - alpha) * hard
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item()
            nb += 1
        print(f'[{log_tag}] ep {ep + 1}/{epochs} loss {tot_loss / nb:.4f} '
              f'{time.time() - t0:.1f}s', flush=True)
    return model


@torch.no_grad()
def test_probs(model, xte_norm):
    model.eval()
    ps = []
    for k in range(0, xte_norm.size(0), 1000):
        with torch.amp.autocast('cuda'):
            p = F.softmax(model(xte_norm[k:k + 1000]).float(), dim=1)
        ps.append(p.float())
    return torch.cat(ps)


def acc_of(probs, yte_gpu):
    return (probs.argmax(1) == yte_gpu).float().mean().item()

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', required=True,
                    choices=['clean', 'aggre', 'rand1', 'rand2', 'rand3', 'worst', 'noisy', 'mix25', 'mix50', 'mix75'])
    ap.add_argument('--dataset', default='c10', choices=['c10', 'c100'])
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--data', default='data')
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--n_teachers', type=int, default=6)
    ap.add_argument('--n_students', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0, help='subsample train (smoke)')
    ap.add_argument('--alpha', type=float, default=0.7)
    args = ap.parse_args()

    nc = 10 if args.dataset == 'c10' else 100
    valid = {'c10': ['clean', 'aggre', 'rand1', 'rand2', 'rand3', 'worst'], 'c100': ['clean', 'noisy', 'mix25', 'mix50', 'mix75']}
    assert args.noise in valid[args.dataset], f'{args.noise} invalid for {args.dataset}'
    tag = args.noise if args.dataset == 'c10' else f'c100_{args.noise}'

    torch.set_num_threads(2)  # shared box: heavy CPU gates running
    torch.backends.cudnn.benchmark = True
    dev = torch.device(f'cuda:{args.gpu}')
    m_, s_ = STATS[args.dataset]
    mean = torch.tensor(m_).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(s_).view(1, 3, 1, 1).to(dev)

    ckdir = os.path.join(args.workdir, 'ckpt')
    resdir = os.path.join(args.workdir, 'results')
    os.makedirs(ckdir, exist_ok=True)
    os.makedirs(resdir, exist_ok=True)

    loader = load_cifar10 if args.dataset == 'c10' else load_cifar100
    xtr, ytr_clean, xte, yte = loader(args.data)
    labels, source = get_labels(args.noise, ytr_clean, args.data, dataset=args.dataset)
    measured_rate = float((labels != ytr_clean).mean())
    if args.limit:
        xtr, labels, ytr_clean = xtr[:args.limit], labels[:args.limit], ytr_clean[:args.limit]

    xtr_gpu = torch.from_numpy(xtr).to(dev)                      # uint8
    y_gpu = torch.from_numpy(labels).to(dev)
    xte_norm = (torch.from_numpy(xte).to(dev).float().div_(255.0)
                .sub_(mean).div_(std))
    yte_gpu = torch.from_numpy(yte).to(dev)

    print(f'=== E1 dataset={args.dataset} noise={args.noise} source={source} '
          f'measured_rate={measured_rate:.4f} '
          f'n_train={len(labels)} epochs={args.epochs} gpu={args.gpu} ===', flush=True)
    t_start = time.time()

    # ---- teachers ----
    teachers, t_probs = [], []
    for s in range(args.n_teachers):
        ck = os.path.join(ckdir, f'{tag}_t{s}.pt')
        m = ResNet18(nc).to(dev)
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
            print(f'[teacher {s}] loaded ckpt', flush=True)
        else:
            m = train_model(s, xtr_gpu, y_gpu, args.epochs, dev, mean, std, nc=nc,
                            log_tag=f'{tag} T{s}')
            torch.save(m.state_dict(), ck)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        teachers.append(m)
        pr = test_probs(m, xte_norm)
        t_probs.append(pr)
        print(f'[teacher {s}] clean-test acc {acc_of(pr, yte_gpu):.4f}', flush=True)

    teacher_accs = [acc_of(p, yte_gpu) for p in t_probs]
    # composition-averaged disjoint trios (audit lesson: average trio compositions)
    k = min(3, len(t_probs) // 2)  # k=3 for the real 6-teacher run
    trioA = acc_of(torch.stack(t_probs[:k]).mean(0), yte_gpu)
    trioB = acc_of(torch.stack(t_probs[k:2 * k]).mean(0), yte_gpu)
    ens6 = acc_of(torch.stack(t_probs).mean(0), yte_gpu)

    # ---- students (KD from 6-teacher mean softmax) ----
    s_probs, student_accs = [], []
    for si in range(args.n_students):
        seed = 100 + si
        ck = os.path.join(ckdir, f'{tag}_s{si}.pt')
        m = ResNet18(nc).to(dev)
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
            print(f'[student {si}] loaded ckpt', flush=True)
        else:
            m = train_model(seed, xtr_gpu, y_gpu, args.epochs, dev, mean, std, nc=nc,
                            teachers=teachers, alpha=args.alpha,
                            log_tag=f'{tag} S{si}')
            torch.save(m.state_dict(), ck)
        pr = test_probs(m, xte_norm)
        s_probs.append(pr)
        student_accs.append(acc_of(pr, yte_gpu))
        print(f'[student {si}] clean-test acc {student_accs[-1]:.4f}', flush=True)

    student_ens3 = acc_of(torch.stack(s_probs).mean(0), yte_gpu)
    teacher_ens3 = 0.5 * (trioA + trioB)

    out = {
        'noise': args.noise,
        'dataset': args.dataset,
        'n_classes': nc,
        'label_source': source,
        'measured_noise_rate': measured_rate,
        'epochs': args.epochs,
        'n_train': int(len(labels)),
        'recipe': 'ResNet18-CIFAR, SGD lr0.1 mom0.9 wd5e-4 cosine, bs128, AMP, '
                  'pad4crop+hflip; KD: alpha*softCE(6-teacher mean softmax) + '
                  f'(1-alpha)*CE(noisy,ls=0.1), alpha={args.alpha}',
        'teacher_accs': teacher_accs,
        'single_teacher': float(np.mean(teacher_accs)),
        'single_teacher_std': float(np.std(teacher_accs)),
        'trio_accs': [trioA, trioB],
        'teacher_ens3': teacher_ens3,
        'teacher_ens3_spread': abs(trioA - trioB),
        'teacher_ens6': ens6,
        'student_accs': student_accs,
        'single_student': float(np.mean(student_accs)),
        'single_student_std': float(np.std(student_accs)),
        'student_ens3': student_ens3,
        'gap_student_ens3_minus_teacher_ens3': student_ens3 - teacher_ens3,
        'wall_hours': (time.time() - t_start) / 3600.0,
    }
    path = os.path.join(resdir, f'e1_{tag}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'WROTE {path}', flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()

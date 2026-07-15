"""Train one student by KL to the MEAN of the 6 teachers' action
distributions on the mixture buffer (distribution-averaging distillation)."""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import ACNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.buffer)
    states = data["states"]          # (N,10,10,C) uint8
    targets = data["mean_probs"]     # (N,A) float32, mean over 6 teachers
    n, _, _, c = states.shape
    a = targets.shape[1]

    net = ACNet(c, a).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    targets_t = torch.tensor(targets, device=device)
    # KL(target || student) = sum t*log t - sum t*log s ; first term constant
    target_entropy_term = float(
        (targets_t * torch.log(targets_t.clamp_min(1e-8))).sum(-1).mean())

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            mb = perm[i:i + args.batch_size]
            obs = torch.tensor(states[mb].astype(np.float32), device=device)
            logits, _ = net(obs)
            logp = F.log_softmax(logits, dim=-1)
            ce = -(targets_t[mb] * logp).sum(-1).mean()
            kl = ce + target_entropy_term
            opt.zero_grad()
            ce.backward()
            opt.step()
            tot += kl.item()
            nb += 1
        print(f"student s{args.seed} epoch {epoch}/{args.epochs} "
              f"mean_KL={tot/nb:.5f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"state_dict": net.state_dict(), "in_channels": c,
                "num_actions": a, "seed": args.seed,
                "epochs": args.epochs, "buffer_n": n}, args.out)
    print(f"DONE student seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()

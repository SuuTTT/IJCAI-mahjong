#!/usr/bin/env python3
"""E2 throughput probe: (a) shard load speed, (b) dummy ResNet fwd/bwd on GPU
over the real encoding, to size the training grid. Prints JSON."""
import argparse, glob, json, time
import numpy as np


def resnet(ch, blocks, n_actions=4672):
    import torch, torch.nn as nn

    class Block(nn.Module):
        def __init__(s, c):
            super().__init__()
            s.c1 = nn.Conv2d(c, c, 3, padding=1); s.b1 = nn.BatchNorm2d(c)
            s.c2 = nn.Conv2d(c, c, 3, padding=1); s.b2 = nn.BatchNorm2d(c)
        def forward(s, x):
            h = torch.relu(s.b1(s.c1(x)))
            return torch.relu(x + s.b2(s.c2(h)))

    return nn.Sequential(
        nn.Conv2d(18, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
        *[Block(ch) for _ in range(blocks)],
        nn.Conv2d(ch, 8, 1), nn.Flatten(), nn.Linear(8 * 64, n_actions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob of npz shards")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()

    files = sorted(glob.glob(a.shards))[:4]
    t0 = time.time()
    obs, act = [], []
    for f in files:
        z = np.load(f)
        obs.append(z["obs"]); act.append(z["action"])
    obs = np.concatenate(obs); act = np.concatenate(act)
    load_s = time.time() - t0
    out = {"shards_loaded": len(files), "positions": int(len(act)),
           "load_s": round(load_s, 2),
           "load_positions_per_s": round(len(act) / load_s, 0)}

    import torch
    dev = a.device if torch.cuda.is_available() else "cpu"
    out["device"] = dev
    if dev.startswith("cuda"):
        out["gpu_name"] = torch.cuda.get_device_name(torch.device(dev))
    net = resnet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    n = len(act)
    def batch():
        i = np.random.randint(0, n - a.batch)
        x = torch.from_numpy(obs[i:i + a.batch]).float().to(dev)
        y = torch.from_numpy(act[i:i + a.batch].astype(np.int64)).to(dev)
        return x, y
    # warmup
    for _ in range(5):
        x, y = batch(); loss = lossf(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.iters):
        x, y = batch(); loss = lossf(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    out.update({"model": f"resnet c{a.channels} b{a.blocks}", "batch": a.batch,
                "train_iters_per_s": round(a.iters / dt, 2),
                "train_positions_per_s": round(a.iters * a.batch / dt, 0)})
    # inference-only
    net.eval()
    with torch.no_grad():
        x, _ = batch()
        for _ in range(5):
            net(x)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.iters):
            net(x)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
    out["infer_positions_per_s"] = round(a.iters * a.batch / dt, 0)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

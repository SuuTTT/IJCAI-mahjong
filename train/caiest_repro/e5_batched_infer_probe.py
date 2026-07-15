"""E5 (reshaped): batched-GPU-inference throughput probe.
Ludus measured model inference ~100x env cost -> the lever is a central batched
inference server. Quantify: kdens3-size fused net (128x40) forwards/s under
(a) CPU single-stream (the current gate-sim mode), (b) CPU batched,
(c) GPU batched at B in {64,256,1024,4096}. Output: results/E5_BATCHED_INFER.json
with projected speedup for a 24k-game gate and for RL self-play."""
import json, time, torch, numpy as np, os, sys
sys.path.insert(0, "/root/caiest_repro")
import models_explore
from models_explore import build

CKPT = "ckpt/kd/kd_128x40_s0.pkl"
net = build("resbn_fused", channels=128, blocks=40)
net.load_state_dict(torch.load(CKPT, map_location="cpu")); net.eval()

def feed(B, dev):
    return {"is_training": False, "obs": {
        "observation": torch.randn(B, 38, 4, 9, device=dev),
        "action_mask": torch.ones(B, 235, device=dev)}}

def bench(dev, B, iters, warm=5):
    m = net.to(dev)
    with torch.no_grad():
        for _ in range(warm): m(feed(B, dev))
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters): m(feed(B, dev))
        if dev == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    return dict(batch=B, iters=iters, sec=round(dt, 3),
                forwards_per_s=round(B * iters / dt, 1),
                ms_per_forward=round(1000 * dt / (B * iters), 4))

torch.set_num_threads(1)
res = {"cpu_single_1thread": bench("cpu", 1, 40)}
torch.set_num_threads(8)
res["cpu_batch256_8thread"] = bench("cpu", 256, 4)
for B in (64, 256, 1024, 4096):
    res[f"gpu_batch{B}"] = bench("cuda", B, 60 if B <= 256 else 20)
base = res["cpu_single_1thread"]["forwards_per_s"]
best = max(v["forwards_per_s"] for k, v in res.items() if k.startswith("gpu"))
# a 24k-game gate makes ~24000 games * ~120 decisions * 4 seats forwards
gate_fwd = 24000 * 120 * 4
res["projection"] = dict(
    speedup_gpu_vs_cpu_single=round(best / base, 1),
    gate_24k_forwards=gate_fwd,
    gate_hours_cpu100proc=round(gate_fwd / (base * 100) / 3600, 2),
    gate_hours_1gpu_batched=round(gate_fwd / best / 3600, 2),
    note="batched server also serves RL self-play; env cost ~1% of inference (Ludus measurement)")
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/E5_BATCHED_INFER.json", "w"), indent=1)
print(json.dumps(res["projection"], indent=1))

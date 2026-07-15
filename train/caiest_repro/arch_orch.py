"""
arch_orch.py — gate each exploration arch vs aug_s0 (calibrated placement gate, multi-block CI),
measure params + val + per-move ms (TLE), then aggregate -> ARCH_RESULTS.json + ARCH_WRITEUP.md.
Idempotent: skips gate cells that already exist. Honors /root/STOP_ARCH + a disk floor.
Run detached:  setsid python3 arch_orch.py --loop >/root/arch_orch.log 2>&1 &
"""
import os, sys, json, glob, time, math, subprocess, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); GD = os.path.join(HERE, "ckpt", "archx", "gates")
os.makedirs(GD, exist_ok=True)
REF = "ckpt/aug/aug_128x40_s0.pkl"
NB = 6; SEEDS = 500; WORKERS = 48
STOP = "/root/STOP_ARCH"; DISK_FLOOR_GB = 10
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179}

# name -> (gate, kind_or_None, cfg, ckpt, seed0_base)
ARCHS = {
 "attn_s0":    ("e11",  "attn",     "d_model=256,layers=8,heads=8",                 "ckpt/archx/attn_s0.pkl",    810000),
 "cnnattn_s0": ("e11",  "cnn_attn", "channels=192,conv_blocks=8,layers=6,heads=8",  "ckpt/archx/cnnattn_s0.pkl", 820000),
 "gnn_s0":     ("e11",  "gnn",      "hidden=384,layers=6",                          "ckpt/archx/gnn_s0.pkl",     830000),
 "temporal_s0":("seq",  None,       "channels=128,blocks=40,emb=64,gru=256",        "ckpt/archx/temporal_s0.pkl",900000),
}
CALIB = ("e11", "resbn_fused", "channels=128,blocks=40", REF, 800000)


def disk_gb():
    st = os.statvfs("/"); return st.f_bavail * st.f_frsize / 1e9

def stopped():
    return os.path.exists(STOP) or disk_gb() < DISK_FLOOR_GB

def val_from_log(name):
    lg = f"/root/archx_{name}.log"
    best = None
    if os.path.exists(lg):
        for line in open(lg):
            if "best_ema_val=" in line:
                best = float(line.split("best_ema_val=")[1].split()[0])
            elif "best " in line and "emaval" in line:
                try: best = max(best or 0, float(line.split("best")[1].split()[0]))
                except Exception: pass
    return best

def params_of(gate, kind, cfg, ckpt):
    d = dict(kv.split("=") for kv in cfg.split(",")); d = {k:int(v) for k,v in d.items()}
    if gate == "seq":
        from models_seq import build_seq; m = build_seq("temporal", **d)
    else:
        from models_explore import build; m = build(kind, **d)
    return sum(p.numel() for p in m.parameters())

def tle_ms(gate, kind, cfg, ckpt):
    d = dict(kv.split("=") for kv in cfg.split(",")); d = {k:int(v) for k,v in d.items()}
    torch.set_num_threads(1)
    dd = np.load(os.path.join(HERE, "data", "cooked_single.npz")); o = dd["obs"][:1]; mk = dd["mask"][:1]
    if gate == "seq":
        from models_seq import build_seq; m = build_seq("temporal", **d).eval()
        m.load_state_dict(torch.load(os.path.join(HERE, ckpt), map_location="cpu"))
        sq = torch.zeros(1, 48, dtype=torch.long)
        inp = {"is_training": False, "seq": sq, "obs": {"observation": torch.from_numpy(o).float(),
                "action_mask": torch.from_numpy(mk).float()}}
    else:
        from models_explore import build; m = build(kind, **d).eval()
        m.load_state_dict(torch.load(os.path.join(HERE, ckpt), map_location="cpu"))
        inp = {"is_training": False, "obs": {"observation": torch.from_numpy(o).float(),
                "action_mask": torch.from_numpy(mk).float()}}
    with torch.no_grad():
        for _ in range(3): m(inp)
        t0 = time.time()
        for _ in range(60): m(inp)
    return round((time.time() - t0) / 60 * 1000, 2)

def run_cell(gate, kind, cfg, ckpt, out, seed0):
    if gate == "seq":
        cmd = ["python3","gate_seq.py","--cand",ckpt,"--cand-cfg",cfg,"--ref",REF,
               "--seeds",str(SEEDS),"--workers",str(WORKERS),"--seed0",str(seed0),"--out",out]
    else:
        cmd = ["python3","e11_gate.py","--cand",ckpt,"--cand-kind",kind,"--cand-cfg",cfg,
               "--ref",REF,"--ref-kind","resbn_fused","--ref-cfg","channels=128,blocks=40",
               "--seeds",str(SEEDS),"--workers",str(WORKERS),"--seed0",str(seed0),"--out",out]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run(cmd, cwd=HERE, env=env, check=False)

def train_done(name):
    lg = f"/root/archx_{name}.log"
    return os.path.exists(lg) and any("DONE kind=" in ln for ln in open(lg))

def gate_arch(name, spec):
    gate, kind, cfg, ckpt, s0 = spec
    if not os.path.exists(os.path.join(HERE, ckpt)) or not train_done(name):
        return False
    done = 0
    for i in range(NB):
        if stopped(): print(f"[stop] {name}", flush=True); break
        out = os.path.join(GD, f"{name}_s{s0+1000*i}.json")
        if os.path.exists(out): done += 1; continue
        print(f"[gate] {name} cell {i} seed0={s0+1000*i}", flush=True)
        run_cell(gate, kind, cfg, ckpt, out, s0+1000*i)
        if os.path.exists(out): done += 1
    return done >= NB

def calib():
    gate, kind, cfg, ckpt, s0 = CALIB
    out = os.path.join(GD, f"calib_s{s0}.json")
    if not os.path.exists(out) and not stopped():
        print("[calib] aug_s0 vs aug_s0", flush=True)
        run_cell(gate, kind, cfg, ckpt, out, s0)

def _t(n):
    dfn = n-1
    return TCRIT.get(dfn, 2.101)

def aggregate():
    calib_pts = None
    cf = os.path.join(GD, f"calib_s{CALIB[4]}.json")
    if os.path.exists(cf): calib_pts = json.load(open(cf))["placement_pts"]
    res = {"reference": "aug_s0 = aug_128x40_s0.pkl (deployed best 128x40); calibrated gate 2.500=tied; "
           "BEAT iff placement 95% CI lower bound > 2.500", "calib_augs0_vs_augs0": calib_pts,
           "gate_blocks_target": NB, "seeds_per_block": SEEDS, "archs": {}}
    for name, spec in ARCHS.items():
        gate, kind, cfg, ckpt, s0 = spec
        cells = sorted(glob.glob(os.path.join(GD, f"{name}_s*.json")))
        vals = [json.load(open(c))["placement_pts"] for c in cells]
        entry = {"gate": gate, "kind": kind or "temporal", "cfg": cfg,
                 "ckpt": ckpt, "trained": os.path.exists(os.path.join(HERE, ckpt)),
                 "val_acc": val_from_log(name), "n_blocks": len(vals)}
        if entry["trained"]:
            try: entry["params_M"] = round(params_of(gate, kind, cfg, ckpt)/1e6, 2)
            except Exception as e: entry["params_M"] = f"err:{e}"
            try: entry["per_move_ms"] = tle_ms(gate, kind, cfg, ckpt)
            except Exception as e: entry["per_move_ms"] = f"err:{e}"
            entry["tle_safe_1000ms"] = isinstance(entry.get("per_move_ms"), float) and entry["per_move_ms"] <= 1000
        if len(vals) >= 2:
            mean = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)); se = sd/math.sqrt(len(vals))
            h = _t(len(vals))*se
            entry.update(placement_mean=round(mean,4), placement_sd=round(sd,4),
                         ci95_lo=round(mean-h,4), ci95_hi=round(mean+h,4),
                         margin_lo=round(mean-h-2.5,4),
                         beats_augs0=bool(mean-h > 2.5),
                         worse_than_augs0=bool(mean+h < 2.5),
                         verdict=("BEATS" if mean-h>2.5 else ("WORSE" if mean+h<2.5 else "TIED_NOT_SEPARATED")),
                         block_placements=[round(v,4) for v in vals])
        res["archs"][name] = entry
    with open(os.path.join(HERE, "ARCH_RESULTS.json"), "w") as f: json.dump(res, f, indent=2)
    write_md(res)
    print("[agg] wrote ARCH_RESULTS.json + ARCH_WRITEUP.md", flush=True)
    return res

def write_md(res):
    L = ["# Architecture exploration vs aug_s0 — results\n",
         f"Gate: calibrated duplicate placement, aug_s0-vs-aug_s0 = {res.get('calib_augs0_vs_augs0')} (2.500=tied). "
         "A candidate BEATS aug_s0 iff its placement 95% CI lower bound > 2.500. Lower CNN/arch = same 38-plane "
         "caiest feature + the e11 enhanced aug recipe (suit x reflect x dragon, label-smoothing, EMA). "
         "val_acc uses the SAME rng-12345 val split as aug_s0 (0.887).\n",
         "| arch | params(M) | val_acc | per_move_ms | TLE<=1s | blocks | placement | 95% CI | beats aug_s0 | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in res["archs"].items():
        if "placement_mean" in e:
            L.append(f"| {name} ({e['cfg']}) | {e.get('params_M')} | {e.get('val_acc')} | {e.get('per_move_ms')} | "
                     f"{e.get('tle_safe_1000ms')} | {e['n_blocks']} | {e['placement_mean']} | "
                     f"[{e['ci95_lo']}, {e['ci95_hi']}] | {e.get('beats_augs0')} | {e['verdict']} |")
        else:
            L.append(f"| {name} ({e['cfg']}) | {e.get('params_M','-')} | {e.get('val_acc','-')} | "
                     f"{e.get('per_move_ms','-')} | - | {e['n_blocks']} | (gating/incomplete) | - | - | - |")
    L += ["", "## Deployability", "- attn / cnn_attn / gnn: research-only for the numpy-fused Botzone bot "
          "(no BN-fold path; transformer/GNN ops not in the fused kernel). TLE-safe on 1 core if deployed via torch.",
          "- temporal (CNN+GRU): research-only unless the deploy bot is extended to emit the ordered discard "
          "sequence; the CNN branch alone is BN-fuseable. TLE-safe (~CNN+small GRU).",
          "", "## Context (already-run axes from the campaign, verified)",
          "- Enhanced FEATURES (#3): 44-plane enh_192x40 / enh_384x40 + featA/B/C ablations -> all TIED "
          "(BESTNET_RESULTS.json). Richer features did not CI-beat aug_s0.",
          "- CAPACITY (#5): raw192 / raw384 / big256 / big320 -> all TIED. Bigger CNN did not CI-beat aug_s0.",
          "", "## Verdict", "(auto-filled at completion — see JSON verdict fields per arch)"]
    open(os.path.join(HERE, "ARCH_WRITEUP.md"), "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true"); a = ap.parse_args()
    while True:
        calib()
        for name, spec in ARCHS.items():
            if stopped(): break
            gate_arch(name, spec)
        aggregate()
        all_done = all(len(glob.glob(os.path.join(GD, f"{n}_s*.json"))) >= NB
                       for n, s in ARCHS.items() if os.path.exists(os.path.join(HERE, s[3])))
        trained_all = all(os.path.exists(os.path.join(HERE, s[3])) for s in ARCHS.values())
        if a.once or stopped() or (all_done and trained_all): break
        time.sleep(120)
    print("[orch] done", flush=True)

if __name__ == "__main__":
    main()

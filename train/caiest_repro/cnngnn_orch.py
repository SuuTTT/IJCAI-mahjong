"""
cnngnn_orch.py - gate the CNN+GNN HYBRID (resbn_gnn) vs aug_s0 (calibrated placement gate,
multi-block t-CI), measure params + val + per-move ms (TLE), aggregate -> CNNGNN_RESULTS.json +
CNNGNN_WRITEUP.md. Idempotent (skips existing gate cells). Honors /root/STOP_CNNGNN, a disk
floor, and a 1-min load-average guard (this box was thrashed once). CPU gate.
  setsid python3 cnngnn_orch.py --loop >/root/cnngnn_orch.log 2>&1
"""
import os, sys, json, glob, time, math, subprocess, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); GD = os.path.join(HERE, "ckpt", "cnngnn", "gates")
os.makedirs(GD, exist_ok=True)
REF = "ckpt/aug/aug_128x40_s0.pkl"
KIND = "resbn_gnn_fused"; CFG = "channels=128,blocks=40"
NB = 6; SEEDS = 500; WORKERS = 40
STOP = "/root/STOP_CNNGNN"; DISK_FLOOR_GB = 8; LOAD_MAX = 22.0
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179}
# seed -> (ckpt, seed0-base for gate blocks)
SEEDS_MAP = {0: ("ckpt/cnngnn/cnngnn_s0.pkl", 8400000), 1: ("ckpt/cnngnn/cnngnn_s1.pkl", 8500000)}

def disk_gb():
    st = os.statvfs("/"); return st.f_bavail * st.f_frsize / 1e9
def load1():
    return os.getloadavg()[0]
def stopped():
    return os.path.exists(STOP) or disk_gb() < DISK_FLOOR_GB
def wait_load():
    while load1() > LOAD_MAX and not stopped():
        print("[load-guard] 1-min load %.1f > %s; waiting" % (load1(), LOAD_MAX), flush=True); time.sleep(60)

def train_done(seed):
    lg = "/root/cnngnn_s%d.log" % seed
    return os.path.exists(lg) and any("DONE best_ema_val=" in ln for ln in open(lg))
def val_from_log(seed):
    lg = "/root/cnngnn_s%d.log" % seed; best = None
    if os.path.exists(lg):
        for line in open(lg):
            if "best_ema_val=" in line:
                try: best = float(line.split("best_ema_val=")[1].split()[0])
                except Exception: pass
            elif "best " in line and "emaval" in line:
                try: best = max(best or 0, float(line.split("best")[1].split()[0]))
                except Exception: pass
    return best

def params_of():
    from models_explore import build
    d = {k:int(v) for k,v in (kv.split("=") for kv in CFG.split(","))}
    return sum(p.numel() for p in build(KIND, **d).parameters())

def tle_ms(ckpt):
    from models_explore import build
    d = {k:int(v) for k,v in (kv.split("=") for kv in CFG.split(","))}
    torch.set_num_threads(1)
    dd = np.load(os.path.join(HERE, "data", "cooked_single.npz")); o=dd["obs"][:1]; mk=dd["mask"][:1]
    m = build(KIND, **d).eval(); m.load_state_dict(torch.load(os.path.join(HERE,ckpt), map_location="cpu"))
    inp = {"is_training": False, "obs": {"observation": torch.from_numpy(o).float(),
            "action_mask": torch.from_numpy(mk).float()}}
    with torch.no_grad():
        for _ in range(3): m(inp)
        t0=time.time()
        for _ in range(60): m(inp)
    return round((time.time()-t0)/60*1000, 2)

def run_cell(ckpt, out, seed0):
    cmd = ["python3","e11_gate.py","--cand",ckpt,"--cand-kind",KIND,"--cand-cfg",CFG,
           "--ref",REF,"--ref-kind","resbn_fused","--ref-cfg","channels=128,blocks=40",
           "--seeds",str(SEEDS),"--workers",str(WORKERS),"--seed0",str(seed0),"--out",out]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run(cmd, cwd=HERE, env=env, check=False)

def gate_seed(seed):
    ckpt, s0 = SEEDS_MAP[seed]
    if not os.path.exists(os.path.join(HERE, ckpt)) or not train_done(seed): return
    for i in range(NB):
        if stopped(): print("[stop] s%d" % seed, flush=True); break
        out = os.path.join(GD, "s%d_c%d.json" % (seed, i))
        if os.path.exists(out): continue
        wait_load()
        if stopped(): break
        print("[gate] s%d cell %d seed0=%d (load %.1f)" % (seed, i, s0+1000*i, load1()), flush=True)
        run_cell(ckpt, out, s0+1000*i)

def _t(n): return TCRIT.get(n-1, 2.101)

def aggregate():
    res = {"experiment":"CNN+GNN HYBRID (parallel GNN branch concatenated with CNN 512-vec before head) vs aug_s0",
           "graph_spec":{"nodes":34,"node_features":"38-plane per-tile-type feature vector from (38,4,9) obs (counts/ownership)",
             "edges":"within-suit chi sequence-adjacency r+-1 & r+-2 + honor 7-clique + self-loop (peng/same-tile); sym-normalized",
             "message_passing_layers":3,"gnn_hidden":128,"gnn_emb":128,"pool":"mean over 34 nodes","fusion":"concat[cnn_512, gnn_emb128] -> Linear(640,235)"},
           "reference":"aug_s0 = aug_128x40_s0.pkl; calibrated gate 2.500=tied (aug_s0-vs-aug_s0); BEAT iff placement 95% CI lower bound > 2.500",
           "gate_blocks_target":NB,"seeds_per_block":SEEDS,
           "context_standalone_gnn":{"kind":"gnn (REPLACES cnn)","val_acc":0.7697,"placement":2.3053,"verdict":"WORSE (-0.212)"},
           "seeds":{}}
    try: res["params_M"] = round(params_of()/1e6, 2)
    except Exception as e: res["params_M"] = "err:%s" % e
    for seed,(ckpt,s0) in SEEDS_MAP.items():
        if not os.path.exists(os.path.join(HERE, ckpt)): continue
        cells = sorted(glob.glob(os.path.join(GD, "s%d_c*.json" % seed)))
        vals = [json.load(open(c))["placement_pts"] for c in cells]
        e = {"ckpt":ckpt,"trained":train_done(seed),"val_acc":val_from_log(seed),"n_blocks":len(vals)}
        try:
            e["per_move_ms"] = tle_ms(ckpt); e["tle_safe_1000ms"] = e["per_move_ms"] <= 1000
        except Exception as ex:
            e["per_move_ms"] = "err:%s" % ex
        if len(vals) >= 2:
            mean=float(np.mean(vals)); sd=float(np.std(vals,ddof=1)); se=sd/math.sqrt(len(vals)); h=_t(len(vals))*se
            e.update(placement_mean=round(mean,4), placement_sd=round(sd,4),
                     ci95_lo=round(mean-h,4), ci95_hi=round(mean+h,4), margin_lo=round(mean-h-2.5,4),
                     beats_augs0=bool(mean-h>2.5), worse_than_augs0=bool(mean+h<2.5),
                     verdict=("BEATS" if mean-h>2.5 else ("WORSE" if mean+h<2.5 else "TIES")),
                     block_placements=[round(v,4) for v in vals])
        res["seeds"]["s%d" % seed] = e
    done = [e for e in res["seeds"].values() if "verdict" in e]
    if done:
        v0 = done[0]["verdict"]
        res["overall_verdict"] = ("HYBRID "+v0+" aug_s0" if all(d["verdict"]==v0 for d in done)
                                  else "mixed: "+", ".join(d["verdict"] for d in done))
    with open(os.path.join(HERE,"CNNGNN_RESULTS.json"),"w") as f: json.dump(res,f,indent=2)
    write_md(res); print("[agg] wrote CNNGNN_RESULTS.json + CNNGNN_WRITEUP.md", flush=True); return res

def write_md(res):
    g=res["graph_spec"]
    L=["# CNN+GNN HYBRID vs aug_s0 - results\n",
       "**Question:** does a GNN as a PARALLEL feature branch (concatenated with the CNN features "
       "before the heads, NOT a replacement) BEAT / TIE / WORSEN aug_s0?\n",
       "## Graph spec",
       "- nodes: %s tile-types; node features: %s" % (g["nodes"], g["node_features"]),
       "- edges: %s" % g["edges"],
       "- %s message-passing layers, hidden %s, emb %s, pool=%s" % (g["message_passing_layers"], g["gnn_hidden"], g["gnn_emb"], g["pool"]),
       "- fusion: %s (CNN backbone = aug_s0 arch, UNCHANGED)" % g["fusion"],
       "- params: %s M\n" % res.get("params_M"),
       "## Gate", "Calibrated duplicate placement, aug_s0-vs-aug_s0 = 2.500 (tied). HYBRID BEATS iff "
       "placement 95%% CI lower bound > 2.500. %d blocks x %d seeds (t-CI). e11 enhanced aug "
       "recipe (suit x reflect x dragon, label-smoothing, EMA) - the SAME recipe as aug_s0.\n" % (NB, SEEDS),
       "| seed | val_acc | per_move_ms | TLE<=1s | blocks | placement | 95% CI | beats aug_s0 | verdict |",
       "|---|---|---|---|---|---|---|---|---|"]
    for s,e in res["seeds"].items():
        if "placement_mean" in e:
            L.append("| %s | %s | %s | %s | %d | %s | [%s, %s] | %s | %s |" % (
                s, e.get("val_acc"), e.get("per_move_ms"), e.get("tle_safe_1000ms"),
                e["n_blocks"], e["placement_mean"], e["ci95_lo"], e["ci95_hi"],
                e.get("beats_augs0"), e["verdict"]))
        else:
            L.append("| %s | %s | %s | - | %d | (gating/incomplete) | - | - | - |" % (
                s, e.get("val_acc","-"), e.get("per_move_ms","-"), e["n_blocks"]))
    c=res["context_standalone_gnn"]
    L += ["", "## Context: the STANDALONE GNN (prior)",
          "- kind gnn REPLACED the CNN on the fixed 34-tile graph: val %s, placement %s -> %s." % (c["val_acc"], c["placement"], c["verdict"]),
          "- Hypothesis for the hybrid: keeping the CNN strength and only ADDING relational info "
          "should NOT worsen (worst case ties aug_s0, since the head can zero the GNN branch).",
          "", "## Verdict", res.get("overall_verdict","(gating in progress)")]
    open(os.path.join(HERE,"CNNGNN_WRITEUP.md"),"w").write("\n".join(L))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--loop",action="store_true"); ap.add_argument("--once",action="store_true")
    a=ap.parse_args()
    while True:
        for seed in SEEDS_MAP:
            if stopped(): break
            gate_seed(seed)
        aggregate()
        all_done = all(len(glob.glob(os.path.join(GD,"s%d_c*.json" % s)))>=NB
                       for s in SEEDS_MAP if os.path.exists(os.path.join(HERE,SEEDS_MAP[s][0])))
        any_ckpt = any(os.path.exists(os.path.join(HERE,SEEDS_MAP[s][0])) for s in SEEDS_MAP)
        if a.once or stopped() or (all_done and any_ckpt): break
        time.sleep(120)
    print("[orch] done", flush=True)

if __name__=="__main__":
    main()

"""
feat_orch.py — self-managing orchestrator for the +FEATURES (featplus ABC x e11 recipe) queue.

Priority #1 of the arch-continuation mandate: train the deployable 128x40 CNN with mechanism
feature planes (danger/genbutsu-safe + shanten/useful-tile) using the PROVEN e11 enhanced recipe,
then gate each seed vs aug_s0 with the no-op-guarded parity_gate_plus (edge_per_game, 0=tied;
BEAT iff edge 95% CI lower bound > 0).

HARD ANTI-THRASH DISCIPLINE (never re-thrash the box):
  * STRICT <=4 total training jobs (counts arch_bc/seq_bc/e11_train/e11plus_train/train_plus).
  * Launch a train only on a GPU with <800MB used AND 1-min load < LOAD_TRAIN, then confirm the
    process is alive before scanning again.
  * Run at most ONE gate at a time, workers<=24, and ONLY when NO other gate (e11_gate/gate_seq/
    parity_gate*) is running AND 1-min load < LOAD_GATE.
  * Honor /root/STOP_ARCHCONT and a disk floor. Coexist with arch_orch (do not fight it).
  When in doubt -> WAIT.
"""
import os, sys, json, glob, time, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GD = os.path.join(HERE, "ckpt", "featx", "gates"); os.makedirs(GD, exist_ok=True)
os.makedirs(os.path.join(HERE, "ckpt", "featx"), exist_ok=True)

STOP = "/root/STOP_ARCHCONT"
DISK_FLOOR_GB = 8
LOAD_TRAIN = 20.0      # don't launch a training if 1-min load exceeds this
LOAD_GATE = 18.0       # don't launch a gate block if 1-min load exceeds this
SETS = "ABC"
SEEDS = [0, 1]
REF_BN = "ckpt/aug/aug_128x40_s0.bn.pkl"   # unfused aug_s0 (ResBNCNN) -> loads via _load_base
STEPS = 130000
GATE_BLOCKS = 8
GAMES = 300            # duplicate pairs per block -> 600 games/block
WORKERS = 24
LOG = "/root/feat_orch.log"
TRAIN_PATS = ("e11plus_train.py", "e11_train.py", "arch_bc.py", "seq_bc.py", "train_plus.py")
GATE_PATS = ("e11_gate.py", "gate_seq.py", "parity_gate")
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179}


def log(*x):
    s = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), " ".join(str(i) for i in x))
    print(s, flush=True)
    try:
        open(LOG, "a").write(s + "\n")
    except Exception:
        pass


def disk_gb():
    st = os.statvfs("/"); return st.f_bavail * st.f_frsize / 1e9


def load1():
    return os.getloadavg()[0]


def _pgrep_count(pats):
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return 0
    n = 0
    for ln in out.splitlines():
        if "feat_orch.py" in ln:      # never count self
            continue
        if any(p in ln for p in pats):
            n += 1
    return n


def n_trainings():
    return _pgrep_count(TRAIN_PATS)


def gate_running():
    # any gate process other than our own current subprocess counts; we run gates synchronously
    # so we only START a block when this is 0.
    return _pgrep_count(GATE_PATS) > 0


def seed_running(seed):
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return False
    tag = "e11plus_train.py"
    for ln in out.splitlines():
        if tag in ln and ("--seed %d" % seed) in ln and ("_s%d." % seed) in ln:
            return True
    return False


def free_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None
    for ln in out.strip().splitlines():
        idx, mem = [t.strip() for t in ln.split(",")]
        if int(mem) < 800:
            return int(idx)
    return None


def train_out(seed):
    return "ckpt/featx/featABC_e11_s%d.pkl" % seed


def train_bn(seed):
    return train_out(seed)[:-4] + ".bn.pkl"


def train_done(seed):
    lg = "/root/featABC_s%d.log" % seed
    if os.path.exists(lg):
        try:
            return any("DONE sets=" in ln for ln in open(lg))
        except Exception:
            return False
    return False


def val_from_log(seed):
    lg = "/root/featABC_s%d.log" % seed
    best = None
    if os.path.exists(lg):
        for ln in open(lg):
            if "best_ema_val=" in ln:
                try: best = float(ln.split("best_ema_val=")[1].split()[0])
                except Exception: pass
            elif "best " in ln and "emaval" in ln:
                try: best = max(best or 0.0, float(ln.split("best")[1].split()[0]))
                except Exception: pass
    return best


def stopped():
    return os.path.exists(STOP) or disk_gb() < DISK_FLOOR_GB


def launch_train(seed):
    gpu = free_gpu()
    if gpu is None:
        return False
    out = train_out(seed)
    lg = "/root/featABC_s%d.log" % seed
    cmd = ["setsid", "python3", "e11plus_train.py", "--sets", SETS, "--channels", "128",
           "--blocks", "40", "--steps", str(STEPS), "--seed", str(seed),
           "--valevery", "5000", "--out", out]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log("LAUNCH train seed=%d on GPU%d (ntrain=%d load=%.1f)" % (seed, gpu, n_trainings(), load1()))
    with open(lg, "a") as f:
        subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    # confirmation wait: data load (~1-2min) precedes CUDA alloc; confirm the process is alive.
    time.sleep(60)
    if seed_running(seed):
        log("  confirmed seed=%d alive on GPU%d" % (seed, gpu))
        return True
    log("  WARNING seed=%d not alive after launch (check %s)" % (seed, lg))
    return False


def run_gate_block(seed, blk):
    s0 = 60000 + seed * 20000 + blk * 1000
    out = os.path.join(GD, "featABC_s%d_b%d.json" % (seed, blk))
    if os.path.exists(out):
        return True
    cmd = ["python3", "parity_gate_plus.py", "--cand", train_bn(seed), "--ref", REF_BN,
           "--sets", SETS, "--blocks", "40", "--games", str(GAMES), "--workers", str(WORKERS),
           "--seed0", str(s0), "--out", out]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    log("GATE block seed=%d blk=%d seed0=%d (load=%.1f)" % (seed, blk, s0, load1()))
    subprocess.run(cmd, cwd=HERE, env=env, check=False)
    return os.path.exists(out)


def aggregate():
    res = {"experiment": "FEATURES: featplus(ABC = shanten/useful + danger + genbutsu-safe) x e11 "
           "enhanced recipe on deployable 128x40, vs aug_s0.",
           "gate": "parity_gate_plus (no-op-guarded; edge_per_game score edge, 0=tied). "
           "BEAT iff edge 95% CI lower bound > 0.  NOTE: this is the score-edge gate (native featplus "
           "gate), distinct from the arch campaign's calibrated placement gate (2.500=tied).",
           "sets": SETS, "ref": REF_BN, "steps": STEPS, "seeds": {}}
    for seed in SEEDS:
        entry = {"trained": os.path.exists(os.path.join(HERE, train_bn(seed))),
                 "train_done": train_done(seed), "val_acc": val_from_log(seed),
                 "running": seed_running(seed)}
        cells = sorted(glob.glob(os.path.join(GD, "featABC_s%d_b*.json" % seed)))
        edges = []; guard = None
        for c in cells:
            try:
                j = json.load(open(c))
                edges.append(j["edge_per_game"]); guard = j.get("noop_guard", guard)
            except Exception:
                pass
        entry["n_blocks"] = len(edges); entry["noop_guard"] = guard
        entry["block_edges"] = [round(e, 4) for e in edges]
        if len(edges) >= 2:
            mean = float(np.mean(edges)); sd = float(np.std(edges, ddof=1))
            se = sd / math.sqrt(len(edges)); h = TCRIT.get(len(edges) - 1, 2.101) * se
            entry.update(edge_mean=round(mean, 4), edge_sd=round(sd, 4),
                         ci95_lo=round(mean - h, 4), ci95_hi=round(mean + h, 4),
                         beats_augs0=bool(mean - h > 0.0),
                         worse_than_augs0=bool(mean + h < 0.0),
                         verdict=("BEATS" if mean - h > 0 else ("WORSE" if mean + h < 0 else "TIED_NOT_SEPARATED")))
        res["seeds"]["s%d" % seed] = entry
    with open(os.path.join(HERE, "FEATURES_RESULTS.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_md(res)
    return res


def write_md(res):
    L = ["# +FEATURES (featplus ABC x e11 recipe) vs aug_s0 — results\n",
         res["gate"], "",
         "featplus ABC = base 38 + A(danger: opp-river/meld-commit/progress, +5) + "
         "B(shanten reg/7p/13o + useful-tile, +4) + C(genbutsu safe-tile per opp, +3) = 50 planes. "
         "Trained on the SAME deployable 128x40 CNN with the enhanced e11 recipe "
         "(suit x reflect x dragon aug, label-smoothing, EMA, warmup+cosine).\n",
         "| seed | trained | done | val_acc | blocks | edge/game | 95% CI | beats aug_s0 | verdict | reads_planes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sk, e in res["seeds"].items():
        rp = (e.get("noop_guard") or {}).get("reads_planes")
        if "edge_mean" in e:
            L.append("| %s | %s | %s | %s | %d | %s | [%s, %s] | %s | %s | %s |" % (
                sk, e["trained"], e["train_done"], e.get("val_acc"), e["n_blocks"],
                e["edge_mean"], e["ci95_lo"], e["ci95_hi"], e.get("beats_augs0"), e["verdict"], rp))
        else:
            L.append("| %s | %s | %s | %s | %d | (gating/incomplete) | - | - | - | %s |" % (
                sk, e["trained"], e["train_done"], e.get("val_acc"), e["n_blocks"], rp))
    L += ["", "## Verdict", "(auto-filled at completion — see JSON verdict fields per seed)"]
    open(os.path.join(HERE, "FEATURES_WRITEUP.md"), "w").write("\n".join(L))


def main():
    log("feat_orch START  sets=%s seeds=%s steps=%d gate_blocks=%d workers=%d" %
        (SETS, SEEDS, STEPS, GATE_BLOCKS, WORKERS))
    while True:
        if os.path.exists(STOP):
            log("STOP_ARCHCONT present -> exit"); break
        if disk_gb() < DISK_FLOOR_GB:
            log("disk < %dGB (%.1f) -> WAIT" % (DISK_FLOOR_GB, disk_gb())); time.sleep(120); continue

        # ---- TRAIN phase: launch missing seeds when a safe slot exists ----
        for seed in SEEDS:
            if os.path.exists(os.path.join(HERE, train_bn(seed))) or train_done(seed):
                continue
            if seed_running(seed):
                continue
            nt = n_trainings(); ld = load1(); gpu = free_gpu()
            if nt >= 4:
                log("WAIT train seed=%d: ntrain=%d (>=4)" % (seed, nt)); continue
            if ld >= LOAD_TRAIN:
                log("WAIT train seed=%d: load=%.1f (>=%.0f)" % (seed, ld, LOAD_TRAIN)); continue
            if gpu is None:
                log("WAIT train seed=%d: no free GPU (<800MB)" % seed); continue
            launch_train(seed)
            break   # one launch per pass; re-scan next loop

        # ---- GATE phase: gate trained seeds, one block at a time, thrash-guarded ----
        for seed in SEEDS:
            if not (os.path.exists(os.path.join(HERE, train_bn(seed))) and train_done(seed)):
                continue
            have = len(glob.glob(os.path.join(GD, "featABC_s%d_b*.json" % seed)))
            if have >= GATE_BLOCKS:
                continue
            if gate_running():
                log("WAIT gate seed=%d: another gate running" % seed); break
            ld = load1()
            if ld >= LOAD_GATE:
                log("WAIT gate seed=%d: load=%.1f (>=%.0f)" % (seed, ld, LOAD_GATE)); break
            run_gate_block(seed, have)
            break   # one block per pass; re-check thrash next loop

        aggregate()

        # ---- completion check ----
        all_trained = all(os.path.exists(os.path.join(HERE, train_bn(s))) and train_done(s) for s in SEEDS)
        all_gated = all(len(glob.glob(os.path.join(GD, "featABC_s%d_b*.json" % s))) >= GATE_BLOCKS for s in SEEDS)
        if all_trained and all_gated:
            log("ALL DONE (trained+gated) -> aggregate final + exit")
            aggregate(); break
        time.sleep(90)
    log("feat_orch END")


if __name__ == "__main__":
    main()

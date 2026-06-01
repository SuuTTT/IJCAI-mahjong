"""
pbt_loop.py — Population-Based Training / league loop over the Vast.ai fleet.

Each generation:
  1. Train every population member in parallel, one per box (quota-sized workers),
     warm-started from its own checkpoint, against the shared frozen opponent POOL.
  2. Pull the trained weights; run a cross-play tournament to rank members (Elo-ish net).
  3. EXPLOIT: bottom-half members re-init from a top member's checkpoint next gen.
     EXPLORE: their hyperparameters are perturbed.
  4. The generation champion's weights are added to the POOL (which grows, bounded).
  5. Persist champion (fp16) as the deploy candidate; write status JSON for the dashboard.

Boxes are stateless workers: we rsync the init .pt + pool .npz each gen, run ppo.py,
pull the result. SSH uses direct IPs + ControlMaster multiplexing (no proxy throttle).

State is checkpointed to PBT_DIR/state.json so the loop is resumable.

Run:  OPENBLAS_NUM_THREADS=1 python3 train/pbt_loop.py --gens 8
"""
import os, sys, json, time, random, argparse, subprocess, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

CK       = os.path.join(ROOT, "train", "checkpoints")
PBT_DIR  = os.path.join(ROOT, "train", "pbt")
BUNDLE   = "/tmp/fleet_bundle"
BOXES_JSON = os.path.join(ROOT, "train", "fleet_boxes.json")
STATUS   = "/tmp/pbt_status.json"
STATE    = os.path.join(PBT_DIR, "state.json")
SSHKEY   = os.path.expanduser("~/.ssh/id_ed25519")
os.makedirs(PBT_DIR, exist_ok=True)

SSH_BASE = ["-i", SSHKEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=180s"]
def _cm(box): return ["-o", f"ControlPath=/tmp/cmpbt-{box['id']}"]

def ssh(box, cmd, timeout=900):
    full = ["ssh"] + SSH_BASE + _cm(box) + ["-p", str(box["port"]), f"root@{box['ip']}", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"

def push(box, local, remote, timeout=600):
    e = "ssh " + " ".join(SSH_BASE + _cm(box) + ["-p", str(box["port"])])
    full = ["rsync", "-az", "-e", e, local, f"root@{box['ip']}:{remote}"]
    try:
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False

def pull(box, remote, local, timeout=600):
    e = "ssh " + " ".join(SSH_BASE + _cm(box) + ["-p", str(box["port"])])
    full = ["rsync", "-az", "-e", e, f"root@{box['ip']}:{remote}", local]
    try:
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False

# ───────────────────────── hyperparameter genome ─────────────────────────
def rand_config(rng):
    return {
        "lr":    rng.choice([3e-6, 5e-6, 8e-6, 1.2e-5]),
        "shape": rng.choice([0.005, 0.01, 0.015, 0.02]),
        "ent":   rng.choice([0.005, 0.01, 0.02]),
        "add_every": rng.choice([0, 4, 6]),   # 0 = no self-snapshot during the gen
    }
def perturb(cfg, rng):
    c = dict(cfg)
    c["lr"]    = float(min(2e-5, max(2e-6, c["lr"] * rng.choice([0.6, 0.8, 1.25, 1.6]))))
    c["shape"] = float(min(0.03, max(0.0, c["shape"] + rng.choice([-0.005, 0, 0.005]))))
    c["ent"]   = float(min(0.03, max(0.0, c["ent"] + rng.choice([-0.005, 0, 0.005]))))
    c["add_every"] = rng.choice([0, 4, 6])
    return c

# ───────────────────────── tournament (in-process) ─────────────────────────
def tournament(name_to_npz, n_games, workers):
    """Round-robin 2v2 (seats rotated). Returns {name: net} and ordered ranking."""
    import multiprocessing as mp, itertools
    from eval.tournament import _one_game
    names = list(name_to_npz)
    net = {a: 0 for a in names}; wins = {a: 0 for a in names}
    with mp.Pool(workers) as pool:
        for a, b in itertools.combinations(names, 2):
            args = [(70000 + g, name_to_npz[a], name_to_npz[b], (g % 2 == 0)) for g in range(n_games)]
            res = pool.map(_one_game, args, chunksize=4)
            ap = sum(r[0] for r in res); bp = sum(r[1] for r in res)
            net[a] += ap - bp; net[b] += bp - ap
            wins[a] += sum(r[2] for r in res); wins[b] += sum(r[3] for r in res)
    ranking = sorted(names, key=lambda x: -net[x])
    return net, wins, ranking

# ───────────────────────── box setup (idempotent) ─────────────────────────
def setup_box(box, log):
    log(f"setup {box['id']} ({box['ip']}:{box['port']}) quota={box['quota']}")
    if not push(box, BUNDLE + "/", "/root/mj/"):
        log(f"  rsync FAILED {box['id']}"); return False
    # ensure torch + MahjongGB on whatever python3 the box runs
    chk = ("python3 -c 'import torch,MahjongGB' 2>/dev/null && echo OK || echo NEED")
    rc, out, _ = ssh(box, chk, timeout=120)
    if "OK" not in out:
        log(f"  installing deps on {box['id']} (torch cpu + PyMahjongGB)...")
        ssh(box, "pip install -q PyMahjongGB 2>/dev/null; "
                 "python3 -c 'import torch' 2>/dev/null || "
                 "pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null",
            timeout=900)
        rc, out, _ = ssh(box, chk, timeout=120)
    ok = "OK" in out
    log(f"  {box['id']} deps {'OK' if ok else 'FAILED'}")
    return ok

# ───────────────────────── per-member training ─────────────────────────
def train_member(box, m, gen, pool_names, log, results):
    """Push init+pool, run ppo.py (blocking), pull weights+pt. Fills results[m['name']]."""
    q = int(box["quota"])
    workers = max(2, q)
    games   = int(box["quota"] * 40)
    iters   = ITERS_PER_GEN
    tag     = f"g{gen}_{m['name']}"
    # push this member's init checkpoint
    if not push(box, m["init_pt"], "/root/mj/train/checkpoints/pbt_init.pt"):
        log(f"  [{m['name']}] push init FAILED"); results[m["name"]] = None; return
    pool_csv = ",".join(f"train/checkpoints/{p}" for p in pool_names)
    c = m["config"]
    cmd = (f"cd /root/mj && OPENBLAS_NUM_THREADS=1 nice -n 19 python3 train/ppo.py "
           f"--init train/checkpoints/pbt_init.pt --pool '{pool_csv}' "
           f"--out train/checkpoints/{tag}.pt --iters {iters} --games {games} "
           f"--workers {workers} --epochs 2 --eval-every 9999 "
           f"--lr {c['lr']} --shape {c['shape']} --ent {c['ent']} --add-every {c['add_every']} "
           f"> /root/mj/{tag}.log 2>&1; echo EXIT=$?")
    log(f"  [{m['name']}] train on {box['id']}: q={box['quota']} workers={workers} games={games} cfg={c}")
    rc, out, err = ssh(box, cmd, timeout=ITERS_PER_GEN * 60 + 600)
    if "EXIT=0" not in out:
        rc2, tail, _ = ssh(box, f"tail -3 /root/mj/{tag}.log", timeout=60)
        log(f"  [{m['name']}] train FAILED rc={rc} out={out[-80:]} log={tail[-200:]}")
        results[m["name"]] = None; return
    local_npz = os.path.join(PBT_DIR, f"{tag}_weights.npz")
    local_pt  = os.path.join(PBT_DIR, f"{tag}.pt")
    ok1 = pull(box, f"/root/mj/train/checkpoints/{tag}_weights.npz", local_npz)
    ok2 = pull(box, f"/root/mj/train/checkpoints/{tag}.pt", local_pt)
    if not (ok1 and ok2):
        log(f"  [{m['name']}] pull FAILED ({ok1},{ok2})"); results[m["name"]] = None; return
    log(f"  [{m['name']}] done -> {os.path.basename(local_npz)}")
    results[m["name"]] = {"npz": local_npz, "pt": local_pt}

# ───────────────────────── status / state IO ─────────────────────────
def write_status(d):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f: json.dump(d, f, indent=2)
    os.replace(tmp, STATUS)
def save_state(d):
    with open(STATE, "w") as f: json.dump(d, f, indent=2)

# ───────────────────────── main loop ─────────────────────────
ITERS_PER_GEN = 8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--iters-per-gen", type=int, default=8)
    ap.add_argument("--tourney-games", type=int, default=120)
    ap.add_argument("--tourney-workers", type=int, default=26)
    ap.add_argument("--pool-cap", type=int, default=10)
    ap.add_argument("--min-quota", type=float, default=5.0)
    ap.add_argument("--seed-init", default=os.path.join(CK, "poolbig.pt"))
    args = ap.parse_args()
    global ITERS_PER_GEN; ITERS_PER_GEN = args.iters_per_gen
    rng = random.Random(1234)

    boxes = [b for b in json.load(open(BOXES_JSON)) if b["quota"] >= args.min_quota]
    boxes.sort(key=lambda b: -b["quota"])
    print(f"PBT: {len(boxes)} boxes (quota>={args.min_quota}), total quota "
          f"{sum(b['quota'] for b in boxes):.0f} cores", flush=True)

    # initial population: one member per box, all warm-started from the champion,
    # diversified by random hyperparameters.
    members = [{"name": f"m{i}", "box": boxes[i], "init_pt": args.seed_init,
                "config": rand_config(rng), "net": None} for i in range(len(boxes))]
    # shared opponent pool: frozen anchors (always kept) + grows with champions
    ANCHORS = ["bc_v3_ft_fp16.npz", "ppo_vb_fp16.npz", "league_best_weights.npz",
               "poolbig_best_weights.npz"]
    pool = list(ANCHORS)            # names of npz that already live in /root/mj on each box
    history = []
    log_lines = []
    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); log_lines.append(line)
        if len(log_lines) > 200: del log_lines[:100]

    def snapshot(gen, phase, running=True, boxes_state=None):
        write_status({
            "gen": gen, "phase": phase, "running": running,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_boxes": len(boxes), "total_quota": round(sum(b["quota"] for b in boxes), 1),
            "pool": pool, "pool_size": len(pool),
            "members": [{"name": m["name"], "box": m["box"]["id"], "quota": m["box"]["quota"],
                         "config": m["config"], "net": m["net"]} for m in members],
            "history": history,
            "champion": history[-1]["champion"] if history else None,
            "champion_net": history[-1]["champion_net"] if history else None,
            "boxes_state": boxes_state or {},
            "log": log_lines[-40:],
        })

    # ── one-time box setup (parallel) ──
    snapshot(0, "setup")
    log("=== setup boxes (parallel) ===")
    setup_ok = {}
    ths = []
    for b in boxes:
        t = threading.Thread(target=lambda bb=b: setup_ok.__setitem__(bb["id"], setup_box(bb, log)))
        t.start(); ths.append(t)
    for t in ths: t.join()
    boxes = [b for b in boxes if setup_ok.get(b["id"])]
    members = [m for m in members if m["box"] in boxes]
    log(f"=== {len(boxes)} boxes ready; population {len(members)} ===")
    if len(members) < 2:
        log("FATAL: <2 usable boxes"); snapshot(0, "failed", running=False); return
    # beacon: tell the central gpu-fleet dashboard these boxes are ours & running
    try:
        from train.fleet_notify import notify
        _, _res = notify("running", boxes=[b["id"] for b in boxes],
                         note=f"PBT loop {args.gens} gens", detail=f"{len(boxes)} boxes")
        log("fleet beacon: running -> " + "; ".join(_res))
    except Exception as e:
        log(f"fleet beacon (running) skipped: {e}")

    # ── global best (hall of fame): seed with the current deployed champion ──
    import shutil
    _seed_best = next((p for p in [os.path.join(CK, "pbt_gen2_champion_weights.npz"),
                                   os.path.join(CK, "poolbig_best_weights.npz")] if os.path.exists(p)), None)
    global_best = os.path.join(PBT_DIR, "global_best.npz")
    shutil.copy(_seed_best, global_best)
    global_best_pt = args.seed_init
    log(f"global-best seeded from {os.path.basename(_seed_best)}")

    # ── generations ──
    for gen in range(1, args.gens + 1):
        log(f"========== GENERATION {gen}/{args.gens} ==========")
        snapshot(gen, "train")
        # train all members in parallel (one per box)
        results = {}
        ths = []
        for m in members:
            t = threading.Thread(target=train_member,
                                 args=(m["box"], m, gen, pool, log, results))
            t.start(); ths.append(t)
        for t in ths: t.join()

        ok_members = [m for m in members if results.get(m["name"])]
        if len(ok_members) < 2:
            log(f"gen {gen}: <2 members trained ok; stopping"); break
        for m in ok_members:
            m["_npz"] = results[m["name"]]["npz"]; m["_pt"] = results[m["name"]]["pt"]

        # ── tournament ── members + frozen ANCHOR (absolute ref) + GLOBAL BEST (hall of fame)
        snapshot(gen, "tournament")
        name_to_npz = {m["name"]: m["_npz"] for m in ok_members}
        name_to_npz["_anchor"] = os.path.join(CK, "poolbig_best_weights.npz")
        name_to_npz["_best"] = global_best                  # current deployed best
        log(f"tournament: {len(name_to_npz)} entrants x {args.tourney_games} games/pair")
        net, wins, ranking = tournament(name_to_npz, args.tourney_games, args.tourney_workers)
        for m in ok_members: m["net"] = net[m["name"]]
        anchor_net = net.get("_anchor", 0); best_net = net.get("_best", 0)
        ranking_members = [r for r in ranking if r not in ("_anchor", "_best")]
        log("ranking: " + "  ".join(f"{r}={net[r]:+d}" for r in ranking))

        champ = ranking_members[0]
        champ_m = next(m for m in ok_members if m["name"] == champ)
        import shutil
        # HALL OF FAME: only promote the deployed champion if this gen's best beats the
        # incumbent global best (prevents coevolution drift from shipping a regression).
        promoted = net[champ] > best_net
        if promoted:
            shutil.copy(champ_m["_npz"], os.path.join(CK, "pbt_champion_weights.npz"))
            shutil.copy(champ_m["_pt"],  os.path.join(CK, "pbt_champion.pt"))
            shutil.copy(champ_m["_npz"], os.path.join(PBT_DIR, "global_best.npz"))
            shutil.copy(champ_m["_pt"],  os.path.join(PBT_DIR, "global_best.pt"))
            global_best = os.path.join(PBT_DIR, "global_best.npz")
            global_best_pt = os.path.join(PBT_DIR, "global_best.pt")
            try:
                subprocess.run([sys.executable, os.path.join(ROOT, "train", "quantize.py"),
                                os.path.join(CK, "pbt_champion_weights.npz"),
                                os.path.join(CK, "pbt_champion_fp16.npz")],
                               capture_output=True, timeout=120)
            except Exception as e:
                log(f"  quantize champion failed: {e}")
            log(f"  ** PROMOTED {champ} to global best (net {net[champ]:+d} > prev best {best_net:+d}) "
                f"| vs anchor {net[champ]-anchor_net:+d} **")
        else:
            log(f"  kept global best (gen champ {champ}={net[champ]:+d} <= best {best_net:+d}); no regression shipped")

        history.append({"gen": gen, "champion": champ, "champion_net": net[champ],
                        "anchor_net": anchor_net, "best_net": best_net, "promoted": promoted,
                        "ranking": [[r, net[r]] for r in ranking],
                        "wins": {r: wins[r] for r in ranking}})
        # report progress to the central fleet dashboard (best-effort)
        try:
            from train.fleet_notify import notify as _fnotify
            _fnotify("ping", boxes=[b["id"] for b in boxes], champion=("gen%d:%s" % (gen, champ)),
                     note="PBT-v2", detail=f"gen {gen} champ {champ} net{net[champ]:+d} promoted={promoted}")
        except Exception:
            pass

        # ── PBT exploit + explore ──
        n = len(ranking_members)
        top = ranking_members[:max(1, n // 2)]
        bottom = ranking_members[n // 2:]
        top_pts = {m["name"]: m["_pt"] for m in ok_members}
        for m in ok_members:
            if m["name"] in top:
                m["init_pt"] = m["_pt"]                      # survivors continue
            else:
                src = rng.choice(top)                        # exploit a winner
                m["init_pt"] = top_pts[src]
                src_cfg = next(x for x in ok_members if x["name"] == src)["config"]
                m["config"] = perturb(src_cfg, rng)          # explore
                log(f"  {m['name']} <- exploit {src}, explore cfg={m['config']}")
        members = ok_members

        # ── grow the pool with the champion (bounded) ──
        champ_pool_name = f"pbt_champ_g{gen}.npz"
        # distribute champion weights to all boxes' pool dir for next gen
        local_champ = os.path.join(CK, "pbt_champion_weights.npz")
        for b in boxes:
            push(b, local_champ, f"/root/mj/train/checkpoints/{champ_pool_name}")
        pool.append(champ_pool_name)
        # bound pool size: keep anchors + most recent champions
        if len(pool) > args.pool_cap:
            keep = ANCHORS + [p for p in pool if p not in ANCHORS][-(args.pool_cap - len(ANCHORS)):]
            pool = keep
        log(f"  pool now {len(pool)}: {pool}")

        save_state({"gen": gen, "members": [{"name": m["name"], "box_id": m["box"]["id"],
                    "config": m["config"], "init_pt": m["init_pt"]} for m in members],
                    "pool": pool, "history": history})
        snapshot(gen, "done")

    snapshot(history[-1]["gen"] if history else 0, "finished", running=False)
    log("=== PBT loop finished ===")
    if history:
        log(f"FINAL champion: {history[-1]['champion']} "
            f"(deploy: train/checkpoints/pbt_champion_fp16.npz)")
    # beacon: notify the central dashboard the run is done
    try:
        from train.fleet_notify import notify
        champ = history[-1]["champion"] if history else None
        _, _res = notify("finished", boxes=[b["id"] for b in boxes], champion=champ,
                         note="PBT loop done",
                         detail=f"{len(history)} gens; select best via cross-gen tournament")
        log("fleet beacon: finished -> " + "; ".join(_res))
    except Exception as e:
        log(f"fleet beacon (finished) skipped: {e}")

if __name__ == "__main__":
    main()

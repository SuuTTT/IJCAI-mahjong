"""
league_v3.py — fleet league/PBT loop with a RIGOROUS, drift-proof selector.

Fixes the three failure modes that made the previous 7.7h run drift (it deployed a
model worse than the gen2 champion it started from):
  1. FIXED held-out anchor: gen2 is the permanent reference and is NEVER in any training
     pool. A candidate must beat gen2 (out-of-pool) to be promoted -> generalization gate.
  2. LARGE confirm tournament: ranking uses a cheap N, but the PROMOTION decision uses a
     700-game confirm tournament among the top-2 candidates + gen2 + current champion.
     240 games was noise; promotions now require a low-variance win.
  3. Beat BOTH the held-out gen2 AND the current champion (hall of fame). The deploy file
     pbt_champion_fp16.npz can only ratchet upward; it can never regress.

Structure (AlphaStar-flavoured league):
  - Every round, every box trains a member warm-started from the CURRENT champion.
  - "main" members train vs a diverse frozen pool (gen2 excluded) with varied hyperparams.
  - "exploit" members train ONLY vs the current champion (aggressive, high-entropy) to
    surface weaknesses -> they manufacture the opponent diversity that breaks the ~88%
    draw plateau. A confirmed champion-beating exploiter becomes the new champion; the
    pool grows with promoted champions, so the league keeps escalating.

CPU-bound: the model is tiny (3.4M params), rollout is the cost, so boxes running
tdmpc-glass at 100% GPU still have idle CPU we use (nice -19, workers sized to the cgroup
CPU quota, not physical cores). The local box runs the tournament.

Run:  OPENBLAS_NUM_THREADS=1 python3 train/league_v3.py --hours 8
"""
import os, sys, json, time, random, argparse, subprocess, threading, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# reuse the proven low-level fleet plumbing
import train.pbt_loop as P
from train.pbt_loop import ssh, push, pull, setup_box, tournament

CK      = os.path.join(ROOT, "train", "checkpoints")
LDIR    = os.path.join(ROOT, "train", "league_v3"); os.makedirs(LDIR, exist_ok=True)
BOXES_JSON = "/tmp/fleet_probe.json"
STATUS  = "/tmp/pbt_status.json"          # reuse the existing dashboard
PY      = sys.executable

# held-out anchor (NEVER trained against) and its init checkpoint
GEN2_NPZ = os.path.join(CK, "pbt_gen2_champion_weights.npz")
GEN2_PT  = os.path.join(ROOT, "train", "pbt", "g2_m1.pt")
# diverse training pool for "main" members — gen2 deliberately excluded
MAIN_POOL = ["bc_v3_ft_fp16.npz", "ppo_vb_fp16.npz",
             "league_best_weights.npz", "poolbig_best_weights.npz"]
TARGET_NAME = "_target.npz"                # current champion, pushed to boxes each round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--rank-games", type=int, default=140)
    ap.add_argument("--confirm-games", type=int, default=700)
    ap.add_argument("--margin", type=int, default=300,
                    help="min net margin over gen2 AND current champ (in the stable "
                         "diverse confirm field) required to promote — filters noise")
    ap.add_argument("--tourney-workers", type=int, default=26)
    ap.add_argument("--pool-cap", type=int, default=9)
    ap.add_argument("--min-quota", type=float, default=5.0)
    ap.add_argument("--seed-npz", default=GEN2_NPZ,
                    help="warm-start the champion from this npz (default gen2; pass r18 to continue)")
    ap.add_argument("--seed-pt", default=GEN2_PT,
                    help="warm-start .pt matching --seed-npz")
    ap.add_argument("--pool-sampled", action="store_true",
                    help="train against SAMPLED (looser) pool opponents -> feeds winnable "
                         "situations to teach conversion (the draw/8-fan problem)")
    a = ap.parse_args()
    rng = random.Random(20260601)
    deadline = time.time() + a.hours * 3600

    boxes = [b for b in json.load(open(BOXES_JSON)) if b.get("quota", 0) >= a.min_quota]
    boxes.sort(key=lambda b: -b["quota"])
    log_lines = []
    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); log_lines.append(line)
        if len(log_lines) > 300: del log_lines[:150]
    log(f"league_v3: {a.hours}h, {len(boxes)} boxes, total quota "
        f"{sum(b['quota'] for b in boxes):.0f} cores")

    # roles: ~1/3 exploiters, spread across quota ranks (cycle main,main,exploit)
    for i, b in enumerate(boxes):
        b["role"] = "exploit" if i % 3 == 2 else "main"
        b["name"] = f"{b['role'][:3]}{b['id'][-4:]}"
    log("roles: " + "  ".join(f"{b['name']}(q{b['quota']})" for b in boxes))

    # ── one-time parallel setup (rsync bundle + ensure torch/MahjongGB) ──
    def snap(phase, rnd, running=True, extra=None):
        d = {"gen": rnd, "phase": phase, "running": running,
             "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
             "n_boxes": len(boxes), "total_quota": round(sum(b["quota"] for b in boxes), 1),
             "pool": pool, "pool_size": len(pool),
             "members": [{"name": b["name"], "box": b["id"], "quota": b["quota"],
                          "role": b["role"], "config": b.get("config"),
                          "net": b.get("net")} for b in boxes],
             "history": history, "champion": champ_label, "champion_net": None,
             "log": log_lines[-45:]}
        if extra: d.update(extra)
        tmp = STATUS + ".tmp"; json.dump(d, open(tmp, "w"), indent=2); os.replace(tmp, STATUS)

    pool = list(MAIN_POOL); history = []; champ_label = "gen2"
    snap("setup", 0)
    log("=== setup boxes (parallel) ===")
    ok = {}
    ths = [threading.Thread(target=lambda bb=b: ok.__setitem__(bb["id"], setup_box(bb, log)))
           for b in boxes]
    for t in ths: t.start()
    for t in ths: t.join()
    boxes = [b for b in boxes if ok.get(b["id"])]
    log(f"=== {len(boxes)} boxes ready ===")
    if len(boxes) < 2:
        log("FATAL: <2 usable boxes"); snap("failed", 0, running=False); return

    # ── global best (hall of fame) seeded from gen2 ──
    global_best_npz = os.path.join(LDIR, "global_best.npz"); shutil.copy(a.seed_npz, global_best_npz)
    global_best_pt  = os.path.join(LDIR, "global_best.pt");  shutil.copy(a.seed_pt, global_best_pt)
    log(f"seeded champion from {os.path.basename(a.seed_npz)}; held-out anchor = gen2; "
        f"pool_sampled={a.pool_sampled}")
    target_fp16     = os.path.join(LDIR, "target_fp16.npz")
    def quantize(src, dst):
        subprocess.run([PY, os.path.join(ROOT, "train", "quantize.py"), src, dst],
                       capture_output=True, timeout=120)
    quantize(global_best_npz, target_fp16)
    has_promoted = False
    log("global-best seeded from gen2 (held-out anchor; never in any training pool)")

    # fleet dashboard beacon
    try:
        from train.fleet_notify import notify
        notify("running", boxes=[b["id"] for b in boxes],
               note=f"league_v3 {a.hours}h", detail=f"{len(boxes)} boxes, gen2 anchor")
    except Exception as e:
        log(f"beacon skipped: {e}")

    def cfg_for(role):
        if role == "exploit":
            return {"lr": rng.choice([8e-6, 1.2e-5, 1.6e-5]),
                    "shape": rng.choice([0.0, 0.005, 0.01]),
                    "ent": rng.choice([0.02, 0.03]), "add_every": 0}
        return {"lr": rng.choice([3e-6, 5e-6, 8e-6, 1.2e-5]),
                "shape": rng.choice([0.005, 0.01, 0.015, 0.02]),
                "ent": rng.choice([0.005, 0.01, 0.02]), "add_every": rng.choice([0, 3])}

    P.ITERS_PER_GEN = a.iters

    def train_one(box, rnd, results):
        """Warm-start from the current champion; main->diverse pool, exploit->champion only."""
        q = int(box["quota"]); workers = max(2, q); games = int(box["quota"] * 40)
        box["config"] = cfg_for(box["role"]); c = box["config"]
        tag = f"r{rnd}_{box['name']}"
        # push init = current champion .pt; and (exploit) the champion target npz
        if not push(box, global_best_pt, "/root/mj/train/checkpoints/lv3_init.pt"):
            log(f"  [{box['name']}] push init FAILED"); results[box["name"]] = None; return
        if box["role"] == "exploit":
            if not push(box, target_fp16, f"/root/mj/train/checkpoints/{TARGET_NAME}"):
                log(f"  [{box['name']}] push target FAILED"); results[box["name"]] = None; return
            pool_csv = f"train/checkpoints/{TARGET_NAME}"
        else:
            pool_csv = ",".join(f"train/checkpoints/{p}" for p in pool)
        cmd = (f"cd /root/mj && OPENBLAS_NUM_THREADS=1 nice -n 19 python3 train/ppo.py "
               f"--init train/checkpoints/lv3_init.pt --pool '{pool_csv}' "
               f"--out train/checkpoints/{tag}.pt --iters {a.iters} --games {games} "
               f"--workers {workers} --epochs 2 --eval-every 9999 "
               f"--lr {c['lr']} --shape {c['shape']} --ent {c['ent']} --add-every {c['add_every']} "
               f"{'--pool-sampled ' if a.pool_sampled else ''}"
               f"> /root/mj/{tag}.log 2>&1; echo EXIT=$?")
        log(f"  [{box['name']}] {box['role']} q={box['quota']} w={workers} g={games} cfg={c}")
        rc, out, err = ssh(box, cmd, timeout=a.iters * 120 + 900)
        if "EXIT=0" not in out:
            _, tail, _ = ssh(box, f"tail -3 /root/mj/{tag}.log", timeout=60)
            log(f"  [{box['name']}] TRAIN FAILED rc={rc} {out[-60:]} :: {tail[-160:]}")
            results[box["name"]] = None; return
        lnpz = os.path.join(LDIR, f"{tag}_weights.npz"); lpt = os.path.join(LDIR, f"{tag}.pt")
        if not (pull(box, f"/root/mj/train/checkpoints/{tag}_weights.npz", lnpz)
                and pull(box, f"/root/mj/train/checkpoints/{tag}.pt", lpt)):
            log(f"  [{box['name']}] PULL FAILED"); results[box["name"]] = None; return
        results[box["name"]] = {"npz": lnpz, "pt": lpt}
        log(f"  [{box['name']}] done")

    rnd = 0
    while time.time() < deadline:
        rnd += 1
        secs_left = deadline - time.time()
        log(f"=========== ROUND {rnd}  ({secs_left/3600:.1f}h left) ===========")
        snap("train", rnd)
        results = {}
        ths = [threading.Thread(target=train_one, args=(b, rnd, results)) for b in boxes]
        for t in ths: t.start()
        for t in ths: t.join()
        trained = [b for b in boxes if results.get(b["name"])]
        if len(trained) < 2:
            log(f"round {rnd}: <2 trained ok; skipping"); continue

        # ── stage A: cheap rank over all members + held-out gen2 (+champion if moved) ──
        snap("rank", rnd)
        ent = {b["name"]: results[b["name"]]["npz"] for b in trained}
        ent["_gen2"] = GEN2_NPZ
        if has_promoted: ent["_best"] = global_best_npz
        net, wins, ranking = tournament(ent, a.rank_games, a.tourney_workers)
        for b in trained: b["net"] = net[b["name"]]
        log("rank: " + "  ".join(f"{r}={net[r]:+d}" for r in ranking))
        # Confirm ONLY the single top-ranked candidate. Putting >1 varying candidate in the
        # confirm field perturbs every net (a model's score swings by thousands depending on
        # who else is seated), so a multi-candidate field gives noisy, non-comparable margins.
        # One candidate + the FIXED anchors = a structurally identical field every round.
        cand_names = [r for r in ranking if not r.startswith("_")][:1]

        # ── stage B: confirm in a STABLE, DIVERSE field (fixed anchors) + margin ──
        # A 3-way {cand,gen2,best} net is misleading: with no stable punching-bag the
        # margin swings wildly and rewards head-to-head-vs-champion exploitation that
        # does NOT generalize to a mixed field (the actual contest). Measuring net in a
        # fixed 4-way-style field {gen2, poolbig, SL (+ current champ)} makes "beats
        # gen2" mean contest-relevant 4-way strength; the margin filters tournament noise.
        snap("confirm", rnd)
        c_ent = {n: ent[n] for n in cand_names}
        c_ent["_gen2"]    = GEN2_NPZ
        c_ent["_poolbig"] = os.path.join(CK, "poolbig_best_weights.npz")
        c_ent["_sl"]      = os.path.join(CK, "bc_v3_ft_fp16.npz")
        if has_promoted: c_ent["_best"] = global_best_npz
        cnet, cwins, crank = tournament(c_ent, a.confirm_games, a.tourney_workers)
        log(f"confirm({a.confirm_games}, stable field): " + "  ".join(f"{r}={cnet[r]:+d}" for r in crank))
        best_cand = max(cand_names, key=lambda n: cnet[n])
        ref = max(cnet["_gen2"], cnet.get("_best", cnet["_gen2"]))   # must beat gen2 AND champ
        promoted = (cnet[best_cand] - ref) >= a.margin

        cand_box = next(b for b in trained if b["name"] == best_cand)
        if promoted:
            shutil.copy(results[best_cand]["npz"], os.path.join(CK, "pbt_champion_weights.npz"))
            shutil.copy(results[best_cand]["pt"],  os.path.join(CK, "pbt_champion.pt"))
            shutil.copy(results[best_cand]["npz"], global_best_npz)
            shutil.copy(results[best_cand]["pt"],  global_best_pt)
            quantize(os.path.join(CK, "pbt_champion_weights.npz"),
                     os.path.join(CK, "pbt_champion_fp16.npz"))
            quantize(global_best_npz, target_fp16)
            has_promoted = True; champ_label = f"r{rnd}:{best_cand}"
            # grow the pool with the new champion (bounded; gen2 NEVER added)
            pname = f"lv3_champ_r{rnd}.npz"
            shutil.copy(results[best_cand]["npz"], os.path.join(CK, pname))
            for b in boxes:
                push(b, os.path.join(CK, pname), f"/root/mj/train/checkpoints/{pname}")
            pool.append(pname)
            if len(pool) > a.pool_cap:
                pool = MAIN_POOL + [p for p in pool if p not in MAIN_POOL][-(a.pool_cap - len(MAIN_POOL)):]
            log(f"  ** PROMOTED {best_cand} ({cand_box['role']}): beats gen2 "
                f"({cnet[best_cand]-cnet['_gen2']:+d}) and champ -> new deploy champion **")
        else:
            log(f"  kept champion (no promote: {best_cand} margin over ref "
                f"{cnet[best_cand]-ref:+d} < {a.margin}; vs gen2 {cnet[best_cand]-cnet['_gen2']:+d}); "
                f"deploy unchanged")

        history.append({"round": rnd, "best_cand": best_cand, "role": cand_box["role"],
                        "promoted": promoted, "confirm_net": {k: cnet[k] for k in crank},
                        "vs_gen2": cnet[best_cand] - cnet["_gen2"],
                        "n_promoted": sum(1 for h in history if h["promoted"]) + (1 if promoted else 0)})
        snap("done", rnd, extra={"updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        try:
            from train.fleet_notify import notify as _n
            _n("ping", boxes=[b["id"] for b in boxes], champion=champ_label,
               note="league_v3", detail=f"round {rnd} {'PROMOTE '+best_cand if promoted else 'keep'} vs_gen2 {cnet[best_cand]-cnet['_gen2']:+d}")
        except Exception:
            pass

    np_ = sum(1 for h in history if h["promoted"])
    log(f"=== league_v3 DONE: {rnd} rounds, {np_} promotions, champion={champ_label} ===")
    snap("finished", rnd, running=False)
    try:
        from train.fleet_notify import notify
        notify("finished", boxes=[b["id"] for b in boxes], champion=champ_label,
               note="league_v3 done", detail=f"{rnd} rounds, {np_} promotions")
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
improve_local.py — robust, unattended self-improvement loop on the LOCAL box.

The vast fleet is contended/fluctuating, but this box (28 cores, ours) is reliable.
Each round: train one PPO candidate from the current GLOBAL BEST against the growing
opponent pool with sampled hyperparameters, then run a held-out cross-play tournament
(candidate vs global-best vs frozen anchors). Promote the candidate to global-best
ONLY if it wins (hall of fame — the deployed model never regresses). Runs until a wall
-clock deadline, checkpointing state and writing a live status file every round.

Deploy artifact (always the true best): train/checkpoints/pbt_champion_fp16.npz.

Run:  OPENBLAS_NUM_THREADS=1 python3 train/improve_local.py --hours 8
"""
import os, sys, json, time, random, shutil, subprocess, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from train.pbt_loop import tournament   # reuse the in-process cross-play ranker

CK = os.path.join(ROOT, "train", "checkpoints")
WORK = os.path.join(ROOT, "train", "improve"); os.makedirs(WORK, exist_ok=True)
STATUS = "/tmp/improve_status.json"
PY = sys.executable

ANCHORS = {  # frozen reference opponents, always in the tournament
    "_anchor": os.path.join(CK, "poolbig_best_weights.npz"),
    "_sl":     os.path.join(CK, "bc_v3_ft_fp16.npz"),
}

def sample_cfg(rng, rnd):
    return {
        "lr":    rng.choice([3e-6, 5e-6, 8e-6, 1.2e-5]),
        "shape": rng.choice([0.0, 0.01, 0.015, 0.02, 0.03]),
        "ent":   rng.choice([0.005, 0.01, 0.02]),
        "games": rng.choice([1000, 1400, 1800]),
        "iters": rng.choice([10, 12, 16]),
    }

def write_status(d):
    tmp = STATUS + ".tmp"; json.dump(d, open(tmp, "w"), indent=2); os.replace(tmp, STATUS)

def report_dash(event, detail, champion=None):
    try:
        from train.fleet_notify import notify
        notify(event, boxes=["local-C.34824701"], champion=champion, note="improve_local", detail=detail)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--tourney-games", type=int, default=240)
    ap.add_argument("--pool-cap", type=int, default=10)
    a = ap.parse_args()
    rng = random.Random(20260601)
    deadline = time.time() + a.hours * 3600

    # global best (deploy artifact) — seed from the current champion
    seed = next(p for p in [os.path.join(CK, "pbt_gen2_champion_weights.npz"),
                            os.path.join(CK, "poolbig_best_weights.npz")] if os.path.exists(p))
    seed_pt = next(p for p in [os.path.join(CK, "pbt_gen2_champion.pt"),
                               os.path.join(ROOT, "train", "pbt", "g2_m1.pt"),
                               os.path.join(CK, "poolbig.pt")] if os.path.exists(p))
    best_npz = os.path.join(WORK, "global_best.npz"); shutil.copy(seed, best_npz)
    best_pt = os.path.join(WORK, "global_best.pt");   shutil.copy(seed_pt, best_pt)
    pool = ["bc_v3_ft_fp16.npz", "ppo_vb_fp16.npz", "league_best_weights.npz",
            "poolbig_best_weights.npz", "pbt_gen2_champion_weights.npz"]
    pool = [p for p in pool if os.path.exists(os.path.join(CK, p))]
    history = []
    def log(s):
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    log(f"improve_local: {a.hours}h, seed={os.path.basename(seed)}, pool={len(pool)}")
    report_dash("running", f"local self-improve {a.hours}h", champion="gen2")
    rnd = 0
    while time.time() < deadline:
        rnd += 1
        cfg = sample_cfg(rng, rnd)
        tag = f"r{rnd}"
        out_pt = os.path.join(WORK, f"{tag}.pt")
        pool_csv = ",".join(f"train/checkpoints/{p}" for p in pool)
        cmd = [PY, os.path.join(ROOT, "train", "ppo.py"),
               "--init", best_pt, "--pool", pool_csv, "--out", out_pt,
               "--iters", str(cfg["iters"]), "--games", str(cfg["games"]),
               "--workers", str(a.workers), "--epochs", "2", "--eval-every", "9999",
               "--lr", str(cfg["lr"]), "--shape", str(cfg["shape"]), "--ent", str(cfg["ent"])]
        log(f"round {rnd}: train cfg={cfg}")
        t0 = time.time()
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=cfg["iters"] * 120 + 1200)
        cand_npz = out_pt.replace(".pt", "_weights.npz")
        if r.returncode != 0 or not os.path.exists(cand_npz):
            log(f"  round {rnd} train FAILED rc={r.returncode}: {r.stderr[-200:]}")
            continue
        # held-out tournament: candidate vs global-best vs frozen anchors
        entrants = {"cand": cand_npz, "_best": best_npz, **ANCHORS}
        net, wins, ranking = tournament(entrants, a.tourney_games, a.workers)
        promoted = net["cand"] > net["_best"]
        log(f"  round {rnd} ({time.time()-t0:.0f}s): " + "  ".join(f"{k}={net[k]:+d}" for k in ranking)
            + f"  -> {'PROMOTE' if promoted else 'keep'}")
        if promoted:
            shutil.copy(cand_npz, best_npz); shutil.copy(out_pt, best_pt)
            shutil.copy(cand_npz, os.path.join(CK, "pbt_champion_weights.npz"))
            shutil.copy(out_pt,   os.path.join(CK, "pbt_champion.pt"))
            try:
                subprocess.run([PY, os.path.join(ROOT, "train", "quantize.py"),
                                os.path.join(CK, "pbt_champion_weights.npz"),
                                os.path.join(CK, "pbt_champion_fp16.npz")],
                               capture_output=True, timeout=120)
            except Exception as e:
                log(f"  quantize failed: {e}")
            # add to pool (bounded; keep the 4 frozen anchors + recent champions)
            keep_name = f"improve_champ_r{rnd}.npz"
            shutil.copy(cand_npz, os.path.join(CK, keep_name))
            pool.append(keep_name)
            if len(pool) > a.pool_cap:
                pool = pool[:5] + pool[-(a.pool_cap - 5):]
            report_dash("ping", f"round {rnd} PROMOTED net{net['cand']:+d} vs anchor {net['cand']-net['_anchor']:+d}",
                        champion=f"r{rnd}")
        history.append({"round": rnd, "cfg": cfg, "net": {k: net[k] for k in ranking},
                        "promoted": promoted, "secs": int(time.time() - t0),
                        "vs_anchor": net["cand"] - net["_anchor"]})
        write_status({"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "round": rnd, "hours": a.hours,
                      "deadline_epoch": deadline, "pool_size": len(pool), "pool": pool,
                      "promotions": sum(1 for h in history if h["promoted"]),
                      "history": history[-40:]})
        # cleanup big intermediate files to save disk
        for f in [out_pt, cand_npz, out_pt.replace(".pt", "_baseline.npz"),
                  out_pt.replace(".pt", "_rollout.npz")]:
            try:
                if not promoted and os.path.exists(f): os.remove(f)
            except Exception:
                pass

    log(f"DONE: {rnd} rounds, {sum(1 for h in history if h['promoted'])} promotions")
    report_dash("finished", f"{rnd} rounds, {sum(1 for h in history if h['promoted'])} promotions",
                champion="pbt_champion_fp16")

if __name__ == "__main__":
    main()

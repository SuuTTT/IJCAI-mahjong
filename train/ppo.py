"""
ppo.py — self-play policy improvement for the Mahjong bot (REINFORCE + value
baseline / clipped PPO objective), warm-started from the supervised policy.

Why this shape:
  • Self-play of identical SL policies draws ~95% of games → terminal reward is
    extremely sparse. We densify with SHAPING:
        reward(seat) = game_score/30                      (win≈+1.5, deal-in≈-1)
                     + (HUANG only) closeness*(0.02)       (reward low final shanten)
    so the many drawn games still push the policy toward building better hands.
  • Returns = discounted terminal reward to each of the seat's steps; advantage =
    return - V(obs); clipped PPO policy loss + value loss + small entropy.
  • All 4 seats play the CURRENT policy (self-play). Legal-action masked.

This v1 is single-process (numpy rollout + GPU update) for validation; multiprocess
rollout is added once the loop is confirmed to improve the metric.

Usage:
  OPENBLAS_NUM_THREADS=2 python3 train/ppo.py --init train/checkpoints/bc_v3_ft.pt \
      --iters 50 --games 200 --out train/checkpoints/ppo.pt
"""
import argparse, os, sys, time, numpy as np
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "bot"))
import torch, torch.nn as nn, torch.optim as optim
from train.model import MahjongNet
from train.sim import Sim
from data.feature_agent import ACT_DIM, TILE_LIST
from mahjong_bot import shanten as _shanten

GAMMA = 0.997

# ── numpy policy for rollout (reuses the VALIDATED NumpyMLP forward) ──────────
from train.numpy_infer import NumpyMLP

class NPPolicy:
    """Wrap a NumpyMLP; sample from the masked policy for exploration."""
    def __init__(self, npz_path, greedy=False):
        self.m = NumpyMLP(npz_path)
        self.greedy = greedy
    def __call__(self, obs, mask):
        out = []
        for o, mk in zip(obs, mask):
            probs, _ = self.m.forward(o, mk)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.where(mk, probs, 0.0)          # legal only
            tot = probs.sum()
            if not np.isfinite(tot) or tot <= 0:
                # degenerate -> uniform over legal actions
                legal = np.flatnonzero(mk)
                out.append(int(legal[0]) if len(legal) else 0)
                continue
            probs = probs / tot
            if self.greedy:
                out.append(int(np.argmax(probs)))
            else:
                out.append(int(np.random.choice(len(probs), p=probs)))
        return np.array(out)


def final_shanten(hand, melds):
    try:
        s, _ = _shanten(hand, [(m[0], m[1], 0) for m in melds])
        return max(0, s)
    except Exception:
        return 8


def _rollout_chunk(npz_path, seed0, n_games, opp_npzs=None, shape=0.02, opp_greedy=True):
    """Worker: run n_games and return (obs,mask,act,ret) + stats.
    opp_npzs: list of frozen opponent weight files (the POOL). The learner plays
    seats {0,2}; seats {1,3} are filled by opponents sampled from the pool each
    game (league play). If opp_npzs is None/empty -> pure self-play.
    opp_greedy=False makes pool opponents SAMPLE (looser, varied discards) -> they
    feed winnable situations the cautious greedy meta never produces (teaches conversion)."""
    learner = NPPolicy(npz_path)
    O, M, A, R = [], [], [], []
    wins = draws = 0
    opp_cache = {p: NPPolicy(p, greedy=opp_greedy) for p in (opp_npzs or [])}
    import random as _r
    for g in range(n_games):
        if opp_npzs:
            rg = _r.Random(seed0 + g)
            o1 = opp_cache[rg.choice(opp_npzs)]
            o2 = opp_cache[rg.choice(opp_npzs)]
            seat_pol = [learner, o1, learner, o2]
            lseats = [0, 2]
        else:
            seat_pol = learner
            lseats = [0, 1, 2, 3]
        s = Sim(seat_pol, seed=seed0 + g, learner_seats=lseats)
        traj, sc = s.play()
        # win/draw counted from a learner seat's perspective
        if max(sc[ls] for ls in lseats) > 0: wins += 1
        if max(sc) == 0 and min(sc) == 0: draws += 1
        huang = (max(sc) == 0 and min(sc) == 0)
        for seat in lseats:
            r = sc[seat] / 30.0
            if huang and shape:
                r += (8 - 2 * final_shanten(s.hand[seat], s.melds[seat])) * shape
            steps = traj[seat]; T = len(steps)
            for t, (obs, mask, act) in enumerate(steps):
                O.append(obs); M.append(mask); A.append(act)
                R.append((GAMMA ** (T - 1 - t)) * r)
    if not O:
        return None
    return (np.stack(O), np.stack(M), np.array(A, np.int64),
            np.array(R, np.float32), wins, draws)


def parallel_rollout(npz_path, n_games, seed0, workers, opp_npzs=None, shape=0.02, opp_greedy=True):
    import multiprocessing as mp
    per = max(1, n_games // workers)
    jobs = [(npz_path, seed0 + i * per, per, opp_npzs, shape, opp_greedy) for i in range(workers)]
    with mp.Pool(workers) as pool:
        res = pool.starmap(_rollout_chunk, jobs)
    res = [r for r in res if r]
    O = np.concatenate([r[0] for r in res]); M = np.concatenate([r[1] for r in res])
    A = np.concatenate([r[2] for r in res]); R = np.concatenate([r[3] for r in res])
    wins = sum(r[4] for r in res); draws = sum(r[5] for r in res)
    return O, M, A, R, wins, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="train/checkpoints/bc_v3_ft.pt")
    ap.add_argument("--out",  default="train/checkpoints/ppo.pt")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--vs-baseline", action="store_true",
                    help="train learner (seats 0,2) against the FROZEN SL baseline (1,3)")
    ap.add_argument("--pool", default="",
                    help="comma-sep frozen .npz opponents (LEAGUE). Overrides --vs-baseline.")
    ap.add_argument("--add-every", type=int, default=0,
                    help="snapshot the learner into the pool every N iters (league growth)")
    ap.add_argument("--shape", type=float, default=0.02,
                    help="HUANG shanten-closeness reward weight (0 = pure game score)")
    ap.add_argument("--pool-sampled", action="store_true",
                    help="SAMPLE (non-greedy) from pool opponents -> looser, varied discards "
                         "that create winnable situations the cautious meta never feeds (conversion)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MahjongNet(hidden=args.hidden, n_blocks=args.blocks, dropout=0.0).to(dev)
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); print(f"warm-start {args.init}", flush=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # frozen SL baseline: the progress metric is ALWAYS head-to-head vs this.
    base_npz = args.out.replace(".pt", "_baseline.npz")
    model.export_numpy(base_npz)
    # opponent pool (league): explicit --pool, else the SL baseline if --vs-baseline.
    pool_dir = args.out.replace(".pt", "_pool"); os.makedirs(pool_dir, exist_ok=True)
    if args.pool:
        pool = [p for p in args.pool.split(",") if p]
    elif args.vs_baseline:
        pool = [base_npz]
    else:
        pool = None
    print(f"mode={'league' if pool else 'self-play'} pool_size={len(pool) if pool else 0} "
          f"shape={args.shape} add_every={args.add_every}", flush=True)
    best_net = -1e9

    tmp_npz = args.out.replace(".pt", "_rollout.npz")
    for it in range(1, args.iters + 1):
        t0 = time.time()
        model.eval()
        model.export_numpy(tmp_npz)          # validated forward for rollout
        obs_np, mask, act_np, ret_np, wins, draws = parallel_rollout(
            tmp_npz, args.games, seed0=100000 + it * args.games, workers=args.workers,
            opp_npzs=pool, shape=args.shape, opp_greedy=not args.pool_sampled)
        obs = torch.tensor(obs_np, dtype=torch.float32)
        mask = torch.tensor(mask)
        act = torch.tensor(act_np, dtype=torch.long)
        ret = torch.tensor(ret_np, dtype=torch.float32)
        # old log-probs
        NEG = -1e9
        with torch.no_grad():
            old_lp = torch.zeros(len(act))
            for i in range(0, len(act), args.batch):
                sl = slice(i, i+args.batch)
                o = obs[sl].to(dev); m = mask[sl].to(dev)
                lg, _ = model(o)                       # raw logits (no -inf mask)
                lg = lg.masked_fill(~m, NEG)           # finite mask
                old_lp[sl] = torch.log_softmax(lg, -1)[range(len(act[sl])), act[sl]].cpu()
        # PPO epochs
        for _ in range(args.epochs):
            perm = torch.randperm(len(act))
            for i in range(0, len(act), args.batch):
                idx = perm[i:i+args.batch]
                o = obs[idx].to(dev); m = mask[idx].to(dev)
                a = act[idx].to(dev); R = ret[idx].to(dev); olp = old_lp[idx].to(dev)
                lg, val = model(o)                     # raw logits
                lg = lg.masked_fill(~m, NEG)           # finite mask -> no NaN
                lp_all = torch.log_softmax(lg, -1)
                lp = lp_all[range(len(a)), a]
                adv = (R - val.squeeze(-1)).detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                ratio = torch.exp(lp - olp)
                pol_loss = -torch.min(ratio * adv,
                                      torch.clamp(ratio, 1-args.clip, 1+args.clip) * adv).mean()
                val_loss = ((val.squeeze(-1) - R) ** 2).mean()
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                loss = pol_loss + 0.5 * val_loss - args.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        torch.save({"model": model.state_dict(), "iter": it,
                    "args": vars(args)}, args.out)
        # league growth: periodically freeze the current learner into the pool
        if pool is not None and args.add_every and it % args.add_every == 0:
            snap = os.path.join(pool_dir, f"snap_it{it}.npz")
            model.export_numpy(snap); pool.append(snap)
        msg = (f"iter {it:3d}  win%={100*wins/args.games:.1f} "
               f"draw%={100*draws/args.games:.1f} samples={len(act)} pool={len(pool) if pool else 0} "
               f"loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
        # progress metric: head-to-head vs frozen SL baseline (2v2)
        if it % args.eval_every == 0:
            model.export_numpy(tmp_npz)
            diff = head2head(tmp_npz, base_npz, n=80, workers=args.workers, seed0=900000)
            msg += f"  | vs-baseline net={diff:+.0f}"
            if diff > best_net:                      # keep the PEAK checkpoint
                best_net = diff
                torch.save({"model": model.state_dict(), "iter": it,
                            "val": diff, "args": vars(args)}, args.out.replace(".pt", "_best.pt"))
                model.export_numpy(args.out.replace(".pt", "_best_weights.npz"))
                msg += " *BEST*"
        print(msg, flush=True)

    model.export_numpy(args.out.replace(".pt", "_weights.npz"))
    print("done", flush=True)


def _h2h_chunk(npzA, npzB, seed0, n):
    pa, pb = NPPolicy(npzA, greedy=True), NPPolicy(npzB, greedy=True)
    def mix(layout):
        def f(obs, mask):
            return np.array([ (pa if layout[i] else pb)(obs[i:i+1], mask[i:i+1])[0]
                              for i in range(len(obs)) ])
        return f
    sA = sB = 0
    for g in range(n):
        # seats: A at 0,2 ; B at 1,3   (rotate by parity)
        aseat = [0,2] if g % 2 == 0 else [1,3]
        layout = [i in aseat for i in range(4)]
        s = Sim(mix(layout), seed=seed0 + g); _, sc = s.play()
        for i in range(4):
            if layout[i]: sA += sc[i]
            else: sB += sc[i]
    return sA - sB


def head2head(npzA, npzB, n, workers, seed0):
    import multiprocessing as mp
    per = max(1, n // workers)
    jobs = [(npzA, npzB, seed0 + i*per, per) for i in range(workers)]
    with mp.Pool(workers) as pool:
        diffs = pool.starmap(_h2h_chunk, jobs)
    return sum(diffs)


if __name__ == "__main__":
    main()

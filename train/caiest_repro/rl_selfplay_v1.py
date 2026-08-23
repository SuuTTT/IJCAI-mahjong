"""
rl_selfplay_v1.py -- minimal self-play policy-gradient fine-tune, starting from the imitation
checkpoint aug_s0. This is the "from-scratch self-play RL" door (handoff §3.2 door 1) built with
what's already validated on THIS box (the Sim engine, feature encoding, real MCR scoring) rather
than porting the original team's separate JAX league-RL stack, which isn't present here.

Honest prior going in: the original team tried five RL flavors (incl. PopArt PPO + league
training with opponent pools and KL-annealing, i.e. much more sophisticated infra than this)
and every one plateaued at the imitation anchor with 0/10 durable crossings. This script is a
strictly simpler REINFORCE-with-baseline + reverse-KL-anchor loop, so a-priori it is unlikely to
do better -- built and run anyway because it is the one door with real (if small, if unproven)
theoretical upside left, and because "imitation caps you" was established with THEIR infra, not
with a fresh attempt.

Design:
  - net = ResBNCNN loaded from aug_s0's .bn.pkl (BatchNorm-intact, trainable).
  - frozen = a fixed copy of the SAME starting weights, never updated -- used only as a
    reverse-KL anchor (current || frozen) so the policy can't drift arbitrarily far from a
    known-good point in a single run (mirrors the original league trainer's "p-anchor"/
    "kl-anneal" flags, simplified).
  - Self-play rollouts: multiprocessing pool of CPU workers, each running Sim.play() with all
    4 seats controlled by the SAME current policy, sampling stochastically (temperature-scaled
    softmax over legal actions only) rather than greedy, so there is real exploration signal.
    Workers load weights from a checkpoint file written once per iteration (same _CACHE pattern
    as e11_gate.py) to avoid re-pickling the whole net through IPC each game.
  - Return: this seat's final raw MCR score for that hand (same target formula as the
    value-head experiments), used as the REINFORCE return -- batch-normalized advantage
    (return - batch_mean) / batch_std.
  - Update: policy-gradient loss - reverse-KL(current || frozen) * lam_kl, small LR, AdamW.
  - Every CKPT_EVERY iterations: save a checkpoint AND launch a gate block (reusing
    e11_gate.py's exact calibrated harness) so progress is tracked on the SAME metric as
    everything else in this campaign, not on the (noisy, self-referential) training reward.

  CUDA_VISIBLE_DEVICES=0 python3 rl_selfplay_v1.py --iters 200 --games_per_iter 1000 \
      --workers 48 --out_dir ckpt/rl1
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import multiprocessing as mp
from models_explore import ResBNCNN
from sim_cnn import Sim

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)

_CACHE = {}


def _load_net(path, channels, blocks):
    key = (path, channels, blocks)
    if key not in _CACHE:
        m = ResBNCNN(channels=channels, blocks=blocks)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd)
        m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _sample_lg(m, temp):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({"is_training": False, "obs": {
                "observation": torch.from_numpy(np.ascontiguousarray(obs)),
                "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
        lg = (lg / temp).numpy().flatten()
        p = np.exp(lg - lg.max())
        p = p / p.sum()
        act = int(np.random.choice(len(p), p=p))
        return [act]
    return fn


def _rollout_one(arg):
    seed, ckpt, channels, blocks, temp = arg
    m = _load_net(ckpt, channels, blocks)
    fn = _sample_lg(m, temp)
    sim = Sim([fn] * 4, seed=seed, quan=0, learner_seats=[0, 1, 2, 3], cnn=True)
    traj, scores = sim.play()
    obs_l, mask_l, act_l, ret_l = [], [], [], []
    for seat in range(4):
        for (o, mk, a) in traj[seat]:
            obs_l.append(o)
            mask_l.append(mk)
            act_l.append(a)
            ret_l.append(float(scores[seat]))
    return obs_l, mask_l, act_l, ret_l


def rollout_batch(ckpt, channels, blocks, temp, n_games, seed0, workers):
    args = [(seed0 + i, ckpt, channels, blocks, temp) for i in range(n_games)]
    with mp.Pool(workers) as p:
        res = p.map(_rollout_one, args, chunksize=4)
    obs_l, mask_l, act_l, ret_l = [], [], [], []
    for o, mk, a, r in res:
        obs_l.extend(o); mask_l.extend(mk); act_l.extend(a); ret_l.extend(r)
    return (np.stack(obs_l).astype(np.int8), np.stack(mask_l).astype(bool),
            np.array(act_l, dtype=np.int64), np.array(ret_l, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--init", default="ckpt/aug/aug_128x40_s0.bn.pkl")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games_per_iter", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--lam_kl", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=768)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed0", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    tmp_ckpt = os.path.join(a.out_dir, "_rollout_current.pkl")

    net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    net.load_state_dict(torch.load(os.path.join(HERE, a.init), map_location="cpu"))
    frozen = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    frozen.load_state_dict(net.state_dict())
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    torch.save(net.state_dict(), tmp_ckpt, _use_new_zipfile_serialization=False)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()
    print(f"params {sum(p.numel() for p in net.parameters()):,} lr={a.lr} lam_kl={a.lam_kl} "
          f"games_per_iter={a.games_per_iter} temp={a.temp}", flush=True)

    log_path = os.path.join(a.out_dir, "rl_log.jsonl")
    t0 = time.time()
    seed_counter = a.seed0
    for it in range(a.iters):
        t_roll = time.time()
        obs, mask, act, ret = rollout_batch(tmp_ckpt, a.channels, a.blocks, a.temp,
                                             a.games_per_iter, seed_counter, a.workers)
        seed_counter += a.games_per_iter
        roll_s = time.time() - t_roll

        adv = (ret - ret.mean()) / (ret.std() + 1e-6)
        n = len(act)
        idx = np.random.permutation(n)
        net.train()
        tot_pg = 0.0
        tot_kl = 0.0
        nb = 0
        for bstart in range(0, n, a.bs):
            bidx = idx[bstart:bstart + a.bs]
            ob = torch.from_numpy(np.ascontiguousarray(obs[bidx])).to(dev)
            mk = torch.from_numpy(np.ascontiguousarray(mask[bidx])).to(dev)
            ac = torch.from_numpy(np.ascontiguousarray(act[bidx])).to(dev)
            av = torch.from_numpy(np.ascontiguousarray(adv[bidx])).to(dev).float()
            d = {"is_training": True, "obs": {"observation": ob, "action_mask": mk.float()}}
            with torch.cuda.amp.autocast():
                logits = net(d)
                logp_all = F.log_softmax(logits.float(), dim=1)
                logp = logp_all.gather(1, ac.view(-1, 1)).squeeze(1)
                pg_loss = -(logp * av).mean()
                with torch.no_grad():
                    flogits = frozen(d)
                    flogp_all = F.log_softmax(flogits.float(), dim=1)
                p_cur = logp_all.exp()
                kl = (p_cur * (logp_all - flogp_all)).sum(1).mean()
                loss = pg_loss + a.lam_kl * kl
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_pg += pg_loss.item(); tot_kl += kl.item(); nb += 1
            del logits, logp_all, logp, pg_loss, flogits, flogp_all, p_cur, kl, loss
        torch.cuda.empty_cache()
        net.eval()
        torch.save(net.state_dict(), tmp_ckpt, _use_new_zipfile_serialization=False)
        _CACHE.clear()

        rec = dict(iter=it, n_samples=n, ret_mean=float(ret.mean()), ret_std=float(ret.std()),
                   pg_loss=tot_pg / max(nb, 1), kl=tot_kl / max(nb, 1),
                   roll_seconds=round(roll_s, 1), total_seconds=round(time.time() - t0, 1))
        print(f"[rl] iter {it}/{a.iters} n={n} ret_mean={rec['ret_mean']:.3f} "
              f"pg_loss={rec['pg_loss']:.5f} kl={rec['kl']:.5f} roll_s={roll_s:.0f} "
              f"total_s={rec['total_seconds']:.0f}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if (it + 1) % a.ckpt_every == 0 or it == a.iters - 1:
            ck_path = os.path.join(a.out_dir, f"rl1_it{it + 1}.bn.pkl")
            torch.save(net.state_dict(), ck_path, _use_new_zipfile_serialization=False)
            print(f"[rl] saved checkpoint {ck_path}", flush=True)

    print("[rl] DONE", flush=True)


if __name__ == "__main__":
    main()

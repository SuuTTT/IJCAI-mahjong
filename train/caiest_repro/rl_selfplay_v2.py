"""
rl_selfplay_v2.py -- actor-critic fix on top of rl_selfplay_v1.py.

v1 result (n=18 gate blocks, 30 iterations): placement CI [2.4891, 2.4982] -- WORSE, entirely
below 2.500, a real (if small) negative effect. Diagnosis: v1's "advantage" was
(return - batch_mean)/batch_std -- a single flat baseline shared across every state in the
batch, from every seat, every game, every point in the hand. MCR hand outcomes are dominated by
the wall/opponents/luck, not this seat's own decision quality, so that baseline barely reduces
variance; REINFORCE with a high-variance advantage reliably learns spurious correlations before
it learns anything real, especially over only ~30 iterations. This is the textbook failure mode
policy-gradient methods use a value-function baseline to fix.

Fix: net = ResBNValueCNN (the same value-head architecture already validated this session for
the value-loss experiments), used here as an actual actor-critic baseline: value_head predicts
V(s) = expected return from this state, advantage = return - V(s).detach(), and the value head
is trained jointly via MSE against the observed return (standard on-policy actor-critic, TD(1)/
Monte-Carlo target since we already have the full-episode return for every state).

Starting point: aug_s0's BN checkpoint for stem/body/foot (proven imitation policy); value_head
starts randomly initialized (no such head exists in the imitation checkpoint) and needs a short
warmup before its baseline is trustworthy -- expected, standard in actor-critic cold-starts.

  CUDA_VISIBLE_DEVICES=0 python3 rl_selfplay_v2.py --iters 100 --games_per_iter 1000 \
      --workers 48 --out_dir ckpt/rl2
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import multiprocessing as mp
from models_explore import ResBNCNN, ResBNValueCNN
from sim_cnn import Sim

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)

_CACHE = {}


def _load_net(path, channels, blocks):
    key = (path, channels, blocks)
    if key not in _CACHE:
        m = ResBNValueCNN(channels=channels, blocks=blocks)
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
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--games_per_iter", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--lam_kl", type=float, default=0.05)
    ap.add_argument("--lam_v", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=768)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--resume_full", default=None,
                     help="path to a full ResBNValueCNN checkpoint (e.g. a prior run's "
                          "_rollout_current.pkl) to resume net from; frozen KL anchor still "
                          "loads from --init (the original imitation checkpoint), unchanged.")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    tmp_ckpt = os.path.join(a.out_dir, "_rollout_current.pkl")

    net = ResBNValueCNN(channels=a.channels, blocks=a.blocks).to(dev)
    init_sd = torch.load(os.path.join(HERE, a.init), map_location="cpu")
    if a.resume_full:
        resume_sd = torch.load(os.path.join(HERE, a.resume_full), map_location="cpu")
        net.load_state_dict(resume_sd)
        print(f"resumed net (incl. value_head) from {a.resume_full}", flush=True)
    else:
        net_sd = net.state_dict()
        net_sd.update({k: v for k, v in init_sd.items() if k in net_sd and not k.startswith("value_head")})
        net.load_state_dict(net_sd)

    frozen = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    frozen.load_state_dict(init_sd)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    torch.save(net.state_dict(), tmp_ckpt, _use_new_zipfile_serialization=False)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()
    print(f"params {sum(p.numel() for p in net.parameters()):,} lr={a.lr} lam_kl={a.lam_kl} "
          f"lam_v={a.lam_v} games_per_iter={a.games_per_iter} temp={a.temp}", flush=True)

    log_path = os.path.join(a.out_dir, "rl_log.jsonl")
    t0 = time.time()
    seed_counter = a.seed0
    for it in range(a.iters):
        t_roll = time.time()
        obs, mask, act, ret = rollout_batch(tmp_ckpt, a.channels, a.blocks, a.temp,
                                             a.games_per_iter, seed_counter, a.workers)
        seed_counter += a.games_per_iter
        roll_s = time.time() - t_roll

        ret_scale = float(ret.std() + 1e-6)
        ret_norm = ret / ret_scale
        n = len(act)
        idx = np.random.permutation(n)
        net.train()
        tot_pg = 0.0
        tot_kl = 0.0
        tot_v = 0.0
        tot_adv_std = 0.0
        nb = 0
        for bstart in range(0, n, a.bs):
            bidx = idx[bstart:bstart + a.bs]
            ob = torch.from_numpy(np.ascontiguousarray(obs[bidx])).to(dev)
            mk = torch.from_numpy(np.ascontiguousarray(mask[bidx])).to(dev)
            ac = torch.from_numpy(np.ascontiguousarray(act[bidx])).to(dev)
            rt = torch.from_numpy(np.ascontiguousarray(ret_norm[bidx])).to(dev).float()
            d = {"is_training": True, "obs": {"observation": ob, "action_mask": mk.float()}}
            with torch.cuda.amp.autocast():
                logits, value = net.forward_train(d)
                logp_all = F.log_softmax(logits.float(), dim=1)
                logp = logp_all.gather(1, ac.view(-1, 1)).squeeze(1)
                adv = (rt - value.float().detach())
                pg_loss = -(logp * adv).mean()
                v_loss = F.smooth_l1_loss(value.float(), rt)
                with torch.no_grad():
                    flogits = frozen(d)
                    flogp_all = F.log_softmax(flogits.float(), dim=1)
                p_cur = logp_all.exp()
                kl = (p_cur * (logp_all - flogp_all)).sum(1).mean()
                loss = pg_loss + a.lam_v * v_loss + a.lam_kl * kl
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_pg += pg_loss.item(); tot_kl += kl.item(); tot_v += v_loss.item()
            tot_adv_std += float(adv.std().item()); nb += 1
            del logits, value, logp_all, logp, adv, pg_loss, v_loss, flogits, flogp_all, p_cur, kl, loss
        torch.cuda.empty_cache()
        net.eval()
        torch.save(net.state_dict(), tmp_ckpt, _use_new_zipfile_serialization=False)
        _CACHE.clear()

        rec = dict(iter=it, n_samples=n, ret_mean=float(ret.mean()), ret_std=float(ret.std()),
                   pg_loss=tot_pg / max(nb, 1), v_loss=tot_v / max(nb, 1),
                   adv_std=tot_adv_std / max(nb, 1), kl=tot_kl / max(nb, 1),
                   roll_seconds=round(roll_s, 1), total_seconds=round(time.time() - t0, 1))
        print(f"[rl2] iter {it}/{a.iters} n={n} ret_mean={rec['ret_mean']:.3f} "
              f"pg_loss={rec['pg_loss']:.5f} v_loss={rec['v_loss']:.4f} adv_std={rec['adv_std']:.3f} "
              f"kl={rec['kl']:.5f} roll_s={roll_s:.0f} total_s={rec['total_seconds']:.0f}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if (it + 1) % a.ckpt_every == 0 or it == a.iters - 1:
            sd = net.state_dict()
            sd_policy = {k: v for k, v in sd.items() if not k.startswith("value_head")}
            bestnet = ResBNCNN(channels=a.channels, blocks=a.blocks)
            bestnet.load_state_dict(sd_policy)
            ck_path = os.path.join(a.out_dir, f"rl2_it{it + 1}.bn.pkl")
            torch.save(bestnet.state_dict(), ck_path, _use_new_zipfile_serialization=False)
            print(f"[rl2] saved checkpoint {ck_path} (policy-only, value_head stripped)", flush=True)

    print("[rl2] DONE", flush=True)


if __name__ == "__main__":
    main()

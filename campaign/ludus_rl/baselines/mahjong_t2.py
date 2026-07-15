"""T2 -- PPO fine-tune of the MCR Mahjong SL policy, KL-leashed to the frozen SL.

Learner = seat 0 of ``mahjong.rl_env`` (reused encoder/legality via FeatureDriver + MyEngine).
Opponents = frozen self-play (frozen SL + rolling frozen learner snapshots, uniform sample).
Masked categorical PPO over 235 legal actions; terminal-only reward = seat-0 MCR score / 8.
KL(current || frozen_SL) held under --kl-target with an adaptive beta (starts --beta0).
Critic init from value_e2e_ckpt.pt (score/30) rescaled to reward units (x30/8), refit by GAE.

All net inference is BATCHED on the GPU across many parallel games; the env step (obs build,
legality, engine) is CPU/Python and dominates wall time, so we run many games concurrently and
issue one big GPU forward per decision-tick.

Modes:
  (default)   train / resume the PPO loop, checkpoint + JSONL eval.
  --parity N  load torch policy + numpy champion, compare masked-argmax on saved states, exit.

Usage:
  python -m baselines.mahjong_t2 --seed 1 --updates 100000 --kl-target 0.05 --beta0 0.08 \
      --out /root/ludus_train/mahjong_t2/ --resume
"""
import argparse, json, os, sys, time, copy, random
import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, "/root/ludus")
sys.path.insert(0, "/root/mcr_champion")
os.environ.setdefault("MCR_CHAMPION_DIR", "/root/mcr_champion")

from mahjong.engine import build_wall
from mahjong.validate_adapter import MyEngine
from mahjong.rl_env import FeatureDriver
from baselines.mahjong_torch_model import load_policy, load_value, PolicyNet

POLICY_INIT = "/root/mcr_t2/kd_128x40_s0.pkl"
VALUE_INIT = "/root/mcr_t2/value_e2e_ckpt.pt"
V_RESCALE = 30.0 / 8.0            # critic predicts score/30, reward is score/8
NEG = -1e9


# ============================================================== game coroutine
def play_game(wall, quan, seed):
    """Generator driving ALL 4 seats via FeatureDriver + MyEngine.
    Yields (seat, obs(38,4,9)f32, mask(235)bool) at every decision; receives the
    chosen action int via .send; returns the terminal ``display`` dict."""
    eng = MyEngine()
    drivers = [FeatureDriver() for _ in range(4)]
    turn = eng.reset(wall, quan, seed)
    while True:
        disp = turn["display"]
        reqs = turn["requests"]
        if not reqs:                                     # terminal turn
            break
        responses = {}
        for seat_str in sorted(reqs, key=lambda s: int(s)):
            seat = int(seat_str)
            gen = drivers[seat].decide(reqs[seat_str])
            try:
                obs, mask = next(gen)
                while True:
                    a = yield (seat, np.asarray(obs, np.float32), np.asarray(mask, bool))
                    obs, mask = gen.send(a)
            except StopIteration as e:
                responses[seat_str] = e.value
        turn = eng.step(responses)
    return disp


def parse_terminal(disp):
    """-> dict(ending, score0, winner, learner_zimo, learner_dealin)."""
    if disp.get("action") != "HU":
        return dict(ending="draw", score0=0, winner=None, zimo=False, dealin=False)
    score = disp["score"]
    winner = disp["player"]
    losers = [s for s in range(4) if s != winner]
    ls = [score[s] for s in losers]
    zimo = (min(ls) == max(ls))                          # all three losers equal -> self-draw
    discarder = None if zimo else losers[int(np.argmin(ls))]
    return dict(ending="hu", score0=score[0], winner=winner,
                zimo=(zimo and winner == 0),
                dealin=((not zimo) and discarder == 0))


# ============================================================== batched forward
def masked_dist(net, obs_np, mask_np, device):
    """obs_np (B,38,4,9), mask_np (B,235) -> (Categorical over masked logits, masked_logits, mask_t)."""
    O = torch.from_numpy(obs_np).to(device)
    M = torch.from_numpy(mask_np).to(device)
    logits = net(O)
    masked = logits.masked_fill(~M, NEG)
    return Categorical(logits=masked), masked, M


# ============================================================== the trainer
class Trainer:
    def __init__(self, a):
        self.a = a
        self.dev = "cuda"
        os.makedirs(a.out, exist_ok=True)
        # nets
        self.policy = load_policy(POLICY_INIT, self.dev)          # learner (trained)
        self.frozen = load_policy(POLICY_INIT, self.dev)          # frozen SL (KL ref + opp seed)
        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.value = load_value(VALUE_INIT, self.dev)
        self.value.eval()                                        # keep BN on frozen running stats
        # opponent pool: index 0 == frozen SL, rest == frozen learner snapshots
        self.pool = [self.frozen]
        # optimizer (policy + critic, separate lrs)
        self.opt = torch.optim.Adam([
            {"params": self.policy.parameters(), "lr": a.lr_pi},
            {"params": self.value.parameters(), "lr": a.lr_v},
        ])
        self.beta = a.beta0
        self.setpoint = 0.5 * a.kl_target                        # aim ~0.5*ceiling (pilot ~0.025)
        self.update = 0
        self.env_steps = 0
        self.games_done = 0
        self.t_start = time.time()
        self.last_ckpt = time.time()
        # persistent parallel games
        self.B = a.games
        self.gseed_ctr = 1_000 + 100_003 * a.seed                # training wall seeds
        self.gens = [None] * self.B
        self.pend = [None] * self.B
        self.traj = [[] for _ in range(self.B)]
        self.oppmap = [None] * self.B
        for i in range(self.B):
            self._start_game(i)

    # ---- game lifecycle
    def _next_seed(self):
        s = self.gseed_ctr
        self.gseed_ctr += 1
        return s

    def _sample_opps(self):
        n = len(self.pool)
        return {seat: random.randrange(n) for seat in (1, 2, 3)}

    def _start_game(self, i):
        s = self._next_seed()
        self.oppmap[i] = self._sample_opps()
        self.traj[i] = []
        g = play_game(build_wall(s), self.a.quan, s)
        try:
            self.pend[i] = g.send(None)
            self.gens[i] = g
        except StopIteration:
            self.pend[i] = None
            self.gens[i] = None

    # ---- one scheduler tick: batched forwards for all live games
    def _tick(self, buffer, greedy_learner=False, opp_greedy=False, kl_accum=None):
        learner_idx = []
        groups = {}                                          # pool_idx -> [game_idx]
        for i in range(self.B):
            if self.pend[i] is None:
                continue
            seat, _, _ = self.pend[i]
            if seat == 0:
                learner_idx.append(i)
            else:
                groups.setdefault(self.oppmap[i][seat], []).append(i)
        actions = {}
        # --- learner (current policy) ---
        if learner_idx:
            obs = np.stack([self.pend[i][1] for i in learner_idx])
            msk = np.stack([self.pend[i][2] for i in learner_idx])
            with torch.no_grad():
                dist, masked, M = masked_dist(self.policy, obs, msk, self.dev)
                if greedy_learner:
                    act = masked.argmax(1)
                else:
                    act = dist.sample()
                logp = dist.log_prob(act)
                val = self.value(torch.from_numpy(obs).to(self.dev)) * V_RESCALE
                if kl_accum is not None:                     # eval-time KL(cur||frozen)
                    flog = self.frozen(torch.from_numpy(obs).to(self.dev)).masked_fill(~M, NEG)
                    p = torch.softmax(masked, 1)
                    kl = (p * (torch.log_softmax(masked, 1) - torch.log_softmax(flog, 1))).sum(1)
                    kl_accum[0] += float(kl.sum()); kl_accum[1] += kl.numel()
            act = act.cpu().numpy(); logp = logp.cpu().numpy(); val = val.cpu().numpy()
            for k, i in enumerate(learner_idx):
                actions[i] = int(act[k])
                if buffer is not None:
                    self.traj[i].append(dict(obs=self.pend[i][1], mask=self.pend[i][2],
                                             action=int(act[k]), logp=float(logp[k]),
                                             value=float(val[k])))
        # --- opponents (frozen pool nets) ---
        for pidx, idxs in groups.items():
            obs = np.stack([self.pend[i][1] for i in idxs])
            msk = np.stack([self.pend[i][2] for i in idxs])
            with torch.no_grad():
                dist, masked, _ = masked_dist(self.pool[pidx], obs, msk, self.dev)
                act = masked.argmax(1) if opp_greedy else dist.sample()
            act = act.cpu().numpy()
            for k, i in enumerate(idxs):
                actions[i] = int(act[k])
        # --- advance every live game by one decision ---
        finished = []
        for i in range(self.B):
            if self.pend[i] is None:
                continue
            try:
                self.pend[i] = self.gens[i].send(actions[i])
            except StopIteration as e:
                finished.append((i, e.value))
        return finished

    # ---- GAE finalize of one game's learner trajectory
    def _finalize(self, i, disp, buffer):
        term = parse_terminal(disp)
        R = term["score0"] / 8.0
        tr = self.traj[i]
        if tr:
            T = len(tr)
            vals = [t["value"] for t in tr]
            adv = [0.0] * T
            lastgae = 0.0
            g, lam = self.a.gamma, self.a.lam
            for t in reversed(range(T)):
                nonterm = 0.0 if t == T - 1 else 1.0
                nextv = 0.0 if t == T - 1 else vals[t + 1]
                r = R if t == T - 1 else 0.0
                delta = r + g * nextv * nonterm - vals[t]
                lastgae = delta + g * lam * nonterm * lastgae
                adv[t] = lastgae
            for t in range(T):
                s = tr[t]
                buffer.append((s["obs"], s["mask"], s["action"], s["logp"],
                               adv[t], adv[t] + vals[t]))
        self.games_done += 1

    # ---- collect one rollout of >= rollout_steps learner transitions
    def collect(self):
        buffer = []
        while len(buffer) < self.a.rollout:
            finished = self._tick(buffer)
            for i, disp in finished:
                self._finalize(i, disp, buffer)
                self._start_game(i)
        self.env_steps += len(buffer)
        return buffer

    # ---- PPO update
    def ppo_update(self, buffer):
        dev = self.dev
        obs = torch.from_numpy(np.stack([b[0] for b in buffer])).to(dev)
        mask = torch.from_numpy(np.stack([b[1] for b in buffer])).to(dev)
        act = torch.tensor([b[2] for b in buffer], dtype=torch.long, device=dev)
        oldlp = torch.tensor([b[3] for b in buffer], dtype=torch.float32, device=dev)
        adv = torch.tensor([b[4] for b in buffer], dtype=torch.float32, device=dev)
        ret = torch.tensor([b[5] for b in buffer], dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        a = self.a
        pg_acc = 0.0; v_acc = 0.0; ent_acc = 0.0; nb = 0

        def buffer_kl():
            """True mean KL(current||frozen_SL) over the whole rollout (current-policy on-dist)."""
            ks = 0.0; kc = 0
            with torch.no_grad():
                for s in range(0, N, a.minibatch):
                    O, Mk = obs[s:s + a.minibatch], mask[s:s + a.minibatch]
                    m = self.policy(O).masked_fill(~Mk, NEG)
                    fl = self.frozen(O).masked_fill(~Mk, NEG)
                    p = torch.softmax(m, 1)
                    kl = (p * (torch.log_softmax(m, 1) - torch.log_softmax(fl, 1))).sum(1)
                    ks += float(kl.sum()); kc += kl.numel()
            return ks / max(1, kc)

        # TRUST REGION: snapshot the policy, run PPO, then measure the TRUE post-update drift on the
        # rollout (which matches the eval-time distribution); if it breached the ceiling, ROLL BACK
        # the policy and ramp beta. This guarantees KL(current||frozen_SL) <= kl_target after every
        # accepted update -- the in-update per-minibatch KL under-reports post-update drift, so we
        # cannot rely on it alone. A per-minibatch soft gate still cuts the worst overshoot cheaply.
        pre_state = {k: v.detach().clone() for k, v in self.policy.state_dict().items()}
        gate = a.kl_gate_frac * a.kl_target
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, a.minibatch):
                idx = perm[s:s + a.minibatch]
                O, Mk = obs[idx], mask[idx]
                logits = self.policy(O)
                masked = logits.masked_fill(~Mk, NEG)
                dist = Categorical(logits=masked)
                newlp = dist.log_prob(act[idx])
                ent = dist.entropy().mean()
                ratio = torch.exp(newlp - oldlp[idx])
                A = adv[idx]
                pg = -torch.min(ratio * A,
                                torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A).mean()
                vpred = self.value(O) * V_RESCALE
                vloss = 0.5 * (vpred - ret[idx]).pow(2).mean()
                with torch.no_grad():
                    flog = self.frozen(O).masked_fill(~Mk, NEG)
                p = torch.softmax(masked, 1)
                kl = (p * (torch.log_softmax(masked, 1) - torch.log_softmax(flog, 1))).sum(1).mean()
                # Soft KL wall at `gate`: below it, do the full PPO improvement step (with the beta
                # leash); at/above it, DROP the advantage term but KEEP the beta*KL term so the step
                # actively pulls KL back down (never freeze the corrective gradient off). The
                # trust-region rollback at `kl_target` is the hard ceiling behind this soft wall.
                loss = a.vcoef * vloss + self.beta * kl
                if float(kl) < gate:
                    loss = loss + pg - a.entcoef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()), a.max_grad_norm)
                self.opt.step()
                pg_acc += float(pg); v_acc += float(vloss); ent_acc += float(ent); nb += 1
        post_kl = buffer_kl()
        rejected = False
        if post_kl > a.kl_target:                         # breach -> reject policy move, keep critic
            self.policy.load_state_dict(pre_state)
            self.beta = min(self.beta * 2.0, 50.0)
            rejected = True
            post_kl = buffer_kl()                          # drift after rollback (last accepted level)
        else:                                             # adapt beta around setpoint (~0.5*ceiling)
            if post_kl > 0.5 * a.kl_target:
                self.beta = min(self.beta * 1.3, 20.0)
            elif post_kl < 0.3 * a.kl_target:
                self.beta = max(self.beta / 1.2, 1e-3)
        return dict(kl=post_kl, pg=pg_acc / max(1, nb), vloss=v_acc / max(1, nb),
                    ent=ent_acc / max(1, nb), beta=self.beta, rejected=rejected)

    # ---- snapshot pool refresh
    def maybe_snapshot(self):
        a = self.a
        if a.snap_every <= 0 or self.update % a.snap_every != 0:
            return
        snap = PolicyNet().to(self.dev)
        snap.load_state_dict(self.policy.state_dict())
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.pool.append(snap)
        while len(self.pool) > a.pool_max:                  # drop oldest snapshot (keep index0=SL)
            self.pool.pop(1)

    # ---- evaluation (greedy learner vs frozen SL argmax) + JSONL
    def evaluate(self):
        a = self.a
        n = a.eval_games
        # temporary eval scheduler that does NOT disturb training games
        save = (self.gens, self.pend, self.traj, self.oppmap)
        self.gens = [None] * self.B; self.pend = [None] * self.B
        self.traj = [[] for _ in range(self.B)]; self.oppmap = [None] * self.B
        eval_seed = 10_000_000
        ctr = [eval_seed]
        def start(i):
            s = ctr[0]; ctr[0] += 1
            self.oppmap[i] = {1: 0, 2: 0, 3: 0}            # all frozen SL
            g = play_game(build_wall(s), a.quan, s)
            try:
                self.pend[i] = g.send(None); self.gens[i] = g
            except StopIteration:
                self.pend[i] = None; self.gens[i] = None
        for i in range(self.B):
            start(i)
        results = []
        kl_accum = [0.0, 0]
        while len(results) < n:
            fin = self._tick(None, greedy_learner=True, opp_greedy=True, kl_accum=kl_accum)
            for i, disp in fin:
                results.append(parse_terminal(disp))
                if len(results) < n:
                    start(i)
                else:
                    self.pend[i] = None
            # drain remaining in-flight once we have enough starts
        results = results[:n]
        self.gens, self.pend, self.traj, self.oppmap = save     # restore training games
        mean_score = float(np.mean([r["score0"] for r in results]))
        zimo = float(np.mean([1.0 if r["zimo"] else 0.0 for r in results]))
        dealin = float(np.mean([1.0 if r["dealin"] else 0.0 for r in results]))
        hu = float(np.mean([1.0 if r["ending"] == "hu" and r["winner"] == 0 else 0.0 for r in results]))
        kl = kl_accum[0] / max(1, kl_accum[1])
        rec = {"step": self.update, "step_kind": "ppo_updates", "env_steps": self.env_steps,
               "KL": round(kl, 5), "mean_score": round(mean_score, 4),
               "zimo_rate": round(zimo, 4), "dealin_rate": round(dealin, 4),
               "win_rate": round(hu, 4), "n_games": len(results), "beta": round(self.beta, 5),
               "wall_s": round(time.time() - self.t_start, 1)}
        with open(os.path.join(a.out, "seed%d_results.jsonl" % a.seed), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # ---- checkpoint / resume
    def save_ckpt(self, tag="latest"):
        a = self.a
        ck = dict(policy=self.policy.state_dict(), value=self.value.state_dict(),
                  opt=self.opt.state_dict(), beta=self.beta, update=self.update,
                  env_steps=self.env_steps, games_done=self.games_done,
                  gseed_ctr=self.gseed_ctr, seed=a.seed)
        torch.save(ck, os.path.join(a.out, "%s.pt" % tag))
        torch.save(ck, os.path.join(a.out, "ckpt_u%d.pt" % self.update))
        # prune old milestone ckpts (keep latest.pt + newest `keep` ckpt_u*.pt) to bound disk
        keep = 6
        import glob as _glob
        cks = sorted(_glob.glob(os.path.join(a.out, "ckpt_u*.pt")),
                     key=lambda p: int(p.split("ckpt_u")[-1].split(".pt")[0]))
        for old in cks[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass

    def try_resume(self):
        p = os.path.join(self.a.out, "latest.pt")
        if not os.path.exists(p):
            return False
        ck = torch.load(p, map_location=self.dev)
        self.policy.load_state_dict(ck["policy"])
        self.value.load_state_dict(ck["value"])
        self.opt.load_state_dict(ck["opt"])
        self.beta = ck["beta"]; self.update = ck["update"]
        self.env_steps = ck["env_steps"]; self.games_done = ck["games_done"]
        self.gseed_ctr = ck.get("gseed_ctr", self.gseed_ctr)
        print(f"[resume] update={self.update} env_steps={self.env_steps} beta={self.beta:.4f}", flush=True)
        return True

    # ---- main loop
    def run(self):
        a = self.a
        if a.eval_at_start and self.update == 0:
            rec = self.evaluate()
            print("[eval@start]", json.dumps(rec), flush=True)
        while self.update < a.updates:
            t0 = time.time()
            buf = self.collect()
            stats = self.ppo_update(buf)
            self.update += 1
            self.maybe_snapshot()
            dt = time.time() - t0
            gps = len(buf) / dt
            if self.update % a.log_every == 0:
                print(f"[u{self.update}] kl={stats['kl']:.4f}{'(REJ)' if stats['rejected'] else ''} "
                      f"beta={stats['beta']:.4f} pg={stats['pg']:.4f} vloss={stats['vloss']:.3f} "
                      f"ent={stats['ent']:.3f} steps/s={gps:.0f} env_steps={self.env_steps} "
                      f"games={self.games_done} pool={len(self.pool)}", flush=True)
            do_eval = (a.eval_every_updates > 0 and self.update % a.eval_every_updates == 0)
            do_ckpt = (time.time() - self.last_ckpt >= a.ckpt_every_sec)
            if do_ckpt:
                do_eval = True
            if do_eval:
                rec = self.evaluate()
                print("[eval]", json.dumps(rec), flush=True)
            if do_ckpt:
                self.save_ckpt()
                self.last_ckpt = time.time()
                print(f"[ckpt] saved at update {self.update}", flush=True)
        # final ckpt+eval
        self.save_ckpt()
        print("[eval-final]", json.dumps(self.evaluate()), flush=True)


# ============================================================== parity mode
def run_parity(n):
    from numpy_resfused import NumpyResFused
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pi = load_policy(POLICY_INIT, dev); pi.eval()
    npm = NumpyResFused("/root/mcr_champion/kdens_s0_fp16.npz")
    obs = np.load("/root/parity_obs.npy")[:n]; mask = np.load("/root/parity_mask.npy")[:n]
    with torch.no_grad():
        logits = pi(torch.from_numpy(obs).float().to(dev)).cpu().numpy()
    dis = 0
    for i in range(obs.shape[0]):
        ta = int(np.where(mask[i], logits[i], NEG).argmax())
        na = int(npm.logits(obs[i], mask[i]).argmax())
        dis += (ta != na)
    print(f"PARITY torch-policy vs numpy-champion: {dis}/{obs.shape[0]} argmax disagreements", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--out", type=str, default="/root/ludus_train/mahjong_t2/")
    ap.add_argument("--games", type=int, default=256)          # parallel games (batch size)
    ap.add_argument("--rollout", type=int, default=4096)       # learner transitions per update
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr-pi", type=float, default=3e-5)
    ap.add_argument("--lr-v", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--entcoef", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--kl-target", type=float, default=0.05)   # KL ceiling; setpoint = 0.5*this
    ap.add_argument("--beta0", type=float, default=0.08)
    ap.add_argument("--kl-gate-frac", type=float, default=0.7)  # policy step gated off above this*ceiling
    ap.add_argument("--quan", type=int, default=0)
    ap.add_argument("--snap-every", type=int, default=25)      # add a frozen learner snapshot
    ap.add_argument("--pool-max", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=2000)
    ap.add_argument("--eval-every-updates", type=int, default=0)  # 0 -> only on ckpt
    ap.add_argument("--eval-at-start", action="store_true")
    ap.add_argument("--ckpt-every-sec", type=float, default=7200)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--parity", type=int, default=0)
    args = ap.parse_args()

    if args.parity:
        run_parity(args.parity)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tr = Trainer(args)
    if args.resume:
        tr.try_resume()
    tr.run()


if __name__ == "__main__":
    main()

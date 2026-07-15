"""
e10_gate.py -- E10: RISK-SEEKING value-guided 1-ply action selection vs base (cnn_lad),
in the calibrated DUPLICATE-format placement gate. INVERSE of E8.

E8 minimized V_place (expected placement) and was NULL/monotonically worse. E10 instead uses
the V_score head (expected raw MCR score -- the UPSIDE signal) and MAXIMIZES it: it pushes the
policy toward high-scoring/winning lines, trading some safety for FIRSTS.

THE OVERLAY (risk-seeking, TRUE 1-ply):
At a candidate-seat decision, take the policy's top-K LEGAL actions. For each candidate a:
  - clone the caiest agent, apply a (Play/Chi/Peng) -> post-action obs (38,4,9)
  - value model V forward -> V_score_after(a) (score head, raw expected MCR), p4_after(a) (P last)
Final choice = argmax over K of:  policy_logit(a) + lambda*V_score_after(a) - mu*p4_after(a)
  - lambda>=0 = risk-seeking weight on upside (lambda=0 recovers base argmax -> calib 2.500)
  - mu>=0 = optional 4th-guard (penalize lines that raise P(4th); default 0 = pure risk-seek)
Hooks ONLY the play axis: DISCARD (Play [2,36)) and CLAIM (Chi [36,99), Peng [99,133)).
Non-hooked top-K candidates use the current-state value (no-op baseline) so none are dropped.
Never overrides into an illegal action.

  python3 e10_gate.py --cand /root/assets/cnn_lad_chunjiandu.npz \
     --ref /root/assets/cnn_lad_chunjiandu.npz \
     --value ckpt/value_256x40.pkl --lam 1.0 --mu 0 --topk 5 --seeds 400 --workers 80 \
     --seed0 70000 --out e10_cells/lam1.0_mu0_s70000.json
"""
import os, sys, json, argparse, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build
from feature import FeatureAgent as CaiAgent

torch.set_num_threads(1)
_CACHE = {}
PASS, HU, PLAY = 0, 1, 2
CHI0, PENG0, GANG0 = 36, 99, 133
ANGANG0 = 167
TILE_LIST = CaiAgent.TILE_LIST

IN_PLANES = 38
class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(x + y)

class ValueMT(nn.Module):
    def __init__(self, channels=128, blocks=20):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.place_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 4))
        self.fourth_head = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
        self.score_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, obs):
        x = self.body(self.stem(obs)); x = x.mean(dim=(2, 3))
        return self.place_head(x), self.fourth_head(x).squeeze(1), self.score_head(x).squeeze(1)


def _load_value(path):
    key = ("VALUE", path)
    if key not in _CACHE:
        ck = torch.load(path, map_location="cpu")
        net = ValueMT(ck["channels"], ck["blocks"]); net.load_state_dict(ck["state"]); net.eval()
        _CACHE[key] = net
    return _CACHE[key]


def _value_score_p4(net, obs_batch):
    """obs_batch: (B,38,4,9) float32 -> V_score (B,) raw expected MCR, p4 (B,) P(4th)."""
    with torch.no_grad():
        pl, fo, sc = net(torch.from_numpy(np.ascontiguousarray(obs_batch.astype(np.float32))))
        p4 = torch.sigmoid(fo).numpy()
        score = sc.numpy()
    return score, p4


def _parse_cfg(s):
    cfg = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("="); cfg[k] = int(v)
    return cfg


def _load_policy(path, kind, cfg):
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        if path.endswith(".npz"):
            z = np.load(path)
            sd = {k: torch.from_numpy(z[k]) for k in z.keys() if k != "meta_blocks"}
        else:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and not any(
                    k.startswith(("stem", "body", "foot", "head")) for k in sd):
                sd = sd["state_dict"]
        m.load_state_dict(sd); m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _logits(m, obs, mask):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()


def _greedy_lg(m):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask)
        if return_logits:
            return lg
        return [int(lg.argmax())]
    return fn


def _apply_for_value(agent, a):
    seat = agent.seatWind
    if PLAY <= a < CHI0:
        tile = TILE_LIST[a - PLAY]
        ag2 = copy.deepcopy(agent)
        try:
            o = ag2.request2obs("Player %d Play %s" % (seat, tile)); return o["observation"]
        except Exception:
            return None
    if CHI0 <= a < PENG0:
        ci = a - CHI0; tt = ci // 3
        suit = "WTB"[tt // 7]; mid = tt % 7 + 2
        ag2 = copy.deepcopy(agent)
        try:
            o = ag2.request2obs("Player %d Chi %s%d" % (seat, suit, mid)); return o["observation"]
        except Exception:
            return None
    if PENG0 <= a < GANG0:
        ag2 = copy.deepcopy(agent)
        try:
            o = ag2.request2obs("Player %d Peng" % seat); return o["observation"]
        except Exception:
            return None
    return None


def risk_seeking_action(lg, mask, agent, vnet, lam, mu, topk, cur_obs):
    """Risk-seeking: argmax lg[a] + lam*V_score_after(a) - mu*p4_after(a). lam<=0,mu<=0 -> base."""
    legal = np.flatnonzero(mask)
    if (lam <= 0 and mu <= 0) or legal.size == 0:
        a = int(lg.argmax())
        return a if mask[a] else int(legal[0])
    order = legal[np.argsort(lg[legal])[::-1]][:topk]
    obs_list = []; idx_map = []
    for a in order:
        po = _apply_for_value(agent, int(a))
        if po is None:
            obs_list.append(cur_obs); idx_map.append(int(a))
        else:
            obs_list.append(po); idx_map.append(int(a))
    score, p4 = _value_score_p4(vnet, np.stack(obs_list))
    best_a = None; best_s = -1e18
    for j, a in enumerate(idx_map):
        s = float(lg[a]) + lam * float(score[j]) - mu * float(p4[j])
        if s > best_s:
            best_s = s; best_a = a
    return best_a


class PlacementSim(Sim):
    def __init__(self, *a, cand_seat=0, vnet=None, lam=0.0, mu=0.0, topk=5, stats=None, **k):
        super().__init__(*a, **k)
        self.cand_seat = cand_seat
        self.vnet = vnet; self.lam = lam; self.mu = mu; self.topk = topk
        self.stats = stats

    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        raw_a = int(lg.argmax())
        if not mask[raw_a]:
            raw_a = int(np.flatnonzero(mask)[0])
        active = (self.lam > 0 or self.mu > 0)
        if seat == self.cand_seat and active:
            agent = self.cai[seat]
            act = risk_seeking_action(lg, mask, agent, self.vnet, self.lam, self.mu, self.topk, obs)
            if self.stats is not None:
                claim_legal = bool(mask[0]) and any(mask[i] for i in range(CHI0, GANG0))
                if claim_legal:
                    self.stats["claim_legal"] += 1
                    if CHI0 <= act < GANG0:
                        self.stats["claim_kept"] += 1
                if act != raw_a:
                    self.stats["overridden"] += 1
                self.stats["cand_decisions"] += 1
        else:
            act = raw_a
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if seat in self.learner_seats:
            self.traj[seat].append((obs, mask, act))
        return act


def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg, vpath, lam, mu, topk = arg
    mc = _load_policy(cand, ck, ccfg); mr = _load_policy(ref, rk, rcfg)
    vnet = _load_value(vpath) if (lam > 0 or mu > 0) else None
    fc = _greedy_lg(mc); fr = _greedy_lg(mr)
    pts = [0, 0, 0, 0]; placement_sum = 0.0; micro_cand = 0
    stats = {"claim_legal": 0, "claim_kept": 0, "overridden": 0, "cand_decisions": 0}
    pg_rank = []
    for cs in range(4):
        pols = [fr, fr, fr, fr]; pols[cs] = fc
        sim = PlacementSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True,
                           cand_seat=cs, vnet=vnet, lam=lam, mu=mu, topk=topk, stats=stats)
        sim.play()
        sc = sim.scores
        cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)
        avg_rank = greater + (equal + 1) / 2.0
        ppt = 5.0 - avg_rank
        placement_sum += ppt; micro_cand += cand_score
        r = max(0, min(3, int(round(avg_rank)) - 1)); pts[r] += 1
        pg_rank.append(float(avg_rank))
    return placement_sum, pts, micro_cand, stats, pg_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn_fused")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn_fused")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--value", default="ckpt/value_256x40.pkl")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg, a.value, a.lam, a.mu, a.topk)
            for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    tot_pts = sum(r[0] for r in res)
    dist = [0, 0, 0, 0]; micro = 0
    st = {"claim_legal": 0, "claim_kept": 0, "overridden": 0, "cand_decisions": 0}
    pg_rank = []
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        micro += r[2]
        for k in st: st[k] += r[3][k]
        pg_rank.extend(r[4])
    pg_rank = np.asarray(pg_rank, dtype=np.float64)
    placement_pts = tot_pts / ngames if ngames else 0.0
    out = dict(cand=os.path.basename(a.cand), ref=os.path.basename(a.ref),
               value=os.path.basename(a.value), lam=a.lam, mu=a.mu, topk=a.topk,
               games=ngames, seeds=len(res),
               placement_pts=round(placement_pts, 4),
               dist_1234=dist,
               dist_pct=[round(100 * d / ngames, 2) for d in dist] if ngames else [0, 0, 0, 0],
               first_pct=round(100 * dist[0] / ngames, 2) if ngames else 0.0,
               fourth_pct=round(100 * dist[3] / ngames, 2) if ngames else 0.0,
               sg_win_rate=round(float((pg_rank <= 1.5).mean()), 4),
               sg_fourth_rate=round(float((pg_rank >= 3.5).mean()), 4),
               claim_rate=round(st["claim_kept"] / st["claim_legal"], 4) if st["claim_legal"] else 0.0,
               claim_legal_states=st["claim_legal"],
               override_rate=round(st["overridden"] / st["cand_decisions"], 4) if st["cand_decisions"] else 0.0,
               cand_decisions=st["cand_decisions"],
               micro_cand_per_game=round(micro / ngames, 3) if ngames else 0.0,
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0:
        print("FAIL: n=0 games", flush=True); sys.exit(2)


if __name__ == "__main__":
    main()

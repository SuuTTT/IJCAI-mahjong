"""
g6_engine.py -- GPU-BATCHED lockstep re-implementation of the s6 PIMC value-cutoff search,
now with BELIEF-WEIGHTED determinization and a PLACEMENT-head leaf (2027 upgrades).

GAME LOGIC is a line-for-line port of sim_cnn.Sim._loop / _resolve_claims /
_check_claims_hu_only and s6_pimc_vcut.PIMCVSim._rollout_vcut, transformed into GENERATORS:
every `self._ask(seat)` (kd forward) and every leaf eval becomes a `yield` of an inference
REQUEST. A coordinator drives ALL (n_worlds x n_candidates) rollouts of one search decision
in LOCKSTEP: advance every live rollout to its next yield, gather all pending requests into
ONE batch, run ONE batched forward, scatter results back via .send().

Evaluators (MODE): "exact" -> one-at-a-time CPU forwards w/ s6's own _ens_logits (bit-for-bit
identical to s6); "gpu" -> one batched GPU forward (~100x). The TOP-LEVEL played game always
uses single-sample CPU _kd_ask, so the played game is bit-identical to s6; only ROLLOUT
inference is batched.

UPGRADES (both default OFF -> reproduces the validated g6 exactly):
  * belief=True   : determinization samples opponent hands BIASED toward the opponent-belief
                    ensemble (oppbelief_more60k, sigmoid->(3,34) per-rel-opp per-tile hold-prob)
                    via weighted sampling-without-replacement (Efraimidis-Spirakis), respecting
                    exact pool tile-counts and each opponent's hand size. Same 3-hand-view resync
                    + reject-resample net as the uniform path.
  * leaf="placement": objective = expected PLACEMENT. Terminated rollout (HU / real wall-HUANG)
                    -> ACTUAL _placement(scores,seat); cutoff (K plies elapsed, hand ongoing)
                    -> placement-head ensemble estimate (placeval, VNet cond=False, 1..4).
    leaf="score"  : the validated g6 objective (terminated/cutoff -> actual score; the score-
                    head leaf is unreachable by construction, exactly as in s6 -> equivalence).
"""
import os, sys, copy
sys.path.insert(0, "/root/caiest_repro")
import numpy as np
import torch
import s6_pimc_vcut as s6
from s6_pimc_vcut import _determinize, _placement, _ens_logits, KD, DEPLOY_SRC, MAX_LEGAL
from sim_cnn import Sim, ACT, TILE_LIST, TILE_INDEX, decode_chi, _fan

BELIEF_CKPTS = ["ckpt/oppbelief/oppbelief_more60k_s20.pt",
                "ckpt/oppbelief/oppbelief_more60k_s21.pt",
                "ckpt/oppbelief/oppbelief_more60k_s22.pt"]
PLACE_CKPTS = ["ckpt/placeval/placeval_s0.pt",
               "ckpt/placeval/placeval_s1.pt",
               "ckpt/placeval/placeval_s2.pt"]

_R = {"mode": "exact", "kd_cpu": None, "kd_gpu": None, "dev": None,
      "bel_cpu": None, "bel_gpu": None, "pl_cpu": None, "pl_gpu": None}


def init_infer(mode="exact", null=False, device="cuda:0", want_belief=False, want_place=False):
    s6.models_explore.IN_PLANES = 38
    from models_explore import build
    kd_cpu = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
        kd_cpu.append(m)
    _R["kd_cpu"] = kd_cpu; _R["mode"] = mode
    s6._G["kd"] = kd_cpu                                   # s6's exact _ens_logits path
    bel_cpu = pl_cpu = None
    if want_belief and not null:
        from oppbelief_train import BeliefFused
        bel_cpu = []
        for p in BELIEF_CKPTS:
            b = BeliefFused(128, 40); b.load_state_dict(torch.load(p, map_location="cpu"))
            b.eval(); bel_cpu.append(b)
    if want_place and not null:
        from f2_value_v2 import VNet
        pl_cpu = []
        for p in PLACE_CKPTS:
            v = VNet(cond=False); v.load_state_dict(torch.load(p, map_location="cpu"))
            v.eval(); pl_cpu.append(v)
    _R["bel_cpu"] = bel_cpu; _R["pl_cpu"] = pl_cpu
    if mode == "gpu":
        dev = torch.device(device); _R["dev"] = dev
        kd_gpu = []
        for m in kd_cpu:
            g = build("resbn_fused", channels=128, blocks=40)
            g.load_state_dict(m.state_dict()); g.eval(); g.to(dev); kd_gpu.append(g)
        _R["kd_gpu"] = kd_gpu
        if bel_cpu is not None:
            from oppbelief_train import BeliefFused
            bg = []
            for b in bel_cpu:
                g = BeliefFused(128, 40); g.load_state_dict(b.state_dict()); g.eval(); g.to(dev); bg.append(g)
            _R["bel_gpu"] = bg
        if pl_cpu is not None:
            from f2_value_v2 import VNet
            pg = []
            for v in pl_cpu:
                g = VNet(cond=False); g.load_state_dict(v.state_dict()); g.eval(); g.to(dev); pg.append(g)
            _R["pl_gpu"] = pg


# ---------------------------------------------------------------------------
# Policy evaluators (kd ensemble)
# ---------------------------------------------------------------------------
def _resolve_action(logits, mask):
    a = int(np.argmax(logits))
    if not mask[a]:
        a = int(np.flatnonzero(mask)[0])
    return a


def eval_policy_exact(reqs):
    out = []
    for obs, mask in reqs:
        lg = _ens_logits(obs[None, :], mask[None, :])
        out.append(_resolve_action(lg, mask))
    return out


@torch.no_grad()
def eval_policy_gpu(reqs):
    dev = _R["dev"]
    obs = torch.from_numpy(np.stack([r[0] for r in reqs]).astype(np.float32)).to(dev)
    mask = torch.from_numpy(np.stack([r[1] for r in reqs])).to(dev)
    mkf = mask.float(); acc = None
    for m in _R["kd_gpu"]:
        lg = m({"is_training": False, "obs": {"observation": obs, "action_mask": mask}})
        lg = torch.where(mask, lg, torch.full_like(lg, -1e30))
        lg = lg - lg.max(dim=1, keepdim=True).values
        p = torch.exp(lg) * mkf
        s = p.sum(dim=1, keepdim=True)
        p = torch.where(s > 0, p / s, mkf / mkf.sum(dim=1, keepdim=True).clamp(min=1.0))
        acc = p if acc is None else acc + p
    logavg = torch.log(torch.clamp(acc / len(_R["kd_gpu"]), min=1e-30))
    am = logavg.argmax(dim=1).cpu().numpy(); mask_np = mask.cpu().numpy()
    out = []
    for i in range(len(reqs)):
        a = int(am[i])
        if not mask_np[i, a]:
            a = int(np.flatnonzero(mask_np[i])[0])
        out.append(a)
    return out


def _eval_policy(reqs):
    return eval_policy_gpu(reqs) if _R["mode"] == "gpu" else eval_policy_exact(reqs)


# ---------------------------------------------------------------------------
# Placement-head leaf evaluator (VNet cond=False -> placement 1..4, ensemble mean)
# ---------------------------------------------------------------------------
def eval_place_exact(reqs):
    out = []
    for obs in reqs:
        x = torch.from_numpy(np.ascontiguousarray(obs).astype(np.float32))[None, :]
        with torch.no_grad():
            vs = [float(v(x, None).item()) for v in _R["pl_cpu"]]
        out.append(sum(vs) / len(vs))
    return out


@torch.no_grad()
def eval_place_gpu(reqs):
    dev = _R["dev"]
    x = torch.from_numpy(np.stack(reqs).astype(np.float32)).to(dev)
    acc = None
    for v in _R["pl_gpu"]:
        o = v(x, None).float()
        acc = o if acc is None else acc + o
    mean = acc / len(_R["pl_gpu"])
    return [float(z) for z in mean.cpu().numpy()]


def _eval_place(reqs):
    return eval_place_gpu(reqs) if _R["mode"] == "gpu" else eval_place_exact(reqs)


# ---------------------------------------------------------------------------
# Opponent-belief array (mean sigmoid over the ensemble) -> (3,34)
# ---------------------------------------------------------------------------
@torch.no_grad()
def belief_array(obs38):
    if _R["mode"] == "gpu":
        dev = _R["dev"]
        x = torch.from_numpy(np.ascontiguousarray(obs38).astype(np.float32))[None, :].to(dev)
        acc = None
        for b in _R["bel_gpu"]:
            p = torch.sigmoid(b(x))
            acc = p if acc is None else acc + p
        arr = (acc / len(_R["bel_gpu"])).cpu().numpy().reshape(3, 34)
    else:
        x = torch.from_numpy(np.ascontiguousarray(obs38).astype(np.float32))[None, :]
        acc = None
        for b in _R["bel_cpu"]:
            p = torch.sigmoid(b(x)).numpy()
            acc = p if acc is None else acc + p
        arr = (acc / len(_R["bel_cpu"])).reshape(3, 34)
    return arr.astype(np.float64)


def _determinize_belief(world, search_seat, rng, belief):
    """Belief-weighted determinization. Opponent hands are sampled from the hidden pool by
    weighted sampling-without-replacement (Efraimidis-Spirakis), weight = belief hold-prob for
    that (rel-opponent, tile-type). Walls get the remainder (uniform). Exact tile-counts and
    hand sizes preserved. All THREE hand views resynced (== uniform _determinize)."""
    pool = []
    hand_sizes = {}
    for s in range(4):
        if s != search_seat:
            hand_sizes[s] = len(world.hand[s]); pool.extend(world.hand[s])
    wall_sizes = [len(w) for w in world.walls]
    for w in world.walls:
        pool.extend(w)
    remaining = list(pool)
    assigned = {}
    for rel in (1, 2, 3):                                  # next, across, prev (seat-relative)
        o = (search_seat + rel) % 4
        H = hand_sizes[o]
        w = np.array([belief[rel - 1, TILE_INDEX[t]] for t in remaining], dtype=np.float64)
        w = np.clip(w, 1e-6, None)
        u = rng.random_sample(len(remaining))
        keys = u ** (1.0 / w)                              # ES weighted sample-without-replacement
        order = np.argsort(-keys)
        pick = order[:H]
        pickset = set(int(i) for i in pick)
        assigned[o] = [remaining[int(i)] for i in pick]
        remaining = [remaining[i] for i in range(len(remaining)) if i not in pickset]
    rng.shuffle(remaining)
    for s in range(4):
        if s != search_seat:
            nh = list(assigned[s])
            world.hand[s] = nh
            if world.cai is not None:
                world.cai[s].hand = list(nh); world.cai[s]._hand_embedding_update()
            if getattr(world, "agents", None) is not None:
                world.agents[s].hand = list(nh); world.agents[s]._update_hand()
    i = 0
    for s in range(4):
        world.walls[s] = list(remaining[i:i + wall_sizes[s]]); i += wall_sizes[s]


# ===========================================================================
class G6Sim(Sim):
    search_seat = 0
    null = False
    true_state = False
    n_worlds = 1
    k_cutoff = 6
    belief = False           # belief-weighted determinization
    leaf = "score"           # "score" (validated g6) | "placement"

    def _kd_ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"], None, None
        lg = _ens_logits(obs[None, :], mask[None, :])
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act, lg, mask

    def _ask_g(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        act = yield ("policy", obs, mask)
        return act

    def _loop_g(self, max_turns):
        quan = self.quan
        for _ in range(max_turns):
            cur = self.cur
            if not self.walls[cur]:
                return "HUANG"                              # genuine wall-exhaustion draw
            t = self.walls[cur].pop()
            self.hand[cur].append(t)
            self.agents[cur].update(f"Draw {t}")
            for s in range(4):
                if s != cur: self.agents[s].update(f"Player {cur} Draw")
            if self.cai is not None:
                self.cai[cur].request2obs(f"Draw {t}")
                for s in range(4):
                    if s != cur: self.cai[s].request2obs(f"Player {cur} Draw")
            act = yield from self._ask_g(cur)
            if act == ACT["Hu"]:
                f = _fan(self.hand[cur], self.melds[cur], t, cur, quan, True, False)
                if f >= 8:
                    self._score_selfdraw(cur, f); return "HU"
                act = ACT["Play"] + TILE_INDEX[self.hand[cur][0]]
            if ACT["AnGang"] <= act < ACT["BuGang"]:
                tile = TILE_LIST[act - ACT["AnGang"]]
                if self.hand[cur].count(tile) == 4 and len(self.walls[cur]) > 0:
                    for _ in range(4): self.hand[cur].remove(tile)
                    self.melds[cur].append(("GANG", tile))
                    self._broadcast(f"Player {cur} AnGang {tile}")
                    continue
                act = ACT["Play"] + TILE_INDEX[self.hand[cur][0]]
            if act >= ACT["BuGang"]:
                tile = TILE_LIST[act - ACT["BuGang"]]
                if tile in self.hand[cur] and any(m[0] == "PENG" and m[1] == tile for m in self.melds[cur]) and self.walls[cur]:
                    self.hand[cur].remove(tile)
                    for i, m in enumerate(self.melds[cur]):
                        if m[0] == "PENG" and m[1] == tile: self.melds[cur][i] = ("GANG", tile)
                    self._broadcast(f"Player {cur} BuGang {tile}")
                    rob = yield from self._check_claims_hu_only_g(tile, cur, is_kong=True)
                    if rob is not None: return "HU"
                    continue
                act = ACT["Play"] + TILE_INDEX[self.hand[cur][0]]
            tile = TILE_LIST[act - ACT["Play"]] if ACT["Play"] <= act < ACT["Chi"] else self.hand[cur][0]
            if tile not in self.hand[cur]:
                tile = self.hand[cur][0]
            self.hand[cur].remove(tile)
            self._broadcast(f"Player {cur} Play {tile}")
            nxt = yield from self._resolve_claims_g(tile, cur)
            if nxt == "HU":
                return "HU"
            self.cur = nxt
        return "CUTOFF"                                     # K plies elapsed, hand ongoing

    def _check_claims_hu_only_g(self, tile, src, is_kong):
        order = [(src + i) % 4 for i in range(1, 4)]
        for s in order:
            f = _fan(self.hand[s], self.melds[s], tile, s, self.quan, False, is_kong)
            if f >= 8:
                self.agents[s].valid = [ACT["Hu"], ACT["Pass"]]
                a = yield from self._ask_g(s)
                if a == ACT["Hu"]:
                    self._score_rong(s, src, f); return s
        return None

    def _resolve_claims_g(self, tile, src):
        order = [(src + i) % 4 for i in range(1, 4)]
        for s in order:
            if not self.walls[src] and len(self.walls[(src + 1) % 4]) == 0:
                pass
            f = _fan(self.hand[s], self.melds[s], tile, s, self.quan, False, False)
            if f >= 8:
                self.agents[s].update(f"__noop")
                self._score_rong(s, src, f); return "HU"
        for s in order:
            cnt = self.hand[s].count(tile)
            if cnt >= 2 and self.walls[s]:
                self.agents[s].update(f"Player {src} Play {tile}")
                a = yield from self._ask_g(s)
                self.agents[s]
                if ACT["Gang"] <= a < ACT["AnGang"] and cnt >= 3:
                    for _ in range(3): self.hand[s].remove(tile)
                    self.melds[s].append(("GANG", tile))
                    self._broadcast(f"Player {s} Gang")
                    return s
                if ACT["Peng"] <= a < ACT["Gang"]:
                    for _ in range(2): self.hand[s].remove(tile)
                    self.melds[s].append(("PENG", tile))
                    self._broadcast(f"Player {s} Peng")
                    d = yield from self._ask_g(s)
                    dt = TILE_LIST[d - ACT["Play"]] if ACT["Play"] <= d < ACT["Chi"] else self.hand[s][0]
                    if dt not in self.hand[s]: dt = self.hand[s][0]
                    self.hand[s].remove(dt)
                    self._broadcast(f"Player {s} Play {dt}")
                    r = yield from self._resolve_claims_g(dt, s)
                    return r
        s = order[0]
        if self.walls[s] and tile[0] in "WBT":
            self.agents[s].update(f"Player {src} Play {tile}")
            chi_opts = [v for v in self.agents[s].valid if ACT["Chi"] <= v < ACT["Peng"]]
            if chi_opts:
                a = yield from self._ask_g(s)
                if ACT["Chi"] <= a < ACT["Peng"]:
                    suit, mid, _ = decode_chi(a)
                    ok = True; rem = []
                    for d in (-1, 0, 1):
                        x = f"{suit}{mid+d}"
                        if x == tile: continue
                        if x in self.hand[s]: self.hand[s].remove(x); rem.append(x)
                        else: ok = False
                    if ok:
                        self.melds[s].append(("CHI", f"{suit}{mid}"))
                        self._broadcast(f"Player {s} Chi {suit}{mid}")
                        d2 = yield from self._ask_g(s)
                        dt = TILE_LIST[d2 - ACT["Play"]] if ACT["Play"] <= d2 < ACT["Chi"] else self.hand[s][0]
                        if dt not in self.hand[s]: dt = self.hand[s][0]
                        self.hand[s].remove(dt)
                        self._broadcast(f"Player {s} Play {dt}")
                        r = yield from self._resolve_claims_g(dt, s)
                        return r
                    else:
                        for x in rem: self.hand[s].append(x)
        return order[0]

    def _rollout_vcut_g(self, seat, tile, K):
        if tile not in self.hand[seat]:
            tile = self.hand[seat][0]
        self.hand[seat].remove(tile)
        self._broadcast(f"Player {seat} Play {tile}")
        place = (self.leaf == "placement")
        nxt = yield from self._resolve_claims_g(tile, seat)
        if nxt == "HU":
            return _placement(self.scores, seat) if place else float(self.scores[seat])
        self.cur = nxt
        res = yield from self._loop_g(K)
        if place:
            if res in ("HU", "HUANG"):                      # terminal (win or real wall draw)
                return _placement(self.scores, seat)
            obs, _ = self._obs_mask(seat)                   # res == "CUTOFF": placement leaf
            v = yield ("place", obs)
            return v
        else:                                               # validated g6 score objective
            # HU / HUANG / CUTOFF all terminal-score (score-head leaf unreachable, as in s6)
            return float(self.scores[seat])

    def _ask(self, seat):
        self._asks = getattr(self, "_asks", 0) + 1
        kd_act, kd_lg, mask = self._kd_ask(seat)
        if seat != self.search_seat or self.null or kd_lg is None:
            return kd_act
        if not (ACT["Play"] <= kd_act < ACT["Chi"]):
            return kd_act
        legal = [a for a in range(ACT["Play"], ACT["Chi"]) if mask[a]]
        if len(legal) <= 1:
            return kd_act
        self._decisions = getattr(self, "_decisions", 0) + 1
        if len(legal) > MAX_LEGAL:
            self._skipped = getattr(self, "_skipped", 0) + 1
            return kd_act
        nW = 1 if self.true_state else self.n_worlds
        best_act = self._batched_search(seat, legal, kd_act, nW)
        if best_act != kd_act:
            self._override = getattr(self, "_override", 0) + 1
        return best_act

    def _batched_search(self, seat, legal, kd_act, nW):
        belief = None
        if self.belief and not self.true_state:
            belief = belief_array(self._obs_mask(seat)[0])
        worlds = []
        for _w in range(nW):
            world = copy.deepcopy(self)
            world.search_seat = -1
            if not self.true_state:
                if belief is not None:
                    _determinize_belief(world, seat, self._rng, belief)
                else:
                    _determinize(world, seat, self._rng)
            worlds.append(world)
        rolls = []
        for wi, world in enumerate(worlds):
            for a in legal:
                tile = TILE_LIST[a - ACT["Play"]]
                clone = copy.deepcopy(world)
                g = clone._rollout_vcut_g(seat, tile, self.k_cutoff)
                rolls.append({"wi": wi, "a": a, "gen": g, "send": None, "done": False, "val": None})
        while True:
            preqs = []; pidx = []; vreqs = []; vidx = []; any_live = False
            for r in rolls:
                if r["done"]:
                    continue
                any_live = True
                try:
                    req = r["gen"].send(r["send"])
                except StopIteration as e:
                    r["done"] = True; r["val"] = e.value; continue
                if req[0] == "policy":
                    pidx.append(r); preqs.append((req[1], req[2]))
                else:                                        # "place"
                    vidx.append(r); vreqs.append(req[1])
            if not any_live or (not preqs and not vreqs):
                break
            if preqs:
                for r, a in zip(pidx, _eval_policy(preqs)):
                    r["send"] = a
            if vreqs:
                for r, v in zip(vidx, _eval_place(vreqs)):
                    r["send"] = v
        val_sum = {a: 0.0 for a in legal}
        by_world = {}
        for r in rolls:
            by_world.setdefault(r["wi"], {})[r["a"]] = r["val"]
        w = 0
        for wi in range(nW):
            wv = by_world.get(wi, {})
            if any(wv.get(a) is None for a in legal):
                self._bad_world = getattr(self, "_bad_world", 0) + 1
                continue
            for a in legal:
                val_sum[a] += wv[a]
            w += 1
        self._good_world = getattr(self, "_good_world", 0) + w
        if w == 0:
            return kd_act
        best_key = None; best_act = kd_act
        for a in legal:
            key = (val_sum[a] / w, 1 if a == kd_act else 0)
            if best_key is None or key > best_key:
                best_key = key; best_act = a
        return best_act

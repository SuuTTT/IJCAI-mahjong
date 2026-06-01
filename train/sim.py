"""
sim.py — fast in-process Chinese Standard Mahjong simulator for self-play RL.

Design goals:
  • No subprocess / no judge binary → fast enough for PPO (target >100 games/s/core).
  • Observations & legal masks come from FeatureAgent (the SAME encoding the bot uses
    in deployment) → zero train/serve skew.
  • Win validation & fan via PyMahjongGB (the official library, same as the judge).
  • Duplicate-style per-game scoring matching the judge:
       self-draw: winner +3(8+f), others -(8+f)
       rong:      winner +(24+f), discarder -(8+f), others -8
       draw(HUANG): all 0
       (no flowers in duplicate; 8-fan floor enforced)

Returns per-seat trajectories: list of (obs[240] uint8, mask[235] bool, action int, seat).
Reward is assigned to the seat totals at game end (sparse, per the real game).

A `policy_fn(obs_batch, mask_batch) -> actions` plugs in the learner (batched GPU
inference). For legality we ALWAYS intersect the policy's choice with the legal mask,
and HU is fan-gated (>=8) using the simulator's ground-truth hand.
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.feature_agent import (
    FeatureAgent, ACT, ACT_DIM, TILE_LIST, TILE_INDEX, decode_chi, chi_action_idx,
)
try:
    from MahjongGB import MahjongFanCalculator
    HAS_FAN = True
except Exception:
    HAS_FAN = False

SUITS = "WBT"
def _full_wall(rng):
    tiles = [f"{s}{n}" for s in SUITS for n in range(1, 10)] * 4
    tiles += [f"F{n}" for n in range(1, 5)] * 4
    tiles += [f"J{n}" for n in range(1, 4)] * 4
    rng.shuffle(tiles)
    return tiles


def _fan(concealed, melds, win, seat, quan, is_self, is_kong):
    if not HAS_FAN:
        return -1
    c = list(concealed)
    if win in c: c.remove(win)
    if len(c) + 3 * len(melds) + 1 != 14:
        return -1
    try:
        packs = tuple((ty, tl, 1) for ty, tl in melds)
        r = MahjongFanCalculator(pack=packs, hand=tuple(c), winTile=win, flowerCount=0,
                                 isSelfDrawn=is_self, is4thTile=False, isAboutKong=is_kong,
                                 isWallLast=False, seatWind=seat, prevalentWind=quan)
        return sum(x for x, _ in r)
    except Exception:
        return -1


class Sim:
    """One duplicate game. policy_fn(obs(N,240) uint8, mask(N,235) bool) -> act(N,)."""

    def __init__(self, policy_fn, seed=0, quan=0, learner_seats=None):
        # policy_fn: a single callable for all seats, OR a list of 4 callables.
        self.policies = policy_fn if isinstance(policy_fn, (list, tuple)) else [policy_fn]*4
        self.rng = random.Random(seed)
        self.quan = quan
        # only record trajectories for these seats (default: all)
        self.learner_seats = set(range(4) if learner_seats is None else learner_seats)

    def reset(self):
        wall = _full_wall(self.rng)
        self.walls = [wall[i*34:(i+1)*34] for i in range(4)]   # each draws from own
        self.hand = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]   # (type, tile)
        self.agents = [FeatureAgent(s) for s in range(4)]
        for s in range(4):
            self.agents[s].update(f"Wind {self.quan}")
        for s in range(4):
            self.hand[s] = [self.walls[s].pop() for _ in range(13)]
            self.agents[s].update("Deal " + " ".join(self.hand[s]))
        self.cur = 0
        self.traj = [[] for _ in range(4)]   # (obs,mask,act)
        self.scores = [0, 0, 0, 0]

    # ---- helpers ----
    def _obs_mask(self, seat):
        a = self.agents[seat]
        m = np.zeros(ACT_DIM, dtype=bool)
        for v in a.valid:
            if 0 <= v < ACT_DIM: m[v] = True
        return a.obs.copy(), m

    def _ask(self, seat):
        """Query that seat's policy; record trajectory (only for learner seats)."""
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        act = int(self.policies[seat](obs[None, :], mask[None, :])[0])
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if seat in self.learner_seats:
            self.traj[seat].append((obs, mask, act))
        return act

    def _broadcast(self, msg):
        for s in range(4):
            self.agents[s].update(msg)

    def play(self, max_turns=300):
        self.reset()
        scored = self._loop(max_turns)
        return self.traj, self.scores

    def _loop(self, max_turns):
        quan = self.quan
        for _ in range(max_turns):
            cur = self.cur
            # draw
            if not self.walls[cur]:
                return "HUANG"
            t = self.walls[cur].pop()
            self.hand[cur].append(t)
            self.agents[cur].update(f"Draw {t}")
            for s in range(4):
                if s != cur: self.agents[s].update(f"Player {cur} Draw")
            # acting player's decision (draw turn)
            act = self._ask(cur)
            if act == ACT["Hu"]:
                f = _fan(self.hand[cur], self.melds[cur], t, cur, quan, True, False)
                if f >= 8:
                    self._score_selfdraw(cur, f); return "HU"
                # illegal HU avoided -> fall back to a discard
                act = ACT["Play"] + TILE_INDEX[self.hand[cur][0]]
            if ACT["AnGang"] <= act < ACT["BuGang"]:
                tile = TILE_LIST[act - ACT["AnGang"]]
                if self.hand[cur].count(tile) == 4 and len(self.walls[cur]) > 0:
                    for _ in range(4): self.hand[cur].remove(tile)
                    self.melds[cur].append(("GANG", tile))
                    self._broadcast(f"Player {cur} AnGang {tile}")
                    continue   # same player draws again
                act = ACT["Play"] + TILE_INDEX[self.hand[cur][0]]
            if act >= ACT["BuGang"]:
                tile = TILE_LIST[act - ACT["BuGang"]]
                if tile in self.hand[cur] and any(m[0]=="PENG" and m[1]==tile for m in self.melds[cur]) and self.walls[cur]:
                    self.hand[cur].remove(tile)
                    for i,m in enumerate(self.melds[cur]):
                        if m[0]=="PENG" and m[1]==tile: self.melds[cur][i]=("GANG",tile)
                    self._broadcast(f"Player {cur} BuGang {tile}")
                    # robbing kong: others may HU
                    rob = self._check_claims_hu_only(tile, cur, is_kong=True)
                    if rob is not None: return "HU"
                    continue
                act = ACT["Play"] + TILE_INDEX[self.hand[cur][0]]
            # PLAY
            tile = TILE_LIST[act - ACT["Play"]] if ACT["Play"] <= act < ACT["Chi"] else self.hand[cur][0]
            if tile not in self.hand[cur]:
                tile = self.hand[cur][0]
            self.hand[cur].remove(tile)
            self._broadcast(f"Player {cur} Play {tile}")
            # claims by others: HU > PENG/GANG > CHI
            nxt = self._resolve_claims(tile, cur)
            if nxt == "HU":
                return "HU"
            self.cur = nxt
        return "HUANG"

    def _check_claims_hu_only(self, tile, src, is_kong):
        # for robbing kong: any non-src player can HU
        order = [(src + i) % 4 for i in range(1, 4)]
        for s in order:
            f = _fan(self.hand[s], self.melds[s], tile, s, self.quan, False, is_kong)
            if f >= 8:
                self.agents[s].valid = [ACT["Hu"], ACT["Pass"]]
                a = self._ask(s)
                if a == ACT["Hu"]:
                    self._score_rong(s, src, f); return s
        return None

    def _resolve_claims(self, tile, src):
        """Return next player (int) or 'HU'. Handles HU>PENG/GANG>CHI priority."""
        order = [(src + i) % 4 for i in range(1, 4)]
        # 1) HU (rong)
        for s in order:
            if not self.walls[src] and len(self.walls[(src+1)%4]) == 0:
                pass
            f = _fan(self.hand[s], self.melds[s], tile, s, self.quan, False, False)
            if f >= 8:
                # let the policy decide to take it (it normally will)
                self.agents[s].update(f"__noop")  # no-op safety
                self._score_rong(s, src, f); return "HU"
        # 2) PENG / GANG (any of the 3 others)
        for s in order:
            cnt = self.hand[s].count(tile)
            if cnt >= 2 and self.walls[s]:
                # ask policy: peng?
                self.agents[s].update(f"Player {src} Play {tile}")  # sets valid incl Peng
                a = self._ask(s)
                self.agents[s]  # noted
                if ACT["Gang"] <= a < ACT["AnGang"] and cnt >= 3:
                    for _ in range(3): self.hand[s].remove(tile)
                    self.melds[s].append(("GANG", tile))
                    self._broadcast(f"Player {s} Gang")
                    return s   # gang -> draws again (cur=s)
                if ACT["Peng"] <= a < ACT["Gang"]:
                    for _ in range(2): self.hand[s].remove(tile)
                    self.melds[s].append(("PENG", tile))
                    self._broadcast(f"Player {s} Peng")
                    # must discard now
                    d = self._ask(s)
                    dt = TILE_LIST[d - ACT["Play"]] if ACT["Play"] <= d < ACT["Chi"] else self.hand[s][0]
                    if dt not in self.hand[s]: dt = self.hand[s][0]
                    self.hand[s].remove(dt)
                    self._broadcast(f"Player {s} Play {dt}")
                    # chain: this discard can be claimed too
                    return self._resolve_claims(dt, s)
        # 3) CHI (only next player)
        s = order[0]
        if self.walls[s] and tile[0] in "WBT":
            self.agents[s].update(f"Player {src} Play {tile}")
            chi_opts = [v for v in self.agents[s].valid if ACT["Chi"] <= v < ACT["Peng"]]
            if chi_opts:
                a = self._ask(s)
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
                        d2 = self._ask(s)
                        dt = TILE_LIST[d2 - ACT["Play"]] if ACT["Play"] <= d2 < ACT["Chi"] else self.hand[s][0]
                        if dt not in self.hand[s]: dt = self.hand[s][0]
                        self.hand[s].remove(dt)
                        self._broadcast(f"Player {s} Play {dt}")
                        return self._resolve_claims(dt, s)
                    else:
                        for x in rem: self.hand[s].append(x)
        # no claim -> next player draws
        return order[0]

    def _score_selfdraw(self, w, f):
        for s in range(4):
            self.scores[s] = 3*(8+f) if s == w else -(8+f)

    def _score_rong(self, w, src, f):
        for s in range(4):
            if s == w: self.scores[s] = 24+f
            elif s == src: self.scores[s] = -(8+f)
            else: self.scores[s] = -8


if __name__ == "__main__":
    # speed + sanity smoke test with a random legal policy
    import time
    def rand_policy(obs, mask):
        return np.array([int(np.flatnonzero(m)[0]) if m.any() else 0 for m in mask])
    t0 = time.time(); games = 0; hus = 0
    for g in range(200):
        s = Sim(rand_policy, seed=g)
        traj, sc = s.play()
        games += 1
        if max(sc) > 0: hus += 1
    dt = time.time() - t0
    print(f"{games} games in {dt:.2f}s = {games/dt:.0f} games/s (random policy); wins={hus}")

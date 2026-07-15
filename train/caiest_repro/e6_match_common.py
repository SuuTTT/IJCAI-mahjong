"""e6_match_common.py — E6 Phase 3 (CROSS-HAND field estimation + match switching).

Phase 2 showed WITHIN-hand field identification is at chance (bal-acc 0.34-0.36,
3 classes; E6_ESTIMATOR.json), so the within-hand switcher was a no-op. Phase 3
tests the realistic setting: a MATCH = M=8 hands vs the SAME field (fresh walls
per hand from disjoint seeds, seat 0 fixed, quan=0, no carried state between
hands beyond the running score). The field is estimated from CROSS-HAND evidence
(per-hand public outcomes) and the seat-0 policy can switch from hand 2 on.

Per-hand features (all observable by seat 0 at the table):
  winner seat (or draw), win type (zimo/ron), seat-0 deal-in flag, announced fan,
  per-opponent claim counts (chi / peng / gang incl. angang+bugang), hand length
  (total discards), seat-0 score delta.
Cumulative feature vector after hand h = means over hands 1..h (20 dims).

Because every policy here is deterministic (argmax) and hands are independent
given (field, seed), a hand's outcome is fully determined by (field, seed,
seat0-mode). The switcher evaluation therefore precomputes each eval hand ONCE
under kd (kdens3) and ONCE under aug (aug_s0) and assembles all arms
(always-kd / always-aug / oracle / switcher) from the same outcomes — exact
pairing across arms by construction.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sim_cnn import Sim, ACT
from e6_switch_common import (FIELDS, FIELD_ORDER, CLS_OF_FIELD, CLASSES,
                              KD_PATHS, AUG_PATH, PLAY_LO, PLAY_HI,
                              load_net, policy_fn, preload_all,
                              single_logits, ens_logits)

M_HANDS_DEFAULT = 8

# cumulative-mean feature layout (after hands 1..h)
FEAT_NAMES = ["draw", "w_s0", "w_s1", "w_s2", "w_s3", "zimo", "ron",
              "s0_dealin", "fan", "plays",
              "chi_o1", "peng_o1", "gang_o1",
              "chi_o2", "peng_o2", "gang_o2",
              "chi_o3", "peng_o3", "gang_o3",
              "s0_score"]
NFEAT = len(FEAT_NAMES)


class RecSim(Sim):
    """One hand with a fixed seat-0 mode ('kd' = kdens3 ensemble, 'aug' = aug_s0
    single), recording the public per-hand outcome features."""

    def __init__(self, policy_fn, mode="kd", kd_models=None, aug_model=None, **k):
        super().__init__(policy_fn, **k)
        self.mode = mode
        self.kd = kd_models
        self.aug = aug_model
        self.win_info = None            # (winner, src_or_-1, fan, zimo)
        self.claims = np.zeros((4, 3), dtype=np.int64)   # seat x (chi,peng,gang)
        self.nplays = 0

    def _broadcast(self, msg):
        super()._broadcast(msg)
        p = msg.split()
        if p and p[0] == "Player":
            s = int(p[1]); v = p[2]
            if v == "Play":
                self.nplays += 1
            elif v == "Chi":
                self.claims[s, 0] += 1
            elif v == "Peng":
                self.claims[s, 1] += 1
            elif v in ("Gang", "AnGang", "BuGang"):
                self.claims[s, 2] += 1

    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        if seat == 0:
            lg = (single_logits(self.aug, obs[None, :], mask[None, :])
                  if self.mode == "aug"
                  else ens_logits(self.kd, obs[None, :], mask[None, :]))
        else:
            lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act

    def _score_selfdraw(self, w, f):
        super()._score_selfdraw(w, f)
        self.win_info = (w, -1, f, 1)

    def _score_rong(self, w, src, f):
        super()._score_rong(w, src, f)
        self.win_info = (w, src, f, 0)


def play_hand(field, seed, mode):
    """Play one hand; return a flat per-hand record (dict of scalars/arrays)."""
    kd = [load_net(p) for p in KD_PATHS]
    aug = load_net(AUG_PATH)
    pols = [None] + [policy_fn([p]) for p in FIELDS[field]]
    sim = RecSim(pols, mode=mode, kd_models=kd, aug_model=aug,
                 seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.play()
    sc = list(sim.scores)
    if sim.win_info is None:
        winner, src, fan, zimo = -1, -1, 0, 0
    else:
        winner, src, fan, zimo = sim.win_info
    c = sc[0]
    greater = sum(1 for j in range(4) if sc[j] > c)
    equal = sum(1 for j in range(4) if sc[j] == c)
    rank = greater + (equal + 1) / 2.0
    return dict(winner=winner, zimo=zimo, ron=int(winner >= 0 and not zimo),
                dealin=int(src == 0 and winner != 0), fan=fan, plays=sim.nplays,
                claims=sim.claims[1:].reshape(-1).astype(np.int16),  # opp seats 1-3
                scores=np.asarray(sc, dtype=np.int32), rank=rank)


# ---------- flat storage layout (per hand) ----------
def rec_to_row(rec):
    """13 scalar cols + 9 claim cols + 4 score cols (order below)."""
    return ([rec["winner"], rec["zimo"], rec["ron"], rec["dealin"], rec["fan"],
             rec["plays"], rec["rank"]]
            + list(rec["claims"]) + list(rec["scores"]))

ROW_COLS = (["winner", "zimo", "ron", "dealin", "fan", "plays", "rank"]
            + [f"claim{i}" for i in range(9)] + [f"score{i}" for i in range(4)])
NROW = len(ROW_COLS)


def rows_to_cumfeat(rows):
    """rows: (M, NROW) float array for one match (hand order).
    Returns (M, NFEAT): cumulative-mean features after hands 1..h."""
    M = rows.shape[0]
    winner = rows[:, 0]
    per = np.zeros((M, NFEAT), dtype=np.float64)
    per[:, 0] = (winner < 0)                       # draw
    for s in range(4):
        per[:, 1 + s] = (winner == s)              # w_s0..w_s3
    per[:, 5] = rows[:, 1]                          # zimo
    per[:, 6] = rows[:, 2]                          # ron
    per[:, 7] = rows[:, 3]                          # s0 dealin
    per[:, 8] = rows[:, 4]                          # fan
    per[:, 9] = rows[:, 5]                          # plays
    per[:, 10:19] = rows[:, 7:16]                   # opp claims
    per[:, 19] = rows[:, 16]                        # seat-0 score delta
    cum = np.cumsum(per, axis=0) / np.arange(1, M + 1)[:, None]
    return cum


def match_seed(seed0, field_idx, match_idx, hand_idx, stride=1000000):
    return seed0 + field_idx * stride + match_idx * 16 + hand_idx

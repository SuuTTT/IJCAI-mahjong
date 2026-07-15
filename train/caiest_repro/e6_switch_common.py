"""e6_switch_common.py — shared pieces for E6 Phase 2 (opponent-conditional style switcher).

Phase 1 (FIELD_RESPONSE.json) measured a significant payoff crossing:
aug_s0 beats kdens3 on WEAK fields (+2.56 score/game), kdens3 beats aug_s0 on
CHAMPION fields (+1.90). Phase 2 builds:
  (a) an opponent-strength estimator: CNN on seat-0's (38,4,9) obs snapshot at
      discard-turn t in {4,8,12} -> field class {weak, mixed, strong/champion}
  (b) a switcher policy: play kdens3 for turns < T; at discard turn T run the
      estimator once; if it says WEAK switch to aug_s0 for the rest of the game,
      else stay kdens3. Mode switch = which net's softmax drives decisions.

"Turn" (here and in the estimator training data) = the count of seat-target
decision points where a discard (Play) action is legal, i.e. the seat's k-th
discard decision (post-draw or post-claim). This is fully observable at seat 0.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn
from sim_cnn import Sim, ACT
from models_explore import build

torch.set_num_threads(1)
CKPT = "/root/caiest_repro/ckpt"
KD_PATHS = [f"{CKPT}/kd/kd_128x40_s{i}.pkl" for i in range(3)]
AUG_PATH = f"{CKPT}/aug/aug_128x40_s0.pkl"
FIELDS = {  # EXACT Phase-1 field setups (field_response.py)
    "weak":  [f"{CKPT}/kdfield/kdf_s0.pkl", f"{CKPT}/kdfield/kdf_s1.pkl", f"{CKPT}/kdfield/kdf_s2.pkl"],
    "mixed": [f"{CKPT}/aug/aug_128x40_s0.pkl", f"{CKPT}/kdfield/kdf_s0.pkl", f"{CKPT}/kdfield/kdf_s1.pkl"],
    "strong": [f"{CKPT}/aug/aug_128x40_s0.pkl"] * 3,
    "champion": [f"{CKPT}/kd/kd_128x40_s{i}.pkl" for i in range(3)],
}
FIELD_ORDER = ["weak", "mixed", "strong", "champion"]
CLS_OF_FIELD = {"weak": 0, "mixed": 1, "strong": 2, "champion": 2}  # 3-class target
CLASSES = ["weak", "mixed", "strongchamp"]
PLAY_LO, PLAY_HI = 2, 36          # caiest OFFSET_ACT Play block (feature.py)
SNAP_TURNS = (4, 8, 12)

_CACHE = {}


def load_net(path):
    if path not in _CACHE:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        _CACHE[path] = m
    return _CACHE[path]


def preload_all():
    """Load every net once in the parent so fork()ed pool workers share pages (COW)."""
    for p in set(KD_PATHS + [AUG_PATH] + sum(FIELDS.values(), [])):
        load_net(p)


def _logits(m, obs, mask):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()


def single_logits(m, obs, mask):
    return _logits(m, obs, mask)


def ens_logits(models, obs, mask):
    """EXACT deploy rule (ensemble_infer / e12_score_gate): mean softmax over legal, log."""
    mk = mask.flatten().astype(np.float32)
    acc = None
    for m in models:
        lg = _logits(m, obs, mask)
        lg = np.where(mk > 0, lg, -1e30)
        lg = lg - lg.max()
        p = np.exp(lg) * (mk > 0)
        s = p.sum()
        p = p / s if s > 0 else (mk / max(1.0, mk.sum()))
        acc = p if acc is None else acc + p
    avg = acc / len(models)
    return np.log(np.where(avg > 0, avg, 1e-30))


def single_fn(m):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask)
        return lg if return_logits else [int(lg.argmax())]
    return fn


def ens_fn(models):
    def fn(obs, mask, return_logits=False):
        out = ens_logits(models, obs, mask)
        return out if return_logits else [int(out.argmax())]
    return fn


def policy_fn(paths):
    models = [load_net(p) for p in paths]
    return ens_fn(models) if len(models) > 1 else single_fn(models[0])


class EstNet(nn.Module):
    """Small CNN: (38,4,9) binary obs -> field class."""
    def __init__(self, nclass=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(38, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 9, 128), nn.ReLU(), nn.Linear(128, nclass))

    def forward(self, x):
        return self.head(self.body(x))


def load_estimator(path):
    key = ("EST", path)
    if key not in _CACHE:
        net = EstNet()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        _CACHE[key] = net
    return _CACHE[key]


def est_predict(est, obs):
    with torch.no_grad():
        x = torch.from_numpy((obs > 0).astype(np.float32))[None]
        return int(est(x)[0].argmax())


class SwitchSim(Sim):
    """kdens3 until the target seat's T-th discard decision; there, one estimator
    call on the current obs; pred==weak -> aug_s0 mode for the rest of the game
    (including that decision), else stay kdens3. Other seats: fixed policy fns."""

    def __init__(self, policy_fn, target=0, kd_models=None, aug_model=None,
                 est=None, T=8, **k):
        super().__init__(policy_fn, **k)
        self.target = target
        self.kd = kd_models; self.aug = aug_model; self.est = est; self.T = T
        self.turn = 0; self.mode = "kd"; self.pred = None
        self.dealins = 0; self.wins = 0

    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        if seat == self.target:
            if mask[PLAY_LO:PLAY_HI].any():           # a discard decision = one turn
                self.turn += 1
                if self.est is not None and self.pred is None and self.turn == self.T:
                    self.pred = est_predict(self.est, obs)
                    if self.pred == 0:
                        self.mode = "aug"
            lg = (single_logits(self.aug, obs[None, :], mask[None, :]) if self.mode == "aug"
                  else ens_logits(self.kd, obs[None, :], mask[None, :]))
        else:
            lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act

    def _score_rong(self, w, src, f):
        super()._score_rong(w, src, f)
        if src == self.target: self.dealins += 1
        if w == self.target: self.wins += 1

    def _score_selfdraw(self, w, f):
        super()._score_selfdraw(w, f)
        if w == self.target: self.wins += 1


def game_stats(sim, target):
    sc = sim.scores; c = sc[target]
    greater = sum(1 for j in range(4) if sc[j] > c)
    equal = sum(1 for j in range(4) if sc[j] == c)   # includes self
    rank = greater + (equal + 1) / 2.0
    draw = int(all(s == 0 for s in sc))
    return rank, float(c), int(sim.dealins > 0), int(sim.wins > 0), draw

"""Gym-like single-agent RL environment over the Ludus MCR (Chinese Standard
Mahjong / 国标麻将) engine.

The learner controls **seat 0 (East)**; the other three seats are driven by
supplied Botzone-protocol opponent objects (``.respond(request:str) -> str``).
The real, oracle-validated ``mahjong.validate_adapter.MyEngine`` drives the
game, so every move is legal and every terminal fan/score is the ground-truth
PyMahjongGB result.

Observation / action encoding is REUSED verbatim from the kdens3 champion
bundle (``/root/mcr_champion/feature.py``: ``FeatureAgent``) so the interface is
identical to the one the champion trains/plays on.  The per-request state
machine that turns MyEngine's numeric Botzone stream into ``FeatureAgent``
calls is a faithful copy of the champion's ``__main__.py:process`` (attribution:
caiest / kdens3), with the model call replaced by a coroutine ``yield`` so an
external RL loop supplies the action.

Observation planes  ``obs``  shape ``(38, 4, 9)``  (float32), reshaped from
(38, 36) where the 36 columns are the 34 tile codes W1-9,T1-9,B1-9,F1-4,J1-3
padded to 36 (4x9):

    plane  0        seat wind (門風)      one-hot at the seat-wind honour
    plane  1        prevalent wind (圈風) one-hot at the round-wind honour
    planes 2..5     own concealed hand    count-encoded (k-th plane set if >=k copies)
    planes 6..21    discards, 4 planes x 4 players (self, then downstream)
    planes 22..37   melds (副露), 4 planes x 4 players, tiles laid out flat

Action space  ``n_actions = 235``  with a boolean legality ``mask``:

    0            Pass
    1            Hu   (declare a win)
    2..35        Play tile t          (discard;  2 + tile_index)
    36..98       Chi                  (36 + suit*21 + (num-2)*3 + offset)
    99..132      Peng tile t
    133..166     Gang tile t          (melded kong of a discard)
    167..200     AnGang tile t        (concealed kong)
    201..234     BuGang tile t        (added kong)

Reward (terminal, sparse; zero-sum across seats):  the engine's final MCR score
of seat 0 divided by 8.0.  Concretely, for an 8-fan hand:
    self-draw win  ->  3*(8+fan)/8   = +6.0
    ron   win      ->  (24+fan)/8     = +4.0
    deal-in loss   -> -(8+fan)/8      = -2.0
    passive loss   -> -8/8            = -1.0  (someone else's ron, you didn't discard it)
    passive loss   -> -(8+fan)/8      = -2.0  (someone else's self-draw)
    draw (荒牌)     ->  0.0
Higher-fan wins/losses scale linearly.  An optional ``reward_shaping`` (default
0.0) subtracts a small constant at every decision step (encourages faster wins);
leave at 0.0 for pure terminal reward.

Usage
-----
    from mahjong.rl_env import MahjongEnv
    from mahjong.bots import EfficiencyBot
    env = MahjongEnv()
    obs, mask = env.reset([EfficiencyBot(), EfficiencyBot(), EfficiencyBot()], seed=0)
    done = False
    while not done:
        a = int(np.random.choice(np.flatnonzero(mask)))     # a legal random action
        obs, mask, r, done, info = env.step(a)
    print(info, r)     # e.g. {'ending': 'hu', 'winner': 2, ...}  -1.0
"""

import os
import sys

import numpy as np

from mahjong.engine import build_wall
from mahjong.validate_adapter import MyEngine

# --- reuse the champion's observation/action encoder (do NOT reinvent) --------
_CHAMP_DIR = os.environ.get("MCR_CHAMPION_DIR", "/root/mcr_champion")
if _CHAMP_DIR not in sys.path:
    sys.path.insert(0, _CHAMP_DIR)
from feature import FeatureAgent  # noqa: E402  (adds /root/mcr_champion to path first)

OBS_SHAPE = (FeatureAgent.OBS_SIZE, 4, 9)   # (38, 4, 9)
N_ACTIONS = FeatureAgent.ACT_SIZE           # 235


class FeatureDriver:
    """Stateful seat driver: consumes MyEngine's numeric Botzone request stream,
    maintains a champion ``FeatureAgent``, and exposes decisions as a coroutine.

    ``decide(request)`` is a generator that ``yield``s ``(obs, mask)`` at every
    real decision point and expects the chosen action (int) back via ``.send``;
    it ``return``s the Botzone response string to hand to the engine.  Non-choice
    requests (INIT / deal / others' draws / own echoes) return immediately
    without yielding, so the RL agent only ever sees genuine choices.

    Faithful port of mcr_champion/__main__.py:process (caiest / kdens3), with the
    neural-net call replaced by a yield.
    """

    def __init__(self):
        self.agent = None
        self.seatWind = 0
        self.zimo = False
        self.angang = None

    @staticmethod
    def _pack(obs):
        return (np.asarray(obs["observation"], dtype=np.float32),
                np.asarray(obs["action_mask"], dtype=bool))

    def decide(self, request):
        t = request.split()
        if not t:
            return "PASS"
        if t[0] == "0":                                   # INIT: 0 <seat> <quan>
            self.seatWind = int(t[1])
            self.agent = FeatureAgent(self.seatWind)
            self.zimo = False
            self.angang = None
            self.agent.request2obs("Wind %s" % t[2])
            return "PASS"
        if t[0] == "1":                                   # DEAL
            self.agent.request2obs(" ".join(["Deal", *t[5:]]))
            return "PASS"
        if t[0] == "2":                                   # own draw -> real choice
            obs = self.agent.request2obs("Draw %s" % t[1])
            action = yield self._pack(obs)
            r = self.agent.action2response(action).split()
            if r[0] == "Hu":
                return "HU"
            if r[0] == "Play":
                return "PLAY %s" % r[1]
            if r[0] == "Gang":
                self.angang = r[1]
                return "GANG %s" % r[1]
            if r[0] == "BuGang":
                return "BUGANG %s" % r[1]
            return "PASS"
        if t[0] == "3":
            p = int(t[1])
            if t[2] == "DRAW":
                self.agent.request2obs("Player %d Draw" % p)
                self.zimo = True
                return "PASS"
            if t[2] == "GANG":
                if p == self.seatWind and self.angang:
                    self.agent.request2obs("Player %d AnGang %s" % (p, self.angang))
                elif self.zimo:
                    self.agent.request2obs("Player %d AnGang" % p)
                else:
                    self.agent.request2obs("Player %d Gang" % p)
                return "PASS"
            if t[2] == "BUGANG":
                obs = self.agent.request2obs("Player %d BuGang %s" % (p, t[3]))
                if p == self.seatWind:
                    return "PASS"
                action = yield self._pack(obs)            # rob-the-kong: Hu or Pass
                return "HU" if self.agent.action2response(action) == "Hu" else "PASS"
            # PLAY / CHI / PENG by some player
            self.zimo = False
            if t[2] == "CHI":
                self.agent.request2obs("Player %d Chi %s" % (p, t[3]))
            elif t[2] == "PENG":
                self.agent.request2obs("Player %d Peng" % p)
            obs = self.agent.request2obs("Player %d Play %s" % (p, t[-1]))
            if p == self.seatWind:                        # my own discard echo
                return "PASS"
            action = yield self._pack(obs)                # claim decision
            r = self.agent.action2response(action)
            rs = r.split()
            if rs[0] == "Hu":
                return "HU"
            if rs[0] == "Pass":
                return "PASS"
            if rs[0] == "Gang":
                self.angang = None
                return "GANG"
            if rs[0] in ("Peng", "Chi"):
                # a Peng/Chi claim also needs the follow-up discard -> 2nd choice
                obs2 = self.agent.request2obs("Player %d " % self.seatWind + r)
                action2 = yield self._pack(obs2)
                r2 = self.agent.action2response(action2)
                out = " ".join([rs[0].upper(), *rs[1:], r2.split()[-1]])
                # undo locally: the engine re-echoes the claim, which re-applies it
                self.agent.request2obs("Player %d Un" % self.seatWind + r)
                return out
            return "PASS"
        return "PASS"


class PolicyAgent:
    """Botzone-protocol opponent that plays via a synchronous ``act(obs, mask)``
    callable (e.g. a frozen policy snapshot for self-play).  Reuses the exact
    same ``FeatureDriver`` decision stream as the learner, so opponents and the
    learner share one encoder/legality implementation."""

    def __init__(self, act):
        self.act = act
        self.driver = FeatureDriver()

    def respond(self, request):
        gen = self.driver.decide(request)
        try:
            obs, mask = next(gen)
            while True:
                a = int(self.act(obs, mask))
                obs, mask = gen.send(a)
        except StopIteration as e:
            return e.value


class MahjongEnv:
    """Single-agent MCR env; learner = seat 0 (East), opponents drive seats 1-3.

    See module docstring for obs planes / action layout / reward.
    """

    obs_shape = OBS_SHAPE          # (38, 4, 9)
    n_actions = N_ACTIONS          # 235

    def __init__(self, quan=0, reward_shaping=0.0, reward_scale=8.0):
        self.quan = int(quan)
        self.reward_shaping = float(reward_shaping)
        self.reward_scale = float(reward_scale)
        self._gen = None
        self._last = None

    # ------------------------------------------------------------------ driver
    def _run(self, wall, seed):
        """Full-game generator: yields (obs, mask) at each seat-0 decision,
        receives the action via .send, and returns (reward, info) at terminal."""
        eng = MyEngine()
        driver = FeatureDriver()
        turn = eng.reset(wall, self.quan, seed)
        disp = turn["display"]
        while True:
            disp = turn["display"]
            reqs = turn["requests"]
            if not reqs:                                  # terminal turn (HU/HUANG)
                break
            responses = {}
            for seat_str, req in reqs.items():
                if int(seat_str) == 0:
                    responses[seat_str] = yield from driver.decide(req)
                else:
                    responses[seat_str] = self.opponents[int(seat_str) - 1].respond(req)
            turn = eng.step(responses)
        return self._terminal(disp)

    def _terminal(self, disp):
        act = disp.get("action")
        if act == "HU":
            score = disp.get("score") or [0, 0, 0, 0]
            winner = disp.get("player")
            fan = disp.get("fanCnt")
            reward = score[0] / self.reward_scale
            info = {"ending": "hu", "winner": winner, "fan": fan,
                    "score0": score[0]}
        else:                                             # HUANG / draw
            reward = 0.0
            info = {"ending": "draw", "winner": None, "fan": None, "score0": 0}
        return reward, info

    # --------------------------------------------------------------------- API
    def reset(self, opponents, seed=0):
        """Start a new game.  ``opponents``: 3 Botzone-protocol objects for seats
        1-3.  Returns ``(obs (38,4,9) float32, legal_mask (235,) bool)`` at the
        learner's first decision (seat 0's opening draw)."""
        assert len(opponents) == 3, "need exactly 3 opponents for seats 1-3"
        self.opponents = list(opponents)
        wall = build_wall(seed)
        self._gen = self._run(wall, seed)
        obs, mask = next(self._gen)                       # -> seat 0's first draw
        self._last = (obs, mask)
        return obs, mask

    def step(self, action):
        """Apply ``action`` (int in [0,235)) at the current decision.  Returns
        ``(obs, legal_mask, reward, done, info)``.  Mid-game reward is 0.0 (or
        ``-reward_shaping``); terminal reward is the seat-0 MCR score / scale."""
        try:
            obs, mask = self._gen.send(action)
            self._last = (obs, mask)
            return obs, mask, -self.reward_shaping, False, {}
        except StopIteration as e:
            reward, info = e.value
            obs, mask = self._last
            return obs, mask, reward, True, info

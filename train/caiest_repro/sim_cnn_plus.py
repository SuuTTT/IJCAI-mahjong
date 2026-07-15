"""
sim_cnn_plus.py — subclass of sim_cnn.Sim that supports PER-SEAT feature flavor.

Each seat may use either the base (38,4,9) caiest obs (moyu) or an AUGMENTED obs
(38+P planes) for a candidate trained on extra mechanism planes. We keep the base
CaiAgent shadow-agents EXACTLY as the parent sim does (so all game logic / claims /
masks are byte-identical), and we APPEND the extra planes (derived from each seat's
own base obs via featplus) only for seats flagged 'plus'.

This is the no-op-trap guard: moyu seats are fed 38 planes, candidate seats are fed
38+P planes computed by featplus — a real behavioral channel, verified by the gate's
calibration (cand==moyu, plus on both flavors -> must reproduce +0.000/g) and by the
md5/behavioral-diff check in run_gate.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim_cnn import Sim as _BaseSim
import featplus

class SimPlus(_BaseSim):
    def __init__(self, *a, seat_sets=None, **k):
        # seat_sets: list of 4 strings, '' for base-38 seats, e.g. 'A' / 'AC' / 'B' for plus seats
        super().__init__(*a, **k)
        self.seat_sets = seat_sets if seat_sets is not None else ['', '', '', '']

    def _obs_mask(self, seat):
        # base obs/mask from the parent (identical game logic)
        obs, mask = super()._obs_mask(seat)
        sset = self.seat_sets[seat]
        if not sset:
            return obs, mask
        extra = featplus.add_planes_one(obs.astype(np.float32), sset)  # (P,4,9)
        obs = np.concatenate([obs.astype(np.float32), extra], axis=0)  # (38+P,4,9)
        return obs, mask

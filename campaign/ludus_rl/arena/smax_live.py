"""Interactive SMAX session: human commands the ally squad, RTS-style.

Sticky orders per ally (move-to-point / attack-unit / stop) are translated
each tick into SMAX discrete actions (0=+y 1=+x 2=-y 3=-x 4=stop 5+j=attack
enemy j — verified empirically). The heuristic enemy plays the other side.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

sys.path.insert(0, "/root/JaxMARL/baselines/MAPPO")


class SmaxLive:
    def __init__(self, map_name: str = "2s3z", seed: int = 0):
        import jax
        from jaxmarl import make
        from jaxmarl.environments.smax import map_name_to_scenario
        from jaxmarl.wrappers.baselines import SMAXLogWrapper

        self._jax = jax
        self.env = SMAXLogWrapper(make(
            "HeuristicEnemySMAX", scenario=map_name_to_scenario(map_name),
            see_enemy_actions=True, walls_cause_death=True,
            attack_mode="closest"))
        self.map_name = map_name
        self.key = jax.random.PRNGKey(seed)
        self.obs, self.state = self.env.reset(self.key)
        self.n_allies = len(self.env.agents)
        self.n_units = len(np.asarray(self._st().unit_health))
        self.orders: dict[int, dict] = {i: {"kind": "stop"} for i in range(self.n_allies)}
        self.prev_cd: Optional[np.ndarray] = None
        self.prev_atk: Optional[np.ndarray] = None
        self.done = False
        self.won = False
        self.t = 0

    def _st(self):
        st = self.state
        for attr in ("env_state", "state"):
            st = getattr(st, attr, st)
        return st

    def set_order(self, ids, kind, x=None, y=None, target=None):
        for i in ids:
            if 0 <= int(i) < self.n_allies:
                self.orders[int(i)] = {"kind": kind, "x": x, "y": y,
                                       "target": target}

    def _action_for(self, i, pos, alive, avail):
        o = self.orders[i]
        if not alive[i]:
            return int(np.argmax(avail[i]))
        if o["kind"] == "attack" and o.get("target") is not None:
            j = int(o["target"]) - self.n_allies      # enemy list index
            if 0 <= j < self.n_units - self.n_allies:
                if not alive[self.n_allies + j]:      # target dead: clear
                    self.orders[i] = {"kind": "stop"}
                    return 4
                if avail[i][5 + j]:                   # in range: fire
                    return 5 + j
                tx, ty = pos[self.n_allies + j]       # chase
                return self._move_toward(pos[i], tx, ty, avail[i])
        if o["kind"] == "amove":
            # attack-move: engage anything in range, else advance to the point
            shots = np.where(avail[i][5:] > 0)[0]
            if len(shots):
                tgts = [self.n_allies + j for j in shots]
                dists = [np.hypot(*(pos[t] - pos[i])) for t in tgts]
                return 5 + int(shots[int(np.argmin(dists))])
            dx, dy = o["x"] - pos[i][0], o["y"] - pos[i][1]
            if abs(dx) + abs(dy) < 0.8:
                return 4 if avail[i][4] else int(np.argmax(avail[i]))
            return self._move_toward(pos[i], o["x"], o["y"], avail[i])
        if o["kind"] == "move":
            dx, dy = o["x"] - pos[i][0], o["y"] - pos[i][1]
            if abs(dx) + abs(dy) < 0.8:
                self.orders[i] = {"kind": "stop"}
                return 4
            return self._move_toward(pos[i], o["x"], o["y"], avail[i])
        return 4 if avail[i][4] else int(np.argmax(avail[i]))

    @staticmethod
    def _move_toward(p, tx, ty, av):
        dx, dy = tx - p[0], ty - p[1]
        prefs = ([1, 3] if dx > 0 else [3, 1]) if abs(dx) > abs(dy) else \
                ([0, 2] if dy > 0 else [2, 0])
        order = ([prefs[0]] + ([0 if dy > 0 else 2] if abs(dx) > abs(dy)
                 else [1 if dx > 0 else 3]) + [4])
        for a in order:
            if av[a]:
                return a
        return int(np.argmax(av))

    def tick(self) -> dict:
        if self.done:
            return self.frame()
        st = self._st()
        pos = np.asarray(st.unit_positions)
        alive = np.asarray(st.unit_alive)
        avail_d = self.env.get_avail_actions(
            getattr(self.state, "env_state", self.state))
        avail = np.stack([np.asarray(avail_d[a]) for a in self.env.agents])
        self.prev_cd = np.asarray(st.unit_weapon_cooldowns).copy()
        actions = {a: self._action_for(i, pos, alive, avail)
                   for i, a in enumerate(self.env.agents)}
        self.key, ks = self._jax.random.split(self.key)
        self.obs, self.state, rew, dones, info = self.env.step(
            ks, self.state, actions)
        self.t += 1
        if bool(dones["__all__"]):
            self.done = True
            self.won = bool(np.asarray(
                info["returned_won_episode"]).ravel()[0] > 0)
        return self.frame()

    def frame(self) -> dict:
        st = self._st()
        pos = np.asarray(st.unit_positions)
        hp = np.asarray(st.unit_health)
        alive = np.asarray(st.unit_alive)
        teams = np.asarray(st.unit_teams)
        types = np.asarray(st.unit_types)
        cd = np.asarray(st.unit_weapon_cooldowns)
        atk = np.asarray(st.prev_attack_actions)
        shots = []
        if self.prev_cd is not None and not self.done:
            for i in range(self.n_units):
                if alive[i] and cd[i] > self.prev_cd[i] and atk[i] > 0:
                    tgt = int(atk[i])
                    if tgt < self.n_units:
                        shots.append({"from": i, "to": tgt})
        units = [{"id": i, "x": float(pos[i][0]), "y": float(pos[i][1]),
                  "hp": float(hp[i]), "alive": int(alive[i]),
                  "team": int(teams[i]), "type": int(types[i]),
                  "cd": float(cd[i])}
                 for i in range(self.n_units)]
        return {"type": "frame", "t": self.t, "units": units, "shots": shots,
                "done": self.done, "won": self.won,
                "orders": {str(k): v["kind"] for k, v in self.orders.items()}}

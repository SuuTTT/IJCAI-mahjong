"""Interactive Warpath session: human = p0, scripted commander = p1."""

from __future__ import annotations

from collections import deque


def jnp_choice(f):
    # bot vehicle pick: mostly tanks, some artillery once it has a factory
    return 5 if (f["tick"] // 50) % 4 == 0 else 3

import jax.numpy as jnp
import numpy as np

from warpath import engine, render

NOOP = [0, 0, 0]


def _pack(x_tile: int, y_tile: int) -> int:
    return int(x_tile) * 64 + int(y_tile)


class WarpathBot:
    """Beatable v1 commander: power -> barracks -> rifle stream -> waves."""

    def __init__(self):
        self._pend = None
        self._pend_t = 0
        self._pend_xy = (34, 15)

    # bot sees frames as p0-centric owner ints; its own units are owner==1
    def act(self, f: dict) -> list[int]:
        credits = f["credits"][1]
        my_b = {b["type"] for b in f["buildings"] if b["owner"] == 1}
        fighters = [u for u in f["units"] if u["owner"] == 1 and u["type"] >= 1]
        harvesters = [u for u in f["units"] if u["owner"] == 1 and u["type"] == 0]
        # bot's own pending state lives in p1's slot — render only exposes p0,
        # so track it locally
        if self._pend is not None:
            self._pend_t -= 1
            if self._pend_t <= 0:                  # ready -> place near base
                bx, by = self._pend_xy
                self._pend = None
                return [6, 0, _pack(bx, by)]
            return NOOP
        def start(bt, x, y, tt):
            self._pend = bt; self._pend_t = tt; self._pend_xy = (x, y)
            return [1, bt, 0]
        if 1 not in my_b and credits >= 800:
            return start(1, 36, 13, 50)
        if 8 not in my_b and credits >= 1500:              # refinery -> free harvester + economy
            return start(8, 33, 17, 60)
        if 2 not in my_b and credits >= 500:
            return start(2, 36, 17, 40)
        if len(harvesters) < 3 and 8 in my_b and credits >= 1400:
            return [2, 0, 0]
        if 3 not in my_b and credits >= 2000:
            return start(3, 32, 13, 90)
        if 6 not in my_b and 2 in my_b and credits >= 600:
            return start(6, 33, 15, 40)            # a gun turret to defend
        if f["prod"][1][0] == -1 and credits >= 200 and 2 in my_b:
            return [2, 1, 0]
        if 3 in my_b and f["prod"][1][1] == -1 and credits >= 700:
            return [2, jnp_choice(f), 0]
        if len(fighters) >= 7 and f["tick"] % 100 == 0:
            return [3, 0, _pack(5, 15)]
        return NOOP


import jax


class WarpathLive:
    MAX_CMDS = 8                                  # commands per player per tick

    def __init__(self):
        # CPU end-to-end: the sim is tiny and the GPU belongs to the league —
        # sharing it queued every device sync behind training kernels (the
        # user-reported 5s command latency). On CPU a tick is sub-millisecond.
        self._cpu = jax.devices("cpu")[0]
        with jax.default_device(self._cpu):
            self.state = engine.reset()
        self.human = deque()
        self.bot = WarpathBot()
        self.done = False
        self.log = []                     # per-tick (a0_list, a1_list) for replay
        self._saved = False
        self._step = jax.jit(engine.step, backend="cpu")
        warm = jnp.zeros((2, self.MAX_CMDS, 3), jnp.int32)
        with jax.default_device(self._cpu):
            self._step(self.state, warm).tick.block_until_ready()

    def command(self, msg: dict):
        k = msg.get("kind")
        if k == "build":                          # start sidebar construction
            self.human.append([1, int(msg["btype"]), 0])
        elif k == "train":
            self.human.append([2, int(msg["utype"]), 0])
        elif k == "amove":
            self.human.append([3, 0, _pack(msg["x"], msg["y"])])
        elif k == "order":                        # per-unit orders (selection)
            for uid in msg.get("ids", [])[:24]:
                self.human.append([5, int(uid), _pack(msg["x"], msg["y"])])
        elif k == "attack":                       # force-attack a specific entity
            enc = engine.MAX_U + int(msg["tgt"]) if int(msg.get("tb", 0)) \
                else int(msg["tgt"])
            for uid in msg.get("ids", [])[:24]:
                self.human.append([8, int(uid), enc])
        elif k == "place":                        # place the ready pending building
            self.human.append([6, 0, _pack(msg["x"], msg["y"])])
        elif k == "cancel":                       # cancel the pending construction
            self.human.append([7, 0, 0])
        elif k == "rally":
            self.human.append([4, 0, _pack(msg["x"], msg["y"])])

    def tick(self) -> dict:
        if self.done:
            return render.frame(self.state)
        f = render.frame(self.state)
        a0s = [self.human.popleft() for _ in range(min(len(self.human), self.MAX_CMDS))]
        a0s += [NOOP] * (self.MAX_CMDS - len(a0s))
        a1s = [self.bot.act(f)] + [NOOP] * (self.MAX_CMDS - 1)
        self.log.append([a0s, a1s])
        with jax.default_device(self._cpu):
            acts = jnp.asarray(np.array([a0s, a1s], dtype=np.int32))
            self.state = self._step(self.state, acts)
        out = render.frame(self.state)
        if out["result"] != -1:
            self.done = True
            self.save_replay(out["result"])
        return out

    def save_replay(self, result):
        if self._saved:
            return
        self._saved = True
        import json, time
        from pathlib import Path
        d = Path("/root/ludus_replays_warpath")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"wp_{int(time.time())}.json").write_text(json.dumps({
            "env_version": engine.ENV_VERSION, "result": int(result),
            "ticks": len(self.log), "action_log": self.log,
        }))

"""Vanilla CFR (with regret-matching+) for Leduc, to produce a near-equilibrium
reference (coherent teacher) strategy. Saves the average strategy to JSON."""
import argparse, json, time
import leduc as L
import exploit as E

DEALS = E.DEALS
DEAL_PROB = E.DEAL_PROB


class CFR:
    def __init__(self, plus=True):
        self.regret = {}   # infoset -> [3] regret sums
        self.strat_sum = {}  # infoset -> [3] strategy sums
        self.plus = plus

    def _strategy(self, key, legal):
        r = self.regret.get(key)
        if r is None:
            r = [0.0, 0.0, 0.0]
            self.regret[key] = r
            self.strat_sum[key] = [0.0, 0.0, 0.0]
        pos = [max(r[a], 0.0) if a in legal else 0.0 for a in range(3)]
        tot = sum(pos)
        strat = [0.0, 0.0, 0.0]
        if tot > 0:
            for a in legal:
                strat[a] = pos[a] / tot
        else:
            for a in legal:
                strat[a] = 1.0 / len(legal)
        return strat

    def walk(self, s, p0r, p1r, cr):
        if s.kind == "terminal":
            return L.payoff_p0(s)
        if s.kind == "chance_pub":
            v = 0.0
            for _, prob, ns in L.public_children(s):
                v += prob * self.walk(ns, p0r, p1r, cr * prob)
            return v
        key = L.infoset_key(s)
        legal = L.legal_actions(s)
        strat = self._strategy(key, legal)
        util = [0.0, 0.0, 0.0]
        node = 0.0
        for a in legal:
            ns = L.apply_action(s, a)
            if s.to_act == 0:
                v = self.walk(ns, p0r * strat[a], p1r, cr)
            else:
                v = self.walk(ns, p0r, p1r * strat[a], cr)
            util[a] = v
            node += strat[a] * v
        r = self.regret[key]
        ss = self.strat_sum[key]
        w = getattr(self, "_it", 1)   # linear averaging weight
        if s.to_act == 0:
            cf = cr * p1r
            for a in legal:
                r[a] += cf * (util[a] - node)
                if self.plus and r[a] < 0:
                    r[a] = 0.0
                ss[a] += w * p0r * strat[a]
        else:
            cf = cr * p0r
            for a in legal:
                r[a] += cf * ((-util[a]) - (-node))
                if self.plus and r[a] < 0:
                    r[a] = 0.0
                ss[a] += w * p1r * strat[a]
        return node

    def iterate(self, iters, verbose=True):
        self._it = 0
        for it in range(iters):
            self._it = it + 1          # linear averaging weight (CFR+)
            for a, b in DEALS:
                s0 = L.new_game(L.DECK[a], L.DECK[b])
                self.walk(s0, 1.0, 1.0, DEAL_PROB)
            if verbose and (it + 1) % max(1, iters // 5) == 0:
                strat = self.average_strategy()
                expl = E.exploitability(strat)
                print(f"  iter {it+1}/{iters} exploitability={expl:.5f}", flush=True)

    def average_strategy(self):
        out = {}
        for key, ss in self.strat_sum.items():
            legal = [a for a in range(3) if ss[a] > 0 or True]
            tot = sum(ss)
            if tot > 0:
                out[key] = [ss[a] / tot for a in range(3)]
            else:
                # fall back to uniform over legal from regret keys presence
                out[key] = [0.0, 0.0, 0.0]
        return out


def build_reference(iters=3000, seed=0, plus=True, verbose=True):
    cfr = CFR(plus=plus)
    cfr.iterate(iters, verbose=verbose)
    strat = cfr.average_strategy()
    return strat


def strat_to_json(strat):
    # keys are tuples -> encode as strings
    out = {}
    for k, v in strat.items():
        priv, pub, rnd, r0, r1 = k
        ks = f"{priv}|{pub}|{rnd}|{','.join(map(str,r0))}|{','.join(map(str,r1))}"
        out[ks] = [round(x, 8) for x in v]
    return out


def json_to_strat(d):
    out = {}
    for ks, v in d.items():
        priv, pub, rnd, r0, r1 = ks.split("|")
        r0t = tuple(int(x) for x in r0.split(",")) if r0 else ()
        r1t = tuple(int(x) for x in r1.split(",")) if r1 else ()
        out[(int(priv), int(pub), int(rnd), r0t, r1t)] = v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", default="reference_strategy.json")
    a = ap.parse_args()
    t0 = time.time()
    print(f"[cfr] running {a.iters} iterations...", flush=True)
    strat = build_reference(a.iters)
    expl = E.exploitability(strat)
    gv = E.game_value_p0(strat)
    print(f"[cfr] final exploitability={expl:.6f} game_value_p0={gv:.5f} "
          f"n_infosets={len(strat)} secs={time.time()-t0:.1f}", flush=True)
    with open(a.out, "w") as f:
        json.dump({"exploitability": expl, "game_value_p0": gv,
                   "iters": a.iters, "n_infosets": len(strat),
                   "strategy": strat_to_json(strat)}, f, indent=2)
    print(f"[cfr] wrote {a.out}", flush=True)

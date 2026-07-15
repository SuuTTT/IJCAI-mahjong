"""T3 -- frozen opponent-pool evaluation harness (shared reference field).

Scores ANY Mahjong agent against an IDENTICAL, never-updated 3-bot reference
field so cross-seed / cross-pilot RL checkpoints are directly comparable.  The
candidate always sits at seat 0 (East); the reference bot fills seats 1-3.  Games
run through ``mahjong.match.run_match`` over the oracle-validated ``MyEngine`` on
deterministic seeded walls, so the field is fully reproducible.

The frozen reference field (fixed forever)
------------------------------------------
1. ``kdens3``       -- the 3-net fp16 ensemble champion
                       (``mahjong.champion.ChampionBot``, default npzs).
2. ``EfficiencyBot``-- shanten/efficiency heuristic ("Medium",
                       ``mahjong.bots.EfficiencyBot``).
3. ``random-legal`` -- ``mahjong.bots.RandomLegalBot``.

Metrics (mirror the T2 names), per reference, from the candidate's seat-0 view:
    win_rate    fraction of games won by the candidate
    mean_score  mean raw MCR score of seat 0
    zimo_rate   fraction of games the candidate self-drew a win
    dealin_rate fraction of games the candidate discarded the winning tile (ron)
    draw_rate   fraction of exhaustive draws (huang)

Public API
----------
    eval_vs_pool(make_agent, n_games=200, seed0=0) -> {ref_name: {metrics...}}
        ``make_agent`` is a ZERO-ARG callable returning a FRESH candidate agent;
        a new one is built per game so ChampionBot subprocesses get ``.close()``d
        between games (no subprocess leak).

CLI
---
    python -m baselines.mahjong_pool_eval --candidate {champion|hard|efficiency|random} \
        [--n 200] [--seed 0]
        -> per-reference table + JSON blob for one candidate vs the pool.
    python -m baselines.mahjong_pool_eval --round-robin [--rr-n 15] [--seed 0]
        -> ranks all 4 built-in bots (random/efficiency/hard/champion) vs the
           pool (the shared reference field).  kdens3 tops it, random bottom.

Candidate names:  champion = kdens3 3-net ensemble; hard = single-net champion
(weaker tier); efficiency / random = the heuristic / random-legal bots.  A T2
torch checkpoint is loadable as ``--candidate t2:/path/to/ckpt.pt`` (CPU
inference via ``baselines.mahjong_torch_model`` + ``rl_env.PolicyAgent``).

Runtime note: champion games are ~30-50s each (numpy ensemble inference in a
subprocess), and EVERY matchup vs the kdens3 reference spawns 3 champion
subprocesses, so keep ``--n`` small (e.g. 30-50) for champion-involved runs and
use a small ``--rr-n`` for the round-robin.  CPU-only harness (the champion runs
its own numpy inference in its subprocess; no GPU is touched here).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/root/ludus")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MCR_CHAMPION_DIR", "/root/mcr_champion")

# ChampionBot spawns its numpy ensemble as ``subprocess.Popen(["python3", ...])``,
# i.e. it relies on whatever ``python3`` is first on PATH.  If we are running
# under a venv (the usual case: ``/root/venv/bin/python -m ...``) but the venv is
# not "activated", PATH still points at the system ``/usr/bin/python3``, which
# lacks ``MahjongGB`` -> the champion subprocess dies on import and ChampionBot
# silently falls back to passive (never-winning) play.  Prepend THIS interpreter's
# bin dir so the champion subprocess runs the same Python that imported us (which
# has MahjongGB).  Environment-only; ``mahjong/champion.py`` is left untouched.
_INTERP_BIN = os.path.dirname(os.path.abspath(sys.executable))
if _INTERP_BIN and _INTERP_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _INTERP_BIN + os.pathsep + os.environ.get("PATH", "")

from mahjong.engine import build_wall
from mahjong.match import run_match
from mahjong.bots import EfficiencyBot, RandomLegalBot
from mahjong.champion import ChampionBot

# The frozen reference field -- these three, in this order, forever.
REFERENCE_NAMES = ["kdens3", "EfficiencyBot", "random-legal"]
_METRIC_KEYS = ["win_rate", "mean_score", "zimo_rate", "dealin_rate", "draw_rate"]


# ----------------------------------------------------------------- bot factories
# The frozen SL anchor (the policy the T2-JAX fine-tune STARTED from). As a
# reference it lets us measure the headline A/B: fine-tuned (seat 0) vs a field of
# 3 SL anchors, on the SAME flax forward -> a clean same-architecture comparison.
SL_ANCHOR = os.environ.get("MCR_SL_ANCHOR", "/root/mcr_champion/kdens_s0_fp16.npz")


def _make_reference(name, seed, seat):
    """Fresh reference-bot instance for one seat of one game."""
    if name == "kdens3":
        return ChampionBot()                       # default 3-net kdens3 ensemble
    if name == "EfficiencyBot":
        return EfficiencyBot(seed=seed * 10 + seat)
    if name == "random-legal":
        return RandomLegalBot(seed=seed * 10 + seat)
    if name == "anchor":                           # frozen SL anchor (JAX, CPU)
        from baselines.mahjong_t2jax_agent import make_agent_factory
        return make_agent_factory(SL_ANCHOR)()
    if name.startswith("t2jax:"):                  # any JAX ckpt as an opponent
        from baselines.mahjong_t2jax_agent import make_agent_factory
        return make_agent_factory(name[6:])()
    raise ValueError("unknown reference %r" % name)


def _t2_factory(ckpt_path):
    """Zero-arg factory for a T2 torch policy (CPU masked-argmax) as an agent.

    Loads a state_dict (raw SL ``.pkl`` or a T2 ``ckpt.pt`` carrying a ``policy``
    key) ONCE into a shared BN-free ``PolicyNet`` and wraps a fresh
    ``rl_env.PolicyAgent`` per game (so the FeatureDriver state resets)."""
    import numpy as np
    import torch
    from baselines.mahjong_torch_model import PolicyNet
    from mahjong.rl_env import PolicyAgent

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "policy" in sd:
        sd = sd["policy"]
    net = PolicyNet()
    net.load_state_dict(sd, strict=True)
    net.eval()
    NEG = -1e9

    def act(obs, mask):
        with torch.no_grad():
            logits = net(torch.from_numpy(obs[None]).float()).numpy()[0]
        return int(np.where(mask, logits, NEG).argmax())

    return lambda: PolicyAgent(act)


def candidate_factory(kind):
    """Zero-arg agent factory for a CLI candidate name."""
    if kind == "champion":
        return lambda: ChampionBot()                              # kdens3 3-net
    if kind == "hard":
        return lambda: ChampionBot(npzs="kdens_s0_fp16.npz")      # single-net tier
    if kind == "efficiency":
        # deterministic per-call seed so the field is reproducible
        c = [0]

        def mk():
            c[0] += 1
            return EfficiencyBot(seed=c[0])
        return mk
    if kind == "random":
        c = [0]

        def mk():
            c[0] += 1
            return RandomLegalBot(seed=c[0])
        return mk
    if kind.startswith("t2:"):
        return _t2_factory(kind[3:])
    if kind.startswith("t2jax:"):
        # JAX-T2 flax policy checkpoint (.msgpack) or SL-anchor .npz, CPU masked
        # argmax over the champion feature encoder. See mahjong_t2jax_agent.
        from baselines.mahjong_t2jax_agent import make_agent_factory
        return make_agent_factory(kind[6:])
    if kind == "anchor":                           # the frozen SL anchor itself
        from baselines.mahjong_t2jax_agent import make_agent_factory
        return make_agent_factory(SL_ANCHOR)
    raise ValueError("unknown candidate %r" % kind)


# The 4 built-in bots the round-robin ranks (candidate-name -> label).
BUILTIN_CANDIDATES = [
    ("champion", "kdens3"),
    ("hard", "hard (1-net)"),
    ("efficiency", "EfficiencyBot"),
    ("random", "random-legal"),
]


# --------------------------------------------------------------------- metrics
def _close_all(agents):
    for a in agents:
        close = getattr(a, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _game_outcome(res):
    """Seat-0 outcome flags from one ``run_match`` result.

    Returns (score0, win, zimo, dealin, draw).  ``dealin`` = seat 0 discarded the
    winning tile on a ron; the discarder is the most-negative-scoring seat on a
    ron (winner is positive), mirroring T2's parse_terminal derivation."""
    scores = res["scores"]
    ending = res["ending"]
    winner = res["winner"]
    score0 = scores[0]
    if ending == "draw":
        return score0, False, False, False, True
    win = (winner == 0)
    zimo = win and (ending == "zimo")
    dealin = False
    if ending == "rong":
        discarder = min(range(4), key=lambda s: scores[s])   # most-negative payer
        dealin = (discarder == 0)
    return score0, win, zimo, dealin, False


def _summarize(rows, n):
    n = max(1, n)
    return {
        "n_games": len(rows),
        "win_rate": round(sum(r[1] for r in rows) / n, 4),
        "mean_score": round(sum(r[0] for r in rows) / n, 3),
        "zimo_rate": round(sum(r[2] for r in rows) / n, 4),
        "dealin_rate": round(sum(r[3] for r in rows) / n, 4),
        "draw_rate": round(sum(r[4] for r in rows) / n, 4),
    }


def eval_vs_pool(make_agent, n_games=200, seed0=0, references=None, progress=False):
    """Score a candidate (seat 0) against each frozen reference (seats 1-3).

    ``make_agent`` -> fresh candidate each game.  Returns
    ``{reference_name: {n_games, win_rate, mean_score, zimo_rate, dealin_rate,
    draw_rate}}``.  Same seeded walls are reused across references so differences
    reflect the opponents, not luck."""
    references = references or REFERENCE_NAMES
    out = {}
    for ref in references:
        rows = []
        t0 = time.time()
        for i in range(n_games):
            seed = seed0 + i
            wall = build_wall(seed)
            cand = make_agent()
            refs = [_make_reference(ref, seed, s) for s in (1, 2, 3)]
            agents = [cand] + refs
            try:
                res = run_match(agents, wall, quan=0, srand=seed)
                rows.append(_game_outcome(res))
            finally:
                _close_all(agents)
            if progress and (i + 1) % 5 == 0:
                dt = time.time() - t0
                print("    [%s] %d/%d  %.1fs (%.1fs/game)"
                      % (ref, i + 1, n_games, dt, dt / (i + 1)), file=sys.stderr,
                      flush=True)
        out[ref] = _summarize(rows, n_games)
        out[ref]["_rows"] = rows          # kept for round-robin aggregation
    return out


# ----------------------------------------------------------------- presentation
def _print_pool_table(title, per_ref):
    print("\n" + title)
    hdr = "  %-16s %5s  %8s  %10s  %8s  %10s  %8s" % (
        "reference", "n", "win_rate", "mean_score", "zimo", "dealin", "draw")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ref in REFERENCE_NAMES:
        if ref not in per_ref:
            continue
        m = per_ref[ref]
        print("  %-16s %5d  %8.3f  %10.3f  %8.3f  %10.3f  %8.3f" % (
            ref, m["n_games"], m["win_rate"], m["mean_score"],
            m["zimo_rate"], m["dealin_rate"], m["draw_rate"]))


def _clean(per_ref):
    """Strip internal ``_rows`` before JSON emission."""
    return {ref: {k: v for k, v in m.items() if not k.startswith("_")}
            for ref, m in per_ref.items()}


# --------------------------------------------------------------------- CLI runs
def run_candidate(kind, n, seed):
    label = kind
    print("=" * 68)
    print("T3 pool eval: candidate=%s  n=%d/reference  seed0=%d" % (kind, n, seed))
    print("frozen reference field: " + ", ".join(REFERENCE_NAMES))
    print("=" * 68)
    t0 = time.time()
    per_ref = eval_vs_pool(candidate_factory(kind), n_games=n, seed0=seed,
                           progress=True)
    dt = time.time() - t0
    _print_pool_table("candidate '%s' vs the frozen pool:" % label, per_ref)
    print("\n[runtime] %.1fs (%d games total)" % (dt, n * len(REFERENCE_NAMES)))
    blob = {"candidate": kind, "n_games_per_ref": n, "seed0": seed,
            "runtime_s": round(dt, 1), "per_reference": _clean(per_ref)}
    print("\nJSON:")
    print(json.dumps(blob, indent=2))
    return blob


def run_round_robin(n, seed):
    print("=" * 68)
    print("T3 ROUND-ROBIN reference ranking: 4 built-in bots vs the frozen pool")
    print("n=%d games / (candidate x reference)  seed0=%d" % (n, seed))
    print("pool = " + ", ".join(REFERENCE_NAMES))
    print("=" * 68)
    t0 = time.time()
    table = []
    detail = {}
    for kind, label in BUILTIN_CANDIDATES:
        print("\n>> candidate: %s" % label, file=sys.stderr, flush=True)
        per_ref = eval_vs_pool(candidate_factory(kind), n_games=n, seed0=seed,
                               progress=True)
        detail[label] = _clean(per_ref)
        # aggregate across all references = performance vs the pool (the field)
        all_rows = [r for ref in REFERENCE_NAMES for r in per_ref[ref]["_rows"]]
        agg = _summarize(all_rows, len(all_rows))
        _print_pool_table("  %s vs pool (per reference):" % label, per_ref)
        table.append((label, agg))

    table.sort(key=lambda x: (x[1]["mean_score"], x[1]["win_rate"]), reverse=True)
    dt = time.time() - t0
    print("\n" + "=" * 68)
    print("REFERENCE FIELD RANKING (all 4 bots vs the 3-bot pool, aggregated)")
    print("n=%d games/(candidate x reference); ranked by mean_score, then win_rate"
          % n)
    print("=" * 68)
    hdr = "  %-4s %-16s %6s  %8s  %10s  %8s  %10s  %8s" % (
        "rank", "bot", "n", "win_rate", "mean_score", "zimo", "dealin", "draw")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, (label, m) in enumerate(table, 1):
        print("  %-4d %-16s %6d  %8.3f  %10.3f  %8.3f  %10.3f  %8.3f" % (
            i, label, m["n_games"], m["win_rate"], m["mean_score"],
            m["zimo_rate"], m["dealin_rate"], m["draw_rate"]))
    print("\n[runtime] %.1fs total" % dt)
    blob = {"mode": "round_robin", "n_per_pair": n, "seed0": seed,
            "runtime_s": round(dt, 1),
            "ranking": [{"rank": i, "bot": label, **m}
                        for i, (label, m) in enumerate(table, 1)],
            "detail": detail}
    print("\nJSON:")
    print(json.dumps(blob, indent=2))
    return blob


def main():
    ap = argparse.ArgumentParser(description="T3 frozen-pool eval harness")
    ap.add_argument("--candidate", type=str, default=None,
                    help="champion|hard|efficiency|random|t2:<ckpt.pt>")
    ap.add_argument("--n", type=int, default=200,
                    help="games per reference for a single-candidate run")
    ap.add_argument("--round-robin", action="store_true",
                    help="rank the 4 built-in bots vs the pool")
    ap.add_argument("--rr-n", type=int, default=15,
                    help="games per (candidate x reference) in round-robin")
    ap.add_argument("--seed", type=int, default=0, help="seed0 for the walls")
    a = ap.parse_args()

    if a.round_robin:
        run_round_robin(a.rr_n, a.seed)
    elif a.candidate:
        run_candidate(a.candidate, a.n, a.seed)
    else:
        ap.error("give --candidate <name> or --round-robin")


if __name__ == "__main__":
    main()

"""Strength evaluation of a T2-JAX Mahjong checkpoint (headline: RL vs SL anchor).

Answers "is the KL-leashed PPO fine-tune STRONGER than the SL anchor it started
from, and where does it rank vs the reference pool?" for ONE checkpoint snapshot,
then appends a single JSONL line the web monitor can chart.

For a fair same-architecture A/B it measures BOTH:
  * ``vs_anchor_winrate``  -- the FINE-TUNED policy (seat 0) vs a field of 3 SL
    anchors, and
  * ``anchor_self_winrate`` -- the SL ANCHOR (seat 0) vs 3 SL anchors (the
    seat-0 / East baseline for an *unchanged* policy),
on the identical flax forward and the SAME seeded walls, so a lift of
``vs_anchor_winrate`` over ``anchor_self_winrate`` is real strength, not seat bias.

Plus cheap pool matchups (EfficiencyBot, random-legal) and a SMALL, explicitly
counted kdens3-ensemble sample (~50s/game).  Every matchup logs its own game
count -- nothing is silently truncated.

CPU-only (``JAX_PLATFORMS=cpu``).  Writes one line to ``--out-jsonl`` and, if this
is the best ``vs_anchor_winrate`` seen, copies the snapshot to ``--best-ckpt`` and
records ``best_by_eval.json``.

Usage (the eval loop calls this; ts_unix + trainer env_steps injected from bash):
    JAX_PLATFORMS=cpu python -m baselines.mahjong_t2jax_strength \
        --ckpt /path/snap.msgpack --ts 1720000000 --env-steps 557056000 \
        --n-anchor 40 --n-anchor-self 30 --n-eff 30 --n-rand 30 --n-kdens3 6 \
        --out-jsonl /root/ludus_train/mahjong_t2_jax/strength_eval.jsonl \
        --best-ckpt /root/ludus_train/mahjong_t2_jax/best_by_eval.msgpack
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/root/ludus")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MCR_CHAMPION_DIR", "/root/mcr_champion")

from baselines.mahjong_pool_eval import eval_vs_pool, candidate_factory, SL_ANCHOR
from baselines.mahjong_t2jax_agent import load_meta


def _one(make_agent, ref, n, seed, progress):
    """Run n games of candidate vs a single reference; return its metric dict."""
    if n <= 0:
        return {"n_games": 0, "win_rate": None, "mean_score": None,
                "dealin_rate": None}
    out = eval_vs_pool(make_agent, n_games=n, seed0=seed, references=[ref],
                       progress=progress)
    return out[ref]


def evaluate(ckpt, anchor, n_anchor, n_anchor_self, n_eff, n_rand, n_kdens3,
             seed, progress):
    cand = candidate_factory("t2jax:" + ckpt)          # fine-tuned policy
    anch = candidate_factory("t2jax:" + anchor)        # SL anchor (control)

    print("[strength] vs anchor (fine-tuned seat0 vs 3 SL anchors) ...", flush=True)
    m_anchor = _one(cand, "anchor", n_anchor, seed, progress)
    print("[strength] anchor-self baseline (SL anchor seat0 vs 3 SL anchors) ...",
          flush=True)
    m_self = _one(anch, "anchor", n_anchor_self, seed, progress)
    print("[strength] vs EfficiencyBot ...", flush=True)
    m_eff = _one(cand, "EfficiencyBot", n_eff, seed, progress)
    print("[strength] vs random-legal ...", flush=True)
    m_rand = _one(cand, "random-legal", n_rand, seed, progress)
    print("[strength] vs kdens3 (small, ~50s/game) ...", flush=True)
    m_kd = _one(cand, "kdens3", n_kdens3, seed, progress)

    meta = load_meta(ckpt)
    rec = {
        "ckpt_env_steps": meta.get("env_steps"),
        "ckpt_update": meta.get("update"),
        "vs_anchor_winrate": m_anchor["win_rate"],
        "vs_anchor_score": m_anchor["mean_score"],
        "vs_anchor_dealin": m_anchor["dealin_rate"],
        "anchor_self_winrate": m_self["win_rate"],
        "anchor_self_score": m_self["mean_score"],
        "vs_efficiency_winrate": m_eff["win_rate"],
        "vs_random_winrate": m_rand["win_rate"],
        "vs_kdens3_winrate": m_kd["win_rate"],
        "n_vs_anchor": m_anchor["n_games"],
        "n_anchor_self": m_self["n_games"],
        "n_vs_efficiency": m_eff["n_games"],
        "n_vs_random": m_rand["n_games"],
        "n_vs_kdens3": m_kd["n_games"],
    }
    rec["games"] = (rec["n_vs_anchor"] + rec["n_anchor_self"] +
                    rec["n_vs_efficiency"] + rec["n_vs_random"] + rec["n_vs_kdens3"])
    return rec


def _update_best(out_jsonl, ckpt, best_ckpt):
    """After appending, copy the snapshot to best_ckpt if it has the best
    vs_anchor_winrate seen so far (tie-break vs_anchor_score)."""
    if not best_ckpt:
        return None
    rows = []
    with open(out_jsonl) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    scored = [r for r in rows if isinstance(r.get("vs_anchor_winrate"), (int, float))]
    if not scored:
        return None
    best = max(scored, key=lambda r: (r["vs_anchor_winrate"],
                                      r.get("vs_anchor_score") or -1e9))
    # is the just-written record (last row) the best?
    last = rows[-1]
    if last is best or (last.get("vs_anchor_winrate") == best.get("vs_anchor_winrate")
                        and last.get("ts_unix") == best.get("ts_unix")):
        try:
            shutil.copyfile(ckpt, best_ckpt)     # snapshot -> best_by_eval.msgpack
        except OSError as e:
            # the jsonl line is already written; never let a copy hiccup crash the eval
            print("[strength] WARN best-ckpt copy failed: %r" % e, flush=True)
            return None
        info = {"env_steps": last.get("env_steps"),
                "ckpt_env_steps": last.get("ckpt_env_steps"),
                "vs_anchor_winrate": last.get("vs_anchor_winrate"),
                "vs_anchor_score": last.get("vs_anchor_score"),
                "anchor_self_winrate": last.get("anchor_self_winrate"),
                "ts_unix": last.get("ts_unix"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.splitext(best_ckpt)[0] + ".json", "w") as f:
            json.dump(info, f, indent=2)
        return info
    return None


def main():
    ap = argparse.ArgumentParser(description="T2-JAX strength eval (RL vs SL anchor)")
    ap.add_argument("--ckpt", required=True, help="checkpoint snapshot (.msgpack)")
    ap.add_argument("--anchor", default=SL_ANCHOR, help="SL anchor (.npz)")
    ap.add_argument("--ts", type=int, default=0, help="ts_unix from `date +%s` (shell)")
    ap.add_argument("--env-steps", type=int, default=0,
                    help="env_steps from the trainer's latest jsonl line (shell)")
    ap.add_argument("--n-anchor", type=int, default=40)
    ap.add_argument("--n-anchor-self", type=int, default=30)
    ap.add_argument("--n-eff", type=int, default=30)
    ap.add_argument("--n-rand", type=int, default=30)
    ap.add_argument("--n-kdens3", type=int, default=6)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--best-ckpt", default=None)
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    rec = evaluate(a.ckpt, a.anchor, a.n_anchor, a.n_anchor_self, a.n_eff,
                   a.n_rand, a.n_kdens3, a.seed, a.progress)
    rec["ts_unix"] = a.ts or int(time.time())
    rec["env_steps"] = a.env_steps or rec.get("ckpt_env_steps")
    rec["eval_runtime_s"] = round(time.time() - t0, 1)
    line = json.dumps(rec)
    print("\n[strength] RECORD: " + line, flush=True)
    if a.out_jsonl:
        with open(a.out_jsonl, "a") as f:
            f.write(line + "\n")
        best = _update_best(a.out_jsonl, a.ckpt, a.best_ckpt)
        if best:
            print("[strength] NEW BEST by vs_anchor_winrate -> %s (env_steps=%s, "
                  "winrate=%s)" % (a.best_ckpt, best.get("env_steps"),
                                   best.get("vs_anchor_winrate")), flush=True)


if __name__ == "__main__":
    main()

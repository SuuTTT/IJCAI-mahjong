"""Ladder core: mirrored-pair round-robin, OpenSkill ratings with CI, standings
with integrity + mechanism metrics, freeze-on-drift (docs/04)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from arena.ladder import duel
from arena.ladder.agents import REGISTRY, get_agent
from arena.ladder.provenance import provenance

FROZEN = "LADDER_FROZEN"
# two-sided Student-t 97.5% quantiles, df 1..30 (block-level CIs, docs/04 §3)
_T = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def tcrit(df: int) -> float:
    ks = sorted(_T)
    return _T[max(k for k in ks if k <= max(df, 1))]


class Ladder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.matches_f = self.root / "matches.jsonl"

    # ---------------- freeze discipline ----------------
    def frozen(self) -> str | None:
        p = self.root / FROZEN
        return p.read_text() if p.exists() else None

    def freeze(self, reason: str):
        (self.root / FROZEN).write_text(json.dumps(
            {"reason": reason, "t": time.time(), **provenance()}, indent=2))

    def unfreeze(self):
        (self.root / FROZEN).unlink(missing_ok=True)

    def _require_live(self):
        if (why := self.frozen()) is not None:
            raise SystemExit(f"LADDER IS FROZEN — fix and `calibrate --unfreeze`.\n{why}")

    # ---------------- match log ----------------
    def record_block(self, a: str, b: str, block: dict, block_key: int):
        _, ha = get_agent(a)
        _, hb = get_agent(b)
        rec = {"match_id": f"{a}--{b}--k{block_key}",
               "a": a, "b": b, "block_key": block_key,
               "submission_hashes": {a: ha, b: hb},
               **block, **provenance(), "t": time.time()}
        with self.matches_f.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def matches(self) -> list[dict]:
        if not self.matches_f.exists():
            return []
        return [json.loads(l) for l in self.matches_f.read_text().splitlines() if l]

    # ---------------- rounds ----------------
    def run_round(self, entrants: list[str], pairs_per_block: int, round_idx: int):
        self._require_live()
        acts = {n: get_agent(n)[0] for n in entrants}
        for i, a in enumerate(entrants):
            for b in entrants[i + 1:]:
                bk = round_idx * 1000 + i * 40 + entrants.index(b)
                seeds = np.arange(bk * 10000, bk * 10000 + pairs_per_block,
                                  dtype=np.uint32)
                block = duel.run_block(acts[a], acts[b], seeds, block_key=bk)
                if block["illegal"] > 0:
                    self.freeze(f"illegal actions in {a} vs {b} block {bk}")
                    raise SystemExit("froze: illegal actions during rated play")
                self.record_block(a, b, block, bk)

    # ---------------- ratings ----------------
    def ratings(self):
        from openskill.models import PlackettLuce
        model = PlackettLuce()
        rs = {}
        stats = {}
        for m in self.matches():
            a, b = m["a"], m["b"]
            for n in (a, b):
                rs.setdefault(n, model.rating(name=n))
                stats.setdefault(n, {"pairs": 0, "illegal": 0, "pair_score_sum": 0.0})
            for s in m["pair_scores"]:
                ranks = [1, 2] if s > 0.5 else ([2, 1] if s < 0.5 else [1, 1])
                [[rs[a]], [rs[b]]] = model.rate([[rs[a]], [rs[b]]], ranks=ranks)
            stats[a]["pairs"] += len(m["pair_scores"])
            stats[b]["pairs"] += len(m["pair_scores"])
            stats[a]["pair_score_sum"] += sum(m["pair_scores"])
            stats[b]["pair_score_sum"] += len(m["pair_scores"]) - sum(m["pair_scores"])
            stats[a]["illegal"] += m["illegal"]   # judge-side total for the block
            stats[b]["illegal"] += m["illegal"]
        return rs, stats

    def standings(self) -> dict:
        rs, stats = self.ratings()
        rows = []
        for n, r in rs.items():
            rows.append({
                "agent": n, "mu": round(r.mu, 3), "sigma": round(r.sigma, 3),
                "ci_low": round(r.mu - 2 * r.sigma, 3),
                "ci_high": round(r.mu + 2 * r.sigma, 3),
                "pairs": stats[n]["pairs"],
                "mean_pair_score": round(stats[n]["pair_score_sum"]
                                         / max(stats[n]["pairs"], 1), 4),
                "illegal_in_blocks": stats[n]["illegal"],
            })
        rows.sort(key=lambda r: -r["mu"])
        # CI separation between adjacent rows: the promotion criterion (docs/04 §3)
        for i in range(len(rows) - 1):
            rows[i]["separated_from_next"] = rows[i]["ci_low"] > rows[i + 1]["ci_high"]
        n_matches = len(self.matches())
        out = {"standings": rows, "frozen": self.frozen() is not None,
               **provenance(),
               "integrity": {"match_blocks": n_matches,
                             "entrants": len(rows),
                             "complete": n_matches > 0 and len(rows) >= 2},
               "t": time.time()}
        (self.root / "standings.json").write_text(json.dumps(out, indent=2))
        return out


def head2head(a: str, b: str, blocks: int = 8, pairs_per_block: int = 16,
              seed0: int = 900000) -> dict:
    """Fresh paired blocks -> block-level Student-t CI on A's pair score.
    Verdict SEPARATED only if the CI excludes 0.5 (docs/04 §3)."""
    act_a, ha = get_agent(a)
    act_b, hb = get_agent(b)
    means = []
    illegal = 0
    for bi in range(blocks):
        seeds = np.arange(seed0 + bi * pairs_per_block,
                          seed0 + (bi + 1) * pairs_per_block, dtype=np.uint32)
        blk = duel.run_block(act_a, act_b, seeds, block_key=seed0 + bi)
        means.append(float(np.mean(blk["pair_scores"])))
        illegal += blk["illegal"]
    m = float(np.mean(means))
    sd = float(np.std(means, ddof=1)) if blocks > 1 else 0.0
    half = tcrit(blocks - 1) * sd / np.sqrt(blocks) if blocks > 1 else float("inf")
    lo, hi = m - half, m + half
    verdict = "SEPARATED" if (lo > 0.5 or hi < 0.5) else "TIE"
    assert len(means) == blocks, "loud-fail: missing blocks"
    return {"a": a, "b": b, "submission_hashes": {a: ha, b: hb},
            "blocks": blocks, "pairs_per_block": pairs_per_block,
            "block_means": [round(x, 4) for x in means],
            "a_pair_score_mean": round(m, 4),
            "t_ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict,
            "illegal": illegal, **provenance(),
            "integrity": {"expected_blocks": blocks, "actual": len(means),
                          "complete": True}}


def audit(ladder: Ladder, match_id: str) -> dict:
    """Replay-verify a recorded block: re-run it from (seeds, agents, block_key)
    and demand bit-identical game results (determinism contract)."""
    rec = next((m for m in ladder.matches() if m["match_id"] == match_id), None)
    if rec is None:
        raise SystemExit(f"no such match block: {match_id}")
    act_a, ha = get_agent(rec["a"])
    act_b, hb = get_agent(rec["b"])
    fresh = duel.run_block(act_a, act_b,
                           np.asarray(rec["seeds"], np.uint32), rec["block_key"])
    same = (fresh["game1_results"] == rec["game1_results"]
            and fresh["game2_results"] == rec["game2_results"])
    hashes_match = {rec["a"]: ha, rec["b"]: hb} == rec["submission_hashes"]
    return {"match_id": match_id, "replay_identical": same,
            "submission_hashes_match": hashes_match,
            "verdict": "PASS" if same and hashes_match else "FAIL",
            **provenance()}

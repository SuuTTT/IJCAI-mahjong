"""Calibration job (docs/04 §1): self-tests + anchor drift check; drift or a
failed self-test FREEZES the ladder. `--inject-bug` proves the tripwire works by
mutating one card stat in a subprocess and demanding a freeze."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from arena.ladder import duel
from arena.ladder.agents import get_agent
from arena.ladder.ladder import Ladder
from arena.ladder.provenance import REPO, provenance

ANCHORS = ["random_v0", "rule_v0", "ppo_weak_v0", "ppo_v0"]
DRIFT_TOL = 0.08          # max |Δ mean pair score| vs reference
REFERENCE = REPO / "benchmarks/results/ladder_reference.json"


def measure(pairs_per_block: int = 32, seed0: int = 777000) -> dict:
    """Self-test every anchor, then measure every anchor pairing (mirrored pairs)."""
    acts = {n: get_agent(n)[0] for n in ANCHORS}
    self_tests = {}
    for n in ANCHORS:
        self_tests[n] = duel.self_test_exact_tie(acts[n], n_pairs=8,
                                                 seed0=seed0 + hash(n) % 1000)
    pairings = {}
    for i, a in enumerate(ANCHORS):
        for b in ANCHORS[i + 1:]:
            seeds = np.arange(seed0 + i * 5000, seed0 + i * 5000 + pairs_per_block,
                              dtype=np.uint32)
            blk = duel.run_block(acts[a], acts[b], seeds, block_key=seed0 + i)
            pairings[f"{a}|{b}"] = {
                "mean_pair_score": round(float(np.mean(blk["pair_scores"])), 4),
                "illegal": blk["illegal"]}
    return {"self_tests": self_tests, "pairings": pairings,
            "pairs_per_block": pairs_per_block, **provenance(), "t": time.time()}


def calibrate(ladder: Ladder, set_reference: bool = False) -> dict:
    cur = measure()
    report = {"measured": cur, "checks": {}}

    st_ok = all(v["pass"] for v in cur["self_tests"].values())
    report["checks"]["self_tests_exact"] = st_ok
    if not st_ok:
        ladder.freeze(f"calibration self-test failed: {cur['self_tests']}")
        report["frozen"] = True
        return report

    if set_reference or not REFERENCE.exists():
        REFERENCE.write_text(json.dumps(cur, indent=2))
        report["checks"]["reference"] = "written"
        return report

    ref = json.loads(REFERENCE.read_text())
    drifts = {}
    for k, v in cur["pairings"].items():
        if k in ref["pairings"]:
            drifts[k] = round(abs(v["mean_pair_score"]
                                  - ref["pairings"][k]["mean_pair_score"]), 4)
    worst = max(drifts.values()) if drifts else 0.0
    judge_changed = cur["judge_hash"] != ref["judge_hash"]
    report["checks"]["drifts"] = drifts
    report["checks"]["worst_drift"] = worst
    report["checks"]["judge_hash_matches_reference"] = not judge_changed
    if worst > DRIFT_TOL or judge_changed:
        why = (f"anchor drift {worst} > {DRIFT_TOL}" if worst > DRIFT_TOL
               else "judge hash changed vs reference (engine/judge code differs)")
        ladder.freeze(f"calibration: {why}; drifts={drifts}")
        report["frozen"] = True
    else:
        report["frozen"] = False
    return report


def inject_bug_drill(scratch: str) -> dict:
    """Prove the tripwire: run calibrate in a subprocess with one card stat mutated
    (Bulwark hp doubled). The scratch ladder MUST end up frozen."""
    env = {"BOOM_CARDS_PATCH": "0:hp:3876", "PATH": "/usr/bin:/bin"}
    import os
    env.update({k: v for k, v in os.environ.items()
                if k.startswith(("CUDA", "XLA", "LD_", "VIRTUAL", "JAX"))})
    code = (f"import sys; sys.path.insert(0, '{REPO}');"
            "from arena.ladder.ladder import Ladder;"
            "from arena.ladder.calibrate import calibrate;"
            f"r = calibrate(Ladder('{scratch}'));"
            "import json; print(json.dumps({'frozen': r['frozen']}))")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, cwd=REPO)
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "{}"
    frozen = json.loads(last).get("frozen", False)
    marker = Path(scratch) / "LADDER_FROZEN"
    return {"drill": "mutate Bulwark hp 1938->3876 in subprocess",
            "subprocess_reported_frozen": frozen,
            "frozen_marker_written": marker.exists(),
            "pass": bool(frozen and marker.exists()),
            "stderr_tail": out.stderr[-400:]}

"""Shared provenance helper (docs/04 §4): every ladder output embeds enough to
re-run and verify it — env version, judge-code hash, submission hashes, seeds."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from boom import ENV_VERSION

REPO = Path(__file__).resolve().parents[2]

# everything whose change invalidates comparability of results
_JUDGE_SOURCES = [
    "boom/engine.py", "boom/cards.py", "boom/cards.csv", "boom/vec.py",
    "baselines/rule_bot.py", "baselines/eval_pair.py", "baselines/ppo_selfplay.py",
    "arena/ladder/duel.py", "arena/ladder/agents.py",
]


def judge_hash() -> str:
    h = hashlib.sha256()
    for rel in _JUDGE_SOURCES:
        p = REPO / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    patch = os.environ.get("BOOM_CARDS_PATCH", "")
    if patch:                       # a patched engine is a DIFFERENT judge
        h.update(f"PATCHED:{patch}".encode())
    return h.hexdigest()[:16]


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def provenance(**extra) -> dict:
    p = {"env_version": ENV_VERSION, "judge_hash": judge_hash(),
         "commit": commit()}
    if os.environ.get("BOOM_CARDS_PATCH"):
        p["cards_patch"] = os.environ["BOOM_CARDS_PATCH"]
    p.update(extra)
    return p

"""Ladder agent registry: named entrants -> act functions + submission hashes.

Built-ins hash to the judge sources (their code IS the judge tree); weights
submissions hash their parameter file."""

from __future__ import annotations

from pathlib import Path

from arena.ladder.provenance import REPO, file_hash, judge_hash
from baselines.eval_pair import make_agent

REGISTRY: dict[str, str] = {
    "random_v0": "random",
    "rule_v0": "rule",
    "ppo_weak_v0": f"ppo:{REPO}/baselines/checkpoints/ppo_v5/params_weak.msgpack",
    "ppo_v0": f"ppo:{REPO}/baselines/checkpoints/ppo_v5/params_latest.msgpack",
}


def get_agent(name: str):
    """-> (act_fn, submission_hash). act_fn(key, states, seat, tick) -> (N,3)."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise SystemExit(f"unknown agent '{name}' (known: {list(REGISTRY)})")
    act, _ = make_agent(spec)
    if spec.startswith("ppo:"):
        sub_hash = file_hash(spec[4:])
    else:
        sub_hash = f"builtin@{judge_hash()}"
    return act, sub_hash


def submit(name: str, params_path: str) -> str:
    """Register a weights-only submission under `name` (session-local registry;
    the persistent multi-user registry is P1 platform work)."""
    p = Path(params_path)
    if not p.exists():
        raise SystemExit(f"no such params file: {p}")
    REGISTRY[name] = f"ppo:{p.resolve()}"
    return file_hash(p)

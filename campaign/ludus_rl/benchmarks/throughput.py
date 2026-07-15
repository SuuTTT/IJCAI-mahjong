"""Throughput benchmark: env-steps/s and matches/s vs batch size.

Usage:
    python benchmarks/throughput.py --batches 512,2048,8192 --ticks 300 \
        --out benchmarks/results/throughput_v1.json

Writes a loud-fail JSON artifact with hardware + commit provenance (AGENTS.md §3/§4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax

from boom import ENV_VERSION, vec


def bench_one(batch: int, ticks: int, reps: int = 3) -> dict:
    key = jax.random.PRNGKey(0)
    # compile + warmup (excluded from timing)
    out = vec.rollout_random_jit(key, batch, ticks, None, False)
    jax.block_until_ready(out)
    times = []
    for r in range(reps):
        t0 = time.perf_counter()
        out = vec.rollout_random_jit(jax.random.PRNGKey(r + 1), batch, ticks, None, False)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    best = min(times)
    steps = batch * ticks
    return {
        "batch": batch, "ticks": ticks, "reps": reps,
        "seconds_best": best, "seconds_all": times,
        "env_steps_per_s": steps / best,
        "matches_per_s_at_900_ticks": steps / best / 900.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="512,2048,8192")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--out", default="benchmarks/results/throughput_v1.json")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    results = []
    for b in batches:
        r = bench_one(b, args.ticks)
        print(f"batch={b:>6}  {r['env_steps_per_s']:.3e} env-steps/s  "
              f"{r['matches_per_s_at_900_ticks']:.1f} matches/s", flush=True)
        results.append(r)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        # exclude the artifact dir itself: this script's own outputs must not
        # count as a dirty worktree
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":!benchmarks/results"],
            text=True).strip())
    except Exception:
        commit, dirty = "unknown", True

    dev = jax.devices()[0]
    artifact = {
        "env_version": ENV_VERSION,
        "device": str(dev), "platform": dev.platform,
        "device_kind": getattr(dev, "device_kind", "unknown"),
        "jax_version": jax.__version__,
        "commit": commit, "dirty_worktree": dirty,
        "results": results,
        "integrity": {"expected_batches": len(batches), "actual": len(results),
                      "complete": len(results) == len(batches)},
    }
    assert artifact["integrity"]["complete"], "partial benchmark must not be written silently"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

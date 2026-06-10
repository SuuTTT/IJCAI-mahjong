# JAX CSM env — high-throughput self-play foundation (R1)

Vectorized JAX environment for Chinese Standard Mahjong: run thousands of games on GPU to escape
the CPU self-play wall and enable scaled self-play RL (the path to exceeding the imitation ceiling /
the journal work). See `docs/RESEARCH_ROADMAP.md` R1.

## Throughput — thesis VALIDATED (2026-06-10, single RTX 3060, Phase-1 env + small conv policy)
| batch | env-steps/s | full-games/s (~72 steps/game) |
|---|---:|---:|
| 256   | 424k  | ~5.9k |
| 4096  | 1.38M | ~19k  |
| 16384 | 1.53M | ~21k  |

**vs ~246 games/s on the old CPU self-play probe → ~85× on ONE mid-tier GPU.** With the full
mechanics (win/claims/fan) this will slow some, but even a 5–10× hit leaves ~10× over CPU; on 4–8
rented 3090/4090s, billion-game self-play (Suphx-scale) is days, not years. This is what makes
scaled self-play RL feasible — and what runs the GPUs at sustained 100% (unlike the bursty distills).

## Build status (honest)
- [x] **Phase 1** (`csm_env.py`, `bench.py`): state arrays, reset/deal, draw→discard round-robin
      core, obs encoding, fixed action space, jit+vmap rollout, throughput benchmark. Reward is a
      placeholder (wall-exhaustion draw) — validates steps/s, NOT yet trainable.
- [ ] **Phase 2 — win detection (agari/shanten):** the algorithmic core. Reward = win/lose. This is
      the next milestone and makes self-play trainable for win-rate.
- [ ] **Phase 3 — claim mechanics (Chi/Peng/Gang/Hu priority):** turn order stops being round-robin;
      the trickiest transition (a discard can be claimed out of order).
- [ ] **Phase 4 — 81-fan scoring:** the multi-week part; gives the true duplicate-score reward.
      Likely incremental (major fans first) or a learned/approximate scorer to start.

## Run
```bash
pip install "jax[cuda12]"
python3 bench.py            # batch sweep
python3 bench.py 16384 72   # full-game-length rollout at B=16384
```

## How this plugs into the win-path
Phase 2 → self-play RL with a small net (the `fast8`-style nets we distilled) → distill the
RL-improved policy back into the deploy net, OR use it as the rollout/value net for the R2 search
(which already beats the teacher at test time). The small-net + JAX-env combo is the only thing that
makes either RL-at-scale or deep PIMC search affordable.

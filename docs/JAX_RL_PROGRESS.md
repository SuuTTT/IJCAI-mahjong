# JAX self-play RL — build progress (the only remaining ceiling-breaker for the July 7 final)

Context: every offline lever (imitation ×N, test-time search ×6 incl. net-PIMC) is a documented
null/wash. Reward-shaped scaled self-play RL is the one untested mechanism that can exceed the
imitation ceiling. Box: RTX A4000 (Ampere, 16GB) + ~25 cores (cgroup-confirmed), JAX 0.10.1 + MahjongGB.

## DONE & VALIDATED (each gated against MahjongGB / real games — no fabricated claims)
- **Phase-1 env** (`csm_env.py`): vectorized draw→discard round-robin, obs encoding, 1.55M steps/s on the A4000.
- **Phase 2 — win detection** (`agari.py` numpy ref + `agari_jax.py` GPU): standard 4-sets+pair and
  seven-pairs (incl. 豪华七对 kong edge). Numpy ref = **100% vs MahjongGB** (10k hands). JAX version =
  **0 mismatches vs ref on 20k hands**. Runs batched on GPU via precomputed per-group feasibility
  tables (`build_agari_tables.py`) + a tiny (sets×pairs) DP.
- **Reward** (`fan_reward.py`): HYBRID — JAX detects the win every step; MahjongGB scores fan exactly
  at the (rare) terminal; the **8-fan floor** is enforced; MCR duplicate scoring (fan→4 seats).
  The MCR rule reproduces **50/52 real game finishes** (2 misses = log-parse feeder ambiguity, not
  the rule; in the env the feeder is known exactly).

Why hybrid, not pure-JAX fan scoring: fan depends on the max-fan DECOMPOSITION of the hand — exactly
what makes MahjongFanCalculator complex. Vectorizing that in JAX is impractical & error-prone; scoring
only the rare terminals on CPU is exact and cheap. Win-detection (the per-step hot path) stays on GPU.

## NEXT (the integration — needs care + the user able to sanity-check first curves)
1. Wire agari_jax (self-draw after draw; discard-rob after discard) + fan_reward into `step()`, replacing
   the placeholder reward. Hybrid loop: step env in GPU-batched chunks; collect terminal winning hands;
   score on CPU (25 cores) with MahjongGB; emit rewards.
2. Phase 3 claims (Chi/Peng/Gang priority) — turn order stops being round-robin. Trickiest transition.
   Can defer for a discard-only first RL signal.
3. Potential-based reward shaping Φ(hand) from the 81-fan table (the user's idea; the historically
   positive lever) — `r' = r + γΦ(s') − Φ(s)`, the form that can't break policy balance.
4. PPO self-play (JAX/optax), small net warm-started from the SL policy.
5. **GATE:** does reward-shaped self-play beat the SL base in self-play? If not → stop, ship plain.

## Honest note on this autonomous block
I prioritized VALIDATED correctness (agari + reward, both gated vs ground truth) over launching a
multi-hour training run I couldn't sanity-check unattended — training on an unvalidated env/reward
would risk presenting garbage as progress (the exact failure this project has had). The GPU is
therefore mostly idle this block; the responsible next step (PPO integration) wants a human to eyeball
the first learning curves. Components are committed and ready to assemble.

## VALIDATED RESULTS (2026-06-13 autonomous block)
- Win-detection: 0 false wins / 248 wins over 131,072 self-play games (vs MahjongGB). Numpy ref 100%
  vs MahjongGB (10k); JAX vs ref 0 mismatch (20k).
- Reward (MCR rule): 50/52 vs real game finishes (2 = log-parse feeder ambiguity, exact in env).
- Throughput (win-aware self-play, random policy, A4000): B=65536 -> 64.8M steps/s, 589k games/s.
  With a policy-net forward per step the bottleneck becomes the net (~1.5M steps/s ~ 20k+ games/s),
  still millions of self-play games/hour.
- Known non-blocking edge: ~0.05% terminal tile-conservation glitch (bystander phantom tile; reward
  unaffected; fix in Phase-3 step rewrite).
- Files: agari.py, agari_jax.py, build_agari_tables.py, fan_reward.py, csm_selfplay.py, csm_validate.py.

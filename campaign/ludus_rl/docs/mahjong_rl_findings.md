# Mahjong (MCR) in JAX — RL environment, trainer, and findings

A practical reference for anyone who wants to **use the JAX Chinese-Standard-Mahjong
(MCR / 国标) environment**, **reproduce our RL experiments**, or **try their own RL
ideas** on it. It records what worked, what did *not*, and — most usefully — *why*,
so you don't have to rediscover the same walls.

For the detailed observation/action/reward spec see
[`docs/mahjong_rl_env.md`](mahjong_rl_env.md); this doc is the higher-level guide +
findings.

---

## TL;DR

- We built a **fully `vmap`pable, oracle-exact JAX MCR engine** (`mahjong/jax_env.py`).
  It reproduces a 12,288-game official judge oracle **byte-exact** and runs
  **~3M env-steps/s** on one GPU. This is the reusable asset.
- We built a **fully-on-GPU fused PPO trainer** (env + policy + PPO in one `lax.scan`,
  `baselines/mahjong_t2_jax.py`) — **~26× a CPU trainer**.
- **RL strength result is negative (so far):** KL-leashed self-play PPO fine-tuning
  of a strong distilled SL policy **does not reliably beat that SL policy**.
- We **diagnosed** the first bottleneck (the value function never converged) and
  **fixed it** (`mahjong_t2_jax_v2.py`, PopArt-lite return normalization → value loss
  dropped ~100×, verified). **It still did not beat the anchor.** So the barrier is
  deeper than critic engineering — see [Findings](#findings--lessons).
- Two **RL-safety bugs** in the env were found and fixed (illegal-Hu reward-hack,
  uncapped melds). Both have regression tests.

---

## 1. The environment (what you get)

`mahjong/jax_env.py` is a pure-functional, JIT/`vmap`-friendly MCR engine.

| property | value |
|---|---|
| Players | 4 (Chinese Standard / MCR, ≥8-fan-to-win) |
| State | fixed-size pytree (`NamedTuple` of int8/int32 tensors) |
| Step | `step(state, action) -> state` — pure, jittable, vmappable |
| Phase machine | `lax.switch` over {draw, discard-claim, gang-draw, bugang, terminal} |
| Claim resolution | branchless priority-matrix argmax (Hu > Peng/Gang > Chi) |
| Fan scoring | `jax.pure_callback` to the ground-truth `PyMahjongGB` at Hu only |
| Validation | **12,288/12,288** replay-equivalence vs the official oracle (221/221 golden re-verified independently) |
| Throughput | **~3.07M env-steps/s** (batch 4096, env-only, one RTX 3090) |

### Minimal use

```python
import jax, jax.numpy as jnp
from mahjong import jax_env as J
from mahjong.engine import build_wall

wall  = jnp.asarray(J.wall_to_codes(build_wall(seed=0)))
state = J.reset(wall, quan=0)                     # deal
mask  = J.legal_mask(state)                       # bool[4, 235] per-seat legality
obs   = J.obs(state, seat=0)                      # (38,4,9) float32, champion feature layout
state = J.step(state, actions)                    # actions int32[4,2] (235-code + follow-up discard)
```

Everything is batchable: `jax.vmap(J.reset)(walls, quans)`, `jax.jit(jax.vmap(J.step))`, and a
whole game with `J.run_game(wall, quan, key, action_fn, cap=300)` (a `lax.while_loop`).

### Observation / action / reward (summary)

- **obs**: `(38, 4, 9)` float32 — seat/prevalent wind, own hand (count-encoded), discards
  (4 planes × 4 players, self-relative), melds. Identical to the published SL agent's
  `feature.py`, so a policy trained here speaks the standard MCR interface.
- **action**: 235 discrete — `0` Pass, `1` Hu, `2–35` discard, `36–98` Chi, `99–132` Peng,
  `133–166` Gang, `167–200` AnGang, `201–234` BuGang. **Always mask with `legal_mask`.**
- **reward**: terminal, sparse, zero-sum. Seat-0 MCR score / 8 (zimo win ≈ +6, ron ≈ +4,
  deal-in ≈ −2, draw 0). The ≥8-fan win condition makes wins **genuinely rare** from scratch.

### RL-safety correctness gates (read this before you train)

Two hazards we hit and fixed — if you fork the env, keep them:

1. **Illegal-Hu forfeit** (`host_score`): a Hu below the 8-fan minimum is *forfeited*
   (declarer penalised, zero-sum), **not** paid. Without this an agent learns to spam
   action 1 (Hu) for a fan-0 "win". Closed the `fan=-1 → 3*(8+fan)` payout hole.
2. **Meld cap** (`legal_mask`): Chi/Peng/Gang/AnGang are gated on `meld_cnt < 4`. A 5th
   meld is structurally impossible and would OOB-write the `melds[4,4,3]` array and
   silently corrupt state.

Both are covered by `tests/test_jax_env.py` (`test_illegal_hu_forfeit`, `test_meld_cap`);
the oracle stays 221/221. General lesson: **a vmappable env has no exceptions to catch —
every illegal transition must be masked or made a no-op/penalty, or it silently corrupts
a fraction of your batch.**

---

## 2. The trainer (fully on-GPU fused PPO)

`baselines/mahjong_t2_jax.py` runs the **entire** rollout — vmapped `jax_env`, the policy
net, GAE, and PPO — inside `lax.scan` on the GPU. The only host touch is the terminal fan
callback (once per rollout, not per step).

- Policy: a BN-free fused ResNet (kdens-style), loaded from `kdens_s0_fp16.npz`; `0/300`
  argmax parity vs the reference NumPy policy in both fp32 and bf16.
- Objective: **KL-leashed** PPO fine-tune (trust-region rollback + adaptive β, KL ≤ 0.05),
  self-play vs frozen SL opponents, reward from `host_score` (so illegal Hu is forfeited).
- Throughput (per GPU): pure fused rollout **13k–16k game-steps/s** (bf16); full training
  loop **~6.5k–8k env-steps/s**. ~26× / ~42× a CPU trainer.

Run it:

```bash
python -m baselines.mahjong_t2_jax --seed 1 --B 256 --T 256 --dtype bf16 \
    --kl-target 0.05 --beta0 0.08 --eval-games 2000 --eval-every-updates 25 \
    --out /root/ludus_train/mahjong_t2_jax/ --resume
```

**VRAM sizing:** the full loop needs ~11 GB at `B=256` bf16, ~22 GB at `B=512`. On a 12 GB
card use `B=256`; on 24 GB use `B=512` (bigger batch amortises the PPO/optimizer overhead —
it's the main reason a 24 GB card out-throughputs a 12 GB one on the full loop even when the
12 GB card wins the pure rollout).

Strength is tracked out-of-band by a CPU eval loop (`baselines/mahjong_t2jax_strength.py`,
`deploy/run_mahjong_t2_jax_v2_eval.sh`) that plays the current checkpoint vs the SL anchor,
EfficiencyBot, random, and a small kdens sample, and appends a JSONL the web monitor charts
(`/train`).

---

## Findings & lessons

### F1 — The value function was the first bottleneck (and it's fixable)

The baseline fine-tune was **flat**: seat-0 win-rate ~0.164 for **590M env-steps**, never
pulling ahead of its frozen opponents. Diagnosis (read-only, from the training logs):

- `win_rate` dead flat; `mean_score` flat with high variance.
- **`vloss` flat at ~1.6–2.5 the entire run — the critic never fit the returns.**
- `entropy` flat, policy-gradient tiny, **KL only ~0.02 (the 0.05 leash was never binding)**.

So it was **advantage-signal-limited, not KL-limited**: a critic that can't predict returns
→ noisy GAE advantages → PPO gradient carries no information → the policy barely moves.
The culprit is the reward: terminal-only, and a single 8-fan Hu is +6 and up, so the value
target has enormous variance and a fresh net can't regress to it.

**Diagnostic lesson:** when RL is "flat," look at `vloss` and the *binding* constraint
before touching the policy. Here everyone's instinct is "loosen the KL leash" — but the KL
had 2.5× unused headroom; the critic was the problem.

### F2 — Return normalization fixes the critic (~100× lower vloss), verified

`baselines/mahjong_t2_jax_v2.py` adds **PopArt-lite return normalization**: an EMA of the
return mean/std (carried in the train state), the critic predicts a *normalized* value,
denormalized (`v*std+mean`) wherever it feeds GAE, with value target `(rt-mean)/std`; plus
`--value-epochs 6` (extra value-only steps). Result, verified live:

| | baseline critic | **v2 critic** |
|---|---|---|
| value loss | flat ~1.6–2.5 forever | **~0.016–0.02, monotonic (~100× lower)** |
| KL | ~0.02 (leashed) | ~0.02 (leashed) |

The critic now actually fits the returns. Toggle with `--popart 0/1` to reproduce either arm.

**Lesson:** for terminal, high-variance, sparse rewards (poker/mahjong-like), **normalize
the value target** (PopArt or a running-std). Normalizing only the *advantages* (as vanilla
PPO does) is not enough — the *critic's regression target* is what blows up.

### F3 — A converged critic did NOT produce a stronger agent (the real wall)

This is the important one. With the critic fixed, we trained v2 to ~136M steps and tracked
strength vs the SL anchor. **It still does not reliably beat the anchor:**

- `vs_anchor_winrate` oscillates **0.125–0.250** (anchor-self baseline ≈ **0.233**).
- `vs_anchor_score` swings **−7.9 to +2.3**; one eval (u1326) briefly hit +2.27/0.250 —
  but the neighbours are −2 to −8, so it's **noise**, not a durable crossing.
- It reliably beats EfficiencyBot (~0.70) and random (~0.73), and loses to the full kdens
  ensemble — i.e. it's a strong policy, just **not stronger than its own SL anchor**.

So fixing the diagnosed bottleneck was **necessary but not sufficient**. The remaining
barrier appears to be structural, not an engineering bug:

1. **Tight trust region around a near-optimal policy.** The SL anchor was distilled from a
   large game corpus and is already strong; a KL≤0.05 ball around it contains little room
   for a strictly better policy. (Note the KL sits at ~0.02 — the policy doesn't even *want*
   to move far; the gradient toward "better" is weak.)
2. **Weak self-play improvement signal.** Training seat-0 against *frozen copies of itself*
   means the opponents are as strong as the learner; there's no exploitable weakness to
   climb, so the reward gradient toward "beat them" is small and noisy.

**Lesson:** "diagnose the bottleneck, fix it, measure" is the right loop — but a *green
leading indicator (vloss) does not imply the terminal metric (strength) will move*. Always
gate on the real objective, and be ready for the fix to reveal a deeper cause rather than
solve the problem.

---

## What to try next (open directions)

Concrete experiments for a continuer, roughly in expected-value order. All plug into the
existing trainer; most are a flag or a small variant.

1. **Anneal / widen the trust region.** KL headroom is unused at 0.05; try KL 0.1–0.3 with
   a schedule (tight → loose) once the critic is warm. If strength still won't move, that
   *confirms* the anchor is near-optimal locally (a finding in itself).
2. **Exploiter / opponent-pool self-play** instead of a frozen-self field: keep a pool of
   past checkpoints + a "best-response" exploiter seat, so there's always a weaker opponent
   to climb. This is the single most likely lever to break F3.
3. **BC-warm-start then WIDE explore.** Start from the SL policy but with a *large* entropy
   bonus and *no* leash for a while, to escape the SL basin, then re-leash. Risky but it's
   how you'd actually exceed a distilled policy.
4. **Reward shaping toward tenpai/shanten** for a denser gradient between the rare ≥8-fan
   wins (the reward is currently terminal-only).
5. **Pretrained-critic warm-start.** `value_e2e_ckpt.pt` (in the checkpoints repo) is a
   critic already trained on the corpus; porting it (BN → flax) would start value learning
   from a good point rather than PopArt-from-scratch. Complementary to F2, not a substitute.
6. **Longer horizon / bigger batch** on a 24 GB card (`B=512`) — cheap to try, but given
   both arms are flat on strength, unlikely to be the missing piece alone.

If you try (2) or (3) and it works, that's the positive result this line has been missing —
please record it here.

---

## Reproduction

- **Repo:** `github.com/SuuTTT/ludus`, branch `fix/canhu-strict`. Key files:
  `mahjong/jax_env.py` (env), `baselines/mahjong_t2_jax.py` (trainer),
  `baselines/mahjong_t2_jax_v2.py` (PopArt critic fix),
  `baselines/mahjong_t2jax_strength.py` (strength eval), `tests/test_jax_env.py`.
- **Weights/testset (HuggingFace):** policy `kdens_s{0,1,2}_fp16.npz`, SL anchor
  `ckpt/kd/kd_128x40_s0.pkl`, pretrained critic `ckpt/value/value_e2e_ckpt.pt`; oracle
  testset `mcr_final2026_{golden.jsonl,full.jsonl.gz}` + `validate_engine.py`.
- **Deps:** `jax[cuda12]`, `flax`, `optax`, `PyMahjongGB` (the ground-truth fan calculator),
  `numpy`. jax 0.6.2 and 0.10.2 both work; `jax[cuda12]` bundles a compatible cuDNN.
- **Validate the env** anytime: `python tests/test_jax_env.py --golden <golden.jsonl>` →
  expect `REPLAY EQUIVALENCE: 221/221` plus the two regression tests.
- **Throughput bench:** `python baselines/mahjong_fused_rollout.py --mode bench
  --batches 512 --T 256 --dtype bf16`.

### Hardware note (measured)

| workload | RTX 4070 (12 GB) | RTX 3090 (24 GB) |
|---|---|---|
| pure fused rollout, B=512 bf16 | **15,795 game-steps/s** | 13,050 |
| full training loop | 6,530 (B=256, VRAM-capped) | ~8,000 (B=512) |

The newer card wins raw simulation; the 24 GB card wins end-to-end training via the larger
batch. Pick the GPU by which you're bottlenecked on.

---

*Status at time of writing: the PopArt critic fix is the live trainer; strength is still
oscillating around the SL anchor (F3 unresolved). This doc will be updated if a direction
above produces a durable crossing.*

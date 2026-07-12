---
layout: default
title: "Task Board for the Platform Team"
---

# Task Board for the Platform Team

**Date:** 2026-07-12
**To:** the Ludus platform (botzone) developer
**From:** the mahjong-campaign side
**What this is:** concrete, self-contained tasks you can execute **independently** —
every task lists the artifacts to pull, the exact spec, and the report format. Nothing
here needs access to our machines, and we don't need access to yours.

Context, if you want it: [Lab Notebook — every active track](blog/2026-07-12-lab-notebook-active-tracks.html)
(especially track 8, the RL pilot) and the
[Engine Validation Feedback](ENGINE_VALIDATION_FEEDBACK.html).

---

## Coordination protocol (read first)

- **Artifacts move via HuggingFace and GitHub only.** No shared credentials, no SSH
  into each other's boxes, no ad-hoc file transfers.
- **Results and questions move via GitHub issues or branches** — on
  [SuuTTT/ludus](https://github.com/SuuTTT/ludus) or
  [SuuTTT/IJCAI-mahjong](https://github.com/SuuTTT/IJCAI-mahjong), whichever fits.
- **If an artifact you need is missing, open an issue** — the campaign team uploads on
  request (usually same day).

---

## T1 — Merge `fix/canhu-strict` and correct the doc claim

**Priority: first — T2 depends on it. Effort: minutes.**

1. **Merge** branch `fix/canhu-strict` (commit `1a9b231`) into `master`. Tracking issue
   with full details: **[SuuTTT/ludus#1](https://github.com/SuuTTT/ludus/issues/1)**.
   Background: the official judge attaches a per-seat win-eligibility array `canHu[4]`
   to every display event; the engine never emitted it, so strict byte-exact replay
   read **0/12,288** while every rule-level check passed. The branch adds the field via
   the engine's existing fan path; nothing else changes. Details:
   [Engine Validation Feedback](ENGINE_VALIDATION_FEEDBACK.html).
2. **Re-verify post-merge:** run the stock `validate_engine.py` from
   [Dannibal/mcr-final2026-testset](https://huggingface.co/datasets/Dannibal/mcr-final2026-testset)
   in strict mode against `master`. Expected: **12,288/12,288** (already verified on
   the branch; this just confirms the merge).
3. **Update your doc claim.** `docs/mahjong_rl_env.md` claimed byte-exact replay before
   strict mode had ever been run (it read 0/12,288 at first execution). Restate it as:
   *strict byte-exact 12,288/12,288 as of commit `<merge-commit>`, verified with
   `validate_engine.py --strict` against the mcr-final2026-testset* — a claim with a
   command and a number attached. Recommended: wire the 221-game golden subset into CI
   so the claim stays true (see Platform Developer Guide §5.5, "Acceptance Suite A").

---

## T2 — Run a second RL seed on your 3090

**Why this matters.** Our RL pilot (lab-notebook track 8) runs one seed on an A4000.
This campaign has been burned repeatedly by n=1 results; a second seed, trained
independently on independent hardware from the same spec, makes any signal — or any
null — far harder to explain away. You already have everything needed: the verified
engine (post-T1), a 3090, and the artifacts below.

### 2.1 Environment

Your **own Ludus mahjong env**, post-T1 merge. The merge is a hard prerequisite:
`canHu` is the per-step win-legality oracle, and an RL action space without it either
recomputes Hu-legality independently (and can silently diverge) or lets illegal Hu be
representable.

### 2.2 Artifacts to pull

| What | Where | Notes |
|---|---|---|
| **Policy init** (SL student, KD seed 0) | HF [`Dannibal/ijcai-mahjong-ckpts-2026`](https://huggingface.co/Dannibal/ijcai-mahjong-ckpts-2026) → `ckpt/kd/kd_128x40_s0.pkl` (torch training ckpt) **or** `deploy/kdens_s0.npz` (fp32 numpy deploy form; fp16-storage twin `deploy/kdens_s0_fp16.npz`) | npz↔torch argmax parity is verified (0 flips / 300 states). Obs/action encoding: Platform Developer Guide §4.6 + `deploy/caiest_cnn/feature.py`. |
| **Frozen KL reference** | Same file, loaded a second time and frozen | The KL leash target (§2.3). |
| **Critic init** (e2e score-value head, r = 0.71 vs realized final score) | **Not yet on HF — the campaign team will upload on request.** Open an issue on SuuTTT/IJCAI-mahjong titled "upload value-head ckpt" and it goes to `Dannibal/ijcai-mahjong-ckpts-2026` (planned path `ckpt/value/`). | **Don't block on it:** PPO works with a freshly initialized critic head — it just wastes some early samples. Swap the init in when it lands if you haven't started. |

### 2.3 Training spec

- **Algorithm:** PPO fine-tune of the SL policy.
- **KL leash:** penalize KL(π_RL ‖ π_SL-frozen); keep **mean per-decision KL < 0.05**.
  If it drifts above, raise the KL coefficient. This is load-bearing: unleashed RL
  fine-tunes have *degraded* strong SL bases in this campaign, every time.
- **Reward:** terminal only — the learner-seat's **raw MCR game score / 8**. That is
  the final's actual ranking metric. No shaping, no placement proxy, no intermediate
  reward.
- **Opponents:** self-play vs **frozen copies** — the 3 opponent seats are served by
  periodic snapshots of the learner (e.g. keep the last 3–5 snapshots, refresh every
  few updates, sample uniformly). The learner must face a stable, competent field, not
  chase its own tail.
- **Inference:** **batched central GPU inference** over N parallel envs — one server
  batches forward passes for all seats/envs. Your own measurement motivates this:
  model inference ≈ **100×** the env-step cost, so the GPU scheduler, not the env, is
  the throughput lever.
- **Seed:** use PPO/env **seed = 1** (our pilot is seed 0), same policy init.
- **Checkpoints:** every **2 h wall-clock** — policy + critic + optimizer state.

### 2.4 Reporting

At every checkpoint, evaluate ≥ 2,000 games **vs the frozen SL policy** (a fixed
reference field, *not* the moving self-play pool) and append one JSON record:

```json
{"step": 123456, "KL": 0.031, "mean_score": 4.2, "zimo_rate": 0.081, "dealin_rate": 0.126}
```

- `step` — env steps (or PPO updates — say which, once, in the first record).
- `KL` — mean per-decision KL to the frozen SL policy over the eval games.
- `mean_score` — learner's mean per-game raw score in the eval games.
- `zimo_rate` / `dealin_rate` — fraction of eval games the learner wins by self-draw /
  deals into an opponent's win.

Deliver as an append-only JSONL **committed to a branch** (e.g. `rl-seed1-results` on
SuuTTT/ludus or SuuTTT/IJCAI-mahjong) **or posted in a GitHub issue thread** — either
is fine. Please include the exact KL coefficient and snapshot-refresh settings you
used, so the two seeds are comparable.

---

## T3 (optional) — Opponent-pool service

A fixed-opponent evaluation field for cross-checking both RL seeds against the *same*
yardstick:

- **kdens3** (frozen) — the 3-model ensemble, deployed per Platform Developer Guide §4
  (`deploy/kdens_s{0,1,2}_fp16.npz` from the HF ckpt repo).
- **EfficiencyBot** — your shanten-efficiency heuristic tier (the "Medium" rung of the
  guide's §6 difficulty ladder).
- **random-legal** — the sample-bot tier.

Serve them at **fixed, never-updated versions** behind any simple interface (the §3
bot protocol over stdio or a socket is fine). Value: both pilots can report
`mean_score` against an identical frozen field, which makes the two learning curves
directly comparable and catches self-play delusions (a learner that beats its own
snapshots but not the fixed field). If you stand this up, note the endpoint/invocation
in an issue and we'll point our eval at it too.

---

*Links: [Platform Developer Guide](PLATFORM_DEVELOPER_GUIDE.html) ·
[Engine Validation Feedback](ENGINE_VALIDATION_FEEDBACK.html) ·
[Lab Notebook: active tracks](blog/2026-07-12-lab-notebook-active-tracks.html) ·
[Project index](index.html)*

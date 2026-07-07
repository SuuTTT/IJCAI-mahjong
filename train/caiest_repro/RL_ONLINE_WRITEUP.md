# Online Self-Play RL (gate-matched PPO) — WRITEUP

**Goal.** Test the biggest untested SOTA lever for the MCR agent: **online self-play
policy-gradient RL** initialised from the imitation ceiling (`aug_128x40_s0`, enhanced
128x40 BC). Every OFFLINE lever failed (offline AWR nulled even with a good critic); we had
never run true online self-play policy-gradient — the mechanism behind Suphx's superhuman play.

## Design (matches the calibrated gate exactly)
- **Env** (`sim_cnn.Sim`, the duplicate-gate engine): ONE learner seat vs THREE **identical
  frozen opponents** — the same configuration the gate uses (candidate at 1 seat, ref at the
  other 3). Learner seat rotates over {0,1,2,3} per game (gate rotation). Reward = **per-deal
  placement** `5 - avg_rank` on the deal scores (the gate's own formula; self-play parity = 2.500).
  Verified with `rl_online_verify.py`: rewards valid, mean over 12 self-play deals = 2.458 (~2.5).
- **Algorithm** (`rl_online.py`): PPO (clip 0.2, 3 epochs) with a value baseline (256-unit head
  on the CNN trunk), **entropy bonus** (0.008), and a **KL-leash to frozen aug_s0** (β=0.4,
  decay 0.999, floor 0.05) so the competent BC policy cannot collapse. Init from the trainable
  BN checkpoint `ckpt/aug/aug_128x40_s0.bn.pkl`. LR 2e-5 (trust-region small). Advantage
  normalised per batch. CPU actors (24, 1 thread each) roll out; one GPU process does the update.
- **Self-play pool**: opponent = frozen aug_s0 with prob `p_anchor=0.6`, else a recent learner
  snapshot (past selves) — proper self-play. Snapshot every 25 iters; pool cap 20.

## Verification discipline
A gain counts ONLY if **CI-separated above aug_s0** in the calibrated gate
(aug_s0-vs-aug_s0 = **2.500**). Each snapshot is fused to `resbn_fused` (same 128x40, TLE-safe)
and gated by `e11_gate.py` at **6 blocks × 150 seeds** (3600 games) vs `ckpt/aug/aug_128x40_s0.pkl`.
`rl_online_agg.py` computes mean ± 1.96·SE across blocks; **CI-beats** iff `ci95_lo > 2.500`.
Results appended to `RL_ONLINE_RESULTS.json`.

## Infrastructure / durability
- Trainer: `rl_online.py` (tag `rlon`), snapshots → `ckpt/rl_online/snap_*.pkl`.
- Durable driver: `rl_online_harness.sh` launched via **setsid** (survives session drop) —
  runs the trainer, then gates each new snapshot at 6 blocks and aggregates. Honors
  `/root/STOP_RL`; 15GB soft / 12GB hard disk guard; coexists with the recipe sweep + arch
  experiment (24 CPU actors, 16 gate workers, one GPU used sparingly).
- Harvest: `RL_ONLINE_RESULTS.json` (per-snapshot CI verdicts), `/root/rl_online.train.log`
  (placement / KL / entropy trend), `/root/rl_online.harness.log`.

## Status
RUN IN PROGRESS. iter 1: placement 2.438, KL 0.012, entropy 0.495 (leash active, no collapse).
Verdict (does online self-play CI-beat aug_s0? which settings mattered? collapse/null failure
mode?) is filled in once gated snapshots accumulate in `RL_ONLINE_RESULTS.json`.

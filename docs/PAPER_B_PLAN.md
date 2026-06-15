# Paper B plan — a verified-SOTA MCR bot (top-1% ladder + beat Tjong)

Two-track program. **Paper A** = "The Evaluation Gap" (ToG, negative-results forensic study; rigorous
results done, polishing prose/citations). **Paper B** = a *positive* result: a Chinese-Standard-Mahjong bot
that reaches **top-1% on the Botzone ladder** and **beats the published SOTA (Tjong)** — and, crucially,
**verifies that claim under Paper A's noise-floor-aware protocol**. The two papers cite each other: A is the
method that makes B's claim trustworthy; B is the existence proof that the protocol still lets you *win*.

## Why B is credible now (what we learned)
- The imitation ceiling is real: distillation/cloning **ties** `lad_chunjiandu` (Paper A §5, rigorous error
  bars). So B cannot be "more SL." The only validated path above the ceiling is **RL**.
- We never implemented the **one published winning recipe**: Tjong's **SL-first → RL with "fan-backward"
  reward shaping** (top-1% Botzone). Our self-play RL failed partly on (a) a now-fixed scoring bug, (b) being
  compute-infeasible at the 40-block scale, and (c) no fan-backward. All three are fixable.
- We have the assets: 5.87M-decision SL corpus, `lad_chunjiandu` (strong warm-start), the strong field-bot
  datasets (league anchors), the JAX self-play env (589k games/s, validated), and the rigorous gauntlet +
  real-field collector.

## Method (the bet)
**Small-net SL→RL with fan-backward, warm-started from `lad_chunjiandu`, evaluated rigorously.**
1. Distill `lad_chunjiandu` → small policy (~64ch×3–6 blocks): ~100× cheaper forward → RL becomes feasible
   (the 40-block net was forward-bound at ~50 min/iter).
2. RL phase: self-play / PFSP league seeded with the strong field bots as anchors, **KL-leashed to the SL
   net**, with **fan-backward reward shaping** (propagate terminal fan backward; compare to potential-based
   shaping as an ablation — Paper-A-relevant).
3. Verify every gain with the **noise-floor-aware gauntlet** (σ≈366 ⇒ need ≥thousands of games or a clear
   >2σ margin) + **real-ladder A/B** (the only ground truth).

## Phased plan with decision gates
- **Phase 0 — De-risk (idle 3060, ~1 day).** Distill the small net; implement fan-backward; short RL run.
  GATE: does win8 climb above the warm-start baseline (53%) AND does the RL net beat the SL net in self-play
  by >2σ? If no → fan-backward doesn't unlock it here → fold into Paper A as another quantified null. If yes →
  Phase 1.
- **Phase 1 — Scale RL (1 clean JAX GPU + the 3060, ~3–5 days).** Larger self-play league, fan-backward,
  diverse anchors, periodic rigorous gauntlet vs `lad_chunjiandu` + strong field bots. GATE: a >2σ gauntlet
  win over `lad_chunjiandu` that survives replication.
- **Phase 2 — Ladder validation (deploy + collect, ~1 week wall-clock).** Ship the RL bot to Botzone, collect
  real games, measure rank. GATE: monotone ladder-rank improvement; target top-1%.
- **Phase 3 — Beat SOTA + write B.** Reproduce a Tjong-style baseline (transformer + fan-backward) for a
  head-to-head; claim SOTA under the rigorous protocol; write Paper B (cites A for the protocol).

## GPU plan (staged — do not over-provision)
- **Phase 0:** the **idle 3060** (have it). No new GPU.
- **Phase 1 (only if Phase 0 passes):** **+1 clean, JAX-only GPU** (A4000/3090-class, ≥16 GB) for the
  self-play env — keep torch (distill/eval) on the 3060 to avoid the torch+jax cuDNN conflict that bit us
  repeatedly. So **2 GPUs total**.
- **Phase 1 seed sweep / league diversity (optional):** +1–2 small GPUs (3060) for parallel seeds. **≤4 total.**
- Never more than ~4. RL seed sweeps can also run sequentially.

## Risks (honest)
- Per Paper A, RL has tied under feasible compute 16+ times. B bets the missing ingredients (fixed scorer +
  fan-backward + small-net feasibility + warm-start) are the difference — which is exactly what Tjong claims.
  Plausible, not guaranteed.
- If B's RL also ties under the σ≈366 floor, **that is a clean fallback**: it becomes the strongest possible
  null for Paper A (we implemented the published winning recipe and it still tied). Either outcome is publishable.
- Real-ladder validation is slow (collection-bound) and meta shifts weekly — Phase 2 needs patience.

## Track A in parallel (no GPU)
Polish: add the offline-RL (Levine 2020, Fujimoto BCQ), WILDS, and "Student of Games" citations; verify the
`[UNVERIFIED]` bib fields; tighten §III/§IV prose; figures. Submittable independent of B.

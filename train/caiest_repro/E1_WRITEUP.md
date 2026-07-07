# E1 — Do imitation-learned Mahjong agents over-claim vs expert play?

**Expert reference claim-rate** (leaders' real decisions, 11175 held-out claim-legal states): **0.2514**.  
Claim-rate = claims (chi/peng, action in [36,133)) / claim-legal states. All models trained from scratch, 24000 steps, AdamW+cosine+suit-aug+AMP, eval on the SAME held-out claim-legal set (disjoint from full-mixed training; top-only training set was made disjoint from this eval set). Placement = duplicate-format gate vs moyu reference (moyu-vs-moyu calibrates to 2.500), mean+/-std over 3 wall-seed blocks x (gate seeds x 4 seat rotations) games.

## Per-model results

| label | ch x blk | data | claim-rate | over-claim Δ | agree | claim-when-expert-passes | pass-when-expert-claims | placement raw | placement τ=2 | τ2−raw |
|---|---|---|---|---|---|---|---|---|---|---|
| full_64x6_s0 | 64x6 | full-mixed | 0.2949 | +0.0436 | 0.8979 | 0.0946 | 0.1057 | 2.4283±0.0146 | 2.4136±0.0136 | -0.0147 |
| full_64x6_s1 | 64x6 | full-mixed | 0.2882 | +0.0369 | 0.9007 | 0.0882 | 0.1143 | 2.426±0.0141 | 2.4078±0.0322 | -0.0182 |
| full_128x20_s0 | 128x20 | full-mixed | 0.2941 | +0.0428 | 0.9083 | 0.0873 | 0.0876 | 2.48±0.0074 | 2.4667±0.0081 | -0.0133 |
| full_128x20_s1 | 128x20 | full-mixed | 0.3012 | +0.0498 | 0.8972 | 0.099 | 0.094 | 2.4822±0.0281 | 2.4711±0.0216 | -0.0111 |
| full_128x40_s0 | 128x40 | full-mixed | 0.3063 | +0.0549 | 0.8963 | 0.1034 | 0.0872 | 2.4819±0.0138 | 2.4638±0.0137 | -0.0181 |
| full_128x40_s1 | 128x40 | full-mixed | 0.3011 | +0.0498 | 0.9002 | 0.0965 | 0.0872 | 2.487±0.0073 | 2.4767±0.0077 | -0.0103 |
| full_256x40_s0 | 256x40 | full-mixed | 0.3089 | +0.0575 | 0.89 | 0.1093 | 0.0936 | 2.4807±0.0104 | 2.4746±0.0137 | -0.0061 |
| full_256x40_s1 | 256x40 | full-mixed | 0.2963 | +0.0449 | 0.9085 | 0.0887 | 0.0829 | 2.4843±0.0315 | 2.4607±0.0187 | -0.0236 |
| top_128x40_s0 | 128x40 | top-only | 0.2469 | -0.0045 | 0.8095 | 0.1194 | 0.3759 | 2.1657±0.0105 | 2.1717±0.0147 | 0.006 |
| top_128x40_s1 | 128x40 | top-only | 0.2455 | -0.0058 | 0.8163 | 0.1138 | 0.3653 | 2.1846±0.0118 | 2.1897±0.0125 | 0.0051 |
| frac25_128x40_s0 | 128x40 | full-mixed-25% | 0.3035 | +0.0522 | 0.894 | 0.1023 | 0.0958 | 2.4595±0.0114 | 2.4397±0.0123 | -0.0198 |
| frac50_128x40_s0 | 128x40 | full-mixed-50% | 0.2928 | +0.0414 | 0.9046 | 0.0887 | 0.0979 | 2.4727±0.0114 | 2.4567±0.0243 | -0.016 |

## Honest verdict

**(i) Systematic over-claiming? YES for mixed-data imitation.** ALL 10/10 models trained on the mixed dataset claim ABOVE the expert reference (0.2514); over-claim Δ ranges +0.037..+0.058 (relative over-claim of ~15-23%). The only exceptions are the 2/2 TOP-ONLY models (Δ ≈ -0.005, i.e. at the expert rate) — which is exactly arm (iii)'s point, not a counterexample. So: over-claiming is a SYSTEMATIC artifact of imitating MIXED-skill data, and it is removed by restricting the training data to experts.

**(ii) Capacity trend** (full-mixed, claim-rate & over-claim Δ, seed-averaged):
  - 64x6: claim-rate 0.2915 (Δ +0.0403)
  - 128x20: claim-rate 0.2976 (Δ +0.0463)
  - 128x40: claim-rate 0.3037 (Δ +0.0523)
  - 256x40: claim-rate 0.3026 (Δ +0.0512)
  -> over-claim Δ changes by +0.0109 from smallest to largest capacity (bigger over-claim MORE).

**(iii) Top-only data effect** (128x40): full-mixed claim-rate 0.3037 (Δ +0.0523) vs top-only 0.2462 (Δ -0.0052). Top-only REDUCES over-claiming; gap to expert closed.

**(iv) τ=2 claim-suppression effect on placement: NULL (slightly NEGATIVE) here.** Improves placement in only 2/12 models (mean Δ -0.0117); for the full-mixed models it is consistently small-negative (mean Δ -0.0151, ~-0.01..-0.02 placement pts, several > the per-model seed-block std). So blindly suppressing claims at τ=2 does NOT improve placement, and mildly hurts it.
  - INTERPRETATION (important, honest): the placement gate scores the candidate against a **moyu** reference field that ITSELF over-claims at the same ~0.29 rate. Suppressing only the candidate's claims while 3 opponents keep claiming forfeits chi/peng value the field still takes (micro-score per game drops, e.g. -0.4..-2.5), so unilateral suppression is not rewarded. This measures the EFFECT of the correction per model (the requested number, feeds E2); it does NOT show over-claiming is harmless — establishing that needs the correction tested against a field of expert-rate (or top-only) opponents, or a wider τ sweep. The CLAIM-RATE / expert-gap results (i-iii) are the load-bearing finding; the τ overlay is exploratory.

## Caveats / skipped arms

- Budget: 24000 steps/model (fixed, comparable across arms; converged enough for stable claim behaviour but below the 16-epoch ~0.894 official peak — see val_acc in /root/e1_train.log).
- Top-only training set = `toponly_disjoint.npz` (56,272 leader decisions, made DISJOINT from the claim_states eval set to avoid train/test contamination). It is far smaller than the 5.87M mixed set, so top-only models see heavy data reuse at 24k steps (realistic expert-only regime).
- `/root/sim10_top10/cooked_top10.npz` was empty/broken at run time; the top-only arm uses leaders_outcome-derived data instead (noted, not faked).

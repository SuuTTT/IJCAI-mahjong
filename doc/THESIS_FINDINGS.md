# Key findings from the PKU thesis (李文新 group, the Botzone/IJCAI organizers)
Source: 郑启帆/Kaifan Cheng, "A Curriculum Learning Method for CSM RL AI", PKU 2024.
https://ai.pku.edu.cn/docs/2024-07/20240704001356567345.pdf

## The field's consensus (Fig 3.1) — VALIDATES our parity finding
- 2020 IJCAI champion = pure RL, but ONLY with very large compute.
- 2022 & 2023: NO pure-RL team reached top-16. Top ranks = Heuristic + Supervised Learning.
- Students trying RL: "基本没有取得成效", reward "原地震荡而不提升" (=parity), over-melds, collapses.
- => Strong SL beats RL for CSM at feasible compute. Our distill100b (strong SL) is on the right axis.

## Why CSM resists RL (§3.1.5)
- 80+ winning patterns (番种), 20+ common, mutually exclusive (三色三步高 #1 = 22.6% of strong-AI wins
  vs 混一色). Commit a main 番 + meld => hand locked => EARLY decisions decide the game.
- Long horizon: ~95% of strong-AI wins need >=7 rounds, 100+ decisions/game.

## The one RL method that works: Curriculum Learning (reproduced in rl_curriculum.py)
- Difficulty = initial-hand shanten. Win-rate by start shanten (Fig 3.5): 0sh=85% 1sh=55% 2sh=37.5% 3sh=27.8%.
- Seed RL from near-win states (rewind winning games to shanten k), stages tenpai->0-1->0-2->0-3->random.
- Reward (Algorithm 1): -1 for a Chi/Peng/Gang that does NOT reduce shanten (severe; win is only +5).
- Simplified near-win feature = 38x4x9 (EXACTLY our feature).
- Result: top-10% ladder, ~4th-5th of IJCAI 2020, far less compute than the 2020 pure-RL champ.

## Current top-7 ladder bots (decoded)
渡鸦#1 "RosmontisNet=ResNet+香厨" (ResNet SL+?), TMahjong#2 "深度强化学习" (pyccc=chunjiandu lineage RL),
res25 (ResNet-25), bot32 (ckpt32). Pattern: ResNet backbone + SL or curriculum-RL.

## Our reproduction (CL-1..4)
curriculum_states.py (shanten buckets 28/76/135/134) + sim_cnn seeding (win-rates track Fig 3.5) +
rl_curriculum.py (5-stage, Algorithm-1 reward, KL-to-SL, gauntlet-gated from distill100b floor).

## Reproduction RESULT (clean G=24 gauntlet, 120 games/candidate)
distill100b -266 (W54) > base -660 (W43) > curriculum -956 (W42).
Curriculum-RL trained STABLY (escaped the parity collapse — validates the method works) but DID NOT
beat distill100b: training on seeded near-win states overfit to FINISHING hands and degraded full-game
play (the short final 'random' stage didn't recover it). Consistent with the thesis claiming only
top-10% — which our strong SL bot already matches. FINAL: distill100b is the Sim-7 submission.
All in-house levers (RL/distill/ensemble/curriculum) now exhausted; distill100b is the confirmed best.

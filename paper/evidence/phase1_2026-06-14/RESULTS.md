# Phase-1 experiment results — 2026-06-14

Raw logs in this directory. All gauntlets: 144-game duplicate (2v2 rotated, same walls, official rebuilt
judge, plain net `CAIEST_PIMC=0`), candidate vs `lad_chunjiandu`. Net = candidate's duplicate net score.

## Strong-teacher distill gauntlet (vs lad_chunjiandu)

| candidate | teacher data (decisions) | β | teacher-agreement | net /144g | wins | verdict |
|-----------|-------------------------:|---|------------------:|----------:|-----:|---------|
| strong5 β0.3   | 8,888  | 0.3 | 0.737 | **−364** | 62 | worst (highest agreement!) |
| strong5 β0.5   | 8,888  | 0.5 | 0.721 | −8   | 70 | tied |
| TypeC β0.3     | 7,733  | 0.3 | 0.707 | **−24** | 67 | tied (best candidate) |
| TypeC β0.5     | 7,733  | 0.5 | 0.713 | −304 | 67 | worse |
| strong5_full β0.3 | 24,401 | 0.3 | 0.736 | −317 | 63 | worse |
| mythos β0.4    | 4,104  | 0.4 | 0.744 | −317 | 66 | worse |

**Conclusion: NULL.** No candidate beats `lad_chunjiandu`; best (TypeC β0.3) ties. Bigger/cleaner data did
not help. Agreement anti-correlated with play (strong5 β0.3: highest agreement, worst play). 0 illegal moves
across all ~864 bench games. → `lad_chunjiandu` stays the lock.
Logs: `bench_{b03,b05,t03,t05,s03,m04}.log`, distill: `distill_chain.log`, `distill2.log`.

## Warm-started self-play RL (JAX, full 40-block net) — infeasibility

| measurement | value |
|-------------|-------|
| warm-start win8 (fixed scorer) | **53.25%** (was 0% with the `verbose=False` bug) |
| steady rollout B4096 N90 | 558 s |
| steady rollout B2048 N55 | 171 s |
| PPO update (full or frozen-trunk) | 5.3 s/minibatch × ~470 ≈ 42 min |
| → per iteration | ~50 min (a useful run needs hundreds) |

**Conclusion: full-net warm-started RL is forward-bound and infeasible on an A4000.** Feasible path = distill
to a small net first (not run this phase).

## Field ranking (SIM-8 duplicate, mean net/game)
`[Claude]aaa` +2.39 (≈27th) · our teacher `[pycc]chunjiandu` +5.16 · strongest non-LLM
`[mythos]mythos` +9.73 / `[Infunus]TypeC青雀` +8.02 / `[aidenh]hhhhhhhhh` +7.82 · LLM-API bots
`kimi_k2` +8.63, `gpt_5_mini`, `glm_5_2`, `opus` (not clonable).

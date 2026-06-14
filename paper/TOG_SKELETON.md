# The Evaluation Gap: Why In-House Benchmarks Fail to Predict Real-Field Performance in Imperfect-Information Games

*Target: IEEE Transactions on Games (ToG). Genre: benchmarking/methodology + forensic case study.
Working thesis: under in-house evaluation (self-play, clone gauntlets, agreement proxies), bot
"improvements" are unfalsifiable — we document a complete competition campaign in which 12
reasonable interventions passed in-house plausibility and failed reality, quantify WHY each
evaluation layer misled, and distill a verified protocol + open toolkit.*

> **See also:** [`PAPER_PLAN.md`](PAPER_PLAN.md) for submit-status per section + the minimum path to a
> draft. New 2026-06-14 material ready to fold in: the **scoring-bug-as-null** (§4), the **measured
> RL-infeasibility** number (§5/§8), and the **strong-teacher distill null with real beat-us-bot data**
> (§5 — the cleanest imitation-ceiling datapoint). Details: [`../docs/FINDINGS_2026-06-14.md`](../docs/FINDINGS_2026-06-14.md);
> autopsy: [`../docs/phase1_autopsy.html`](../docs/phase1_autopsy.html). (Intervention count is now ~18, not 12.)

## 1. Introduction
- Setting: IJCAI Chinese-Standard-Mahjong competition (Botzone, duplicate format, 8-fan floor).
- The practitioner's loop: idea → in-house eval → ship. Claim: every layer of that loop can
  silently invert ground truth in imperfect-information games.
- Contributions: (C1) the evaluation-gap taxonomy with quantified, replicated case studies;
  (C2) the imitation-ceiling experiment (teacher strength × data scale, controlled);
  (C3) a verified evaluation protocol + open toolkit (bench harness, ladder collector, JAX env);
  (C4) the audited 12-intervention campaign record as a public negative-results dataset.

## 2. Background & related work
- Suphx / PKU-MCR line (SL beats feasible-compute RL — our record independently replicates this).
- Negative-results & benchmarking-honesty literature; ties to honest-rl-bench.
- PIMC / IS-MCTS in imperfect-info games; behavior cloning & the agreement proxy.

## 3. The campaign (the case-study substrate)
- Timeline table: 12 interventions, each with hypothesis, in-house verdict, real-field/held-out verdict.
  (RL ×4, AWBC, Q-rerank, V-search, champion clone, redistills ×5, PIMC ×3 variants, defense ×2.)
- All code/configs/walls released; every number traceable to a log file in `paper/evidence/`.

## 4. The evaluation gap — failure taxonomy (CORE)
Each subsection: mechanism → our quantified incident → general lesson.
1. **Self-play artifacts.** In-house: 85–89% draws, 1.8% deal-ins → "conversion is the disease,
   defense is a non-lever". Real ladder: 9% draws, 16–23% deal-ins, both conclusions inverted.
2. **Clone-field circularity.** Gauntlets vs imitations of opponents: ship +6295/240g in-house
   while −0.6 to −6/game on the real ladder. Clone opponents neither convert nor punish.
3. **Agreement ≠ play.** Champion clone val-acc 0.81 ties rank-3 clone in play (§5 grid extends this).
4. **Field-composition dependence.** (From 06-01/06-02 record: r18-vs-gen2 margin swung +152→+4778
   with seating; farming-fitness winner flips per opponent.)
5. **Harness contamination.** Three incidents: co-tenant CPU starvation; stale deadlocking bench
   (all games "stuck", read as a verdict); remembered-baseline transfer across configs ("+4119").
6. **Noise-floor illiteracy.** §6 replication: identical pair, 5 wall sets → spread defines the
   minimum publishable margin; most of our "wins"/"losses" sat inside it.
7. **Selection under noise = coevolution drift.** (06-01 record: 240-game selector promoted a
   regression; the deploy file was silently overwritten twice.)

## 5. The imitation ceiling (controlled experiment — RUNNING on P4000)
- Design: identical 40-block BC students; teacher-strength axis = 4 teachers of KNOWN real strength
  from the same 2025 final (SeaMan +0.89/g, dimaria +0.58, PAMA −0.08, 哞哞哞 −0.31; ~176k decisions
  each); data-scale axis = SeaMan 22k/44k/88k/176k. Same held-out walls (WSB=880000) + ship reference.
- Outputs: val-acc (agreement) AND gauntlet net (play) per cell → the scatter that shows agreement
  rising while play stays flat (or not — either result is reportable).
- RESULTS: `paper/evidence/ssh1/grid_train.log`, `grid_eval.res`. [TBD]

## 6. Quantifying the noise floor (RUNNING on 5060)
- Identical bot pair, 6 opponents × 24 games, replicated over 5 disjoint wall sets.
- RESULTS: `paper/evidence/b5060/noisefloor.res`. [TBD] → recommended minimum N / margin tables.

## 7. The verified protocol (the constructive artifact)
- Persistent-bot harness (kills model-reload stuck-noise: 7–20/72 → ~4%); thread-reader IPC;
  played/stuck accounting as a validity gate; held-out walls; position control; head-to-head only;
  same-box same-config; pre-registered decision rules (the 06-13 swap rule as the worked example);
  real-field telemetry loop (collector + tracked-bot category + ladder_report).
- Toolkit release: eval/, tools/, JAX CSM env (85× throughput as the scale-experiment enabler).

## 8. Discussion
- Why imperfect-information + duplicate scoring amplifies every failure mode (high variance, 8-fan
  floor, field-dependent meta).
- What WOULD escape the ceiling: scaled self-play (JAX env Phases 2–4) — falsifiable future work;
  test-time search post-mortem (why opponent-aware PIMC with a non-converting rollout is null).
- Limits: one game, one team, n=44–100 real-ladder games for some claims.

## 9. Conclusion
- The 12-null record is not a failure story: it is what honest evaluation OUTPUTS when the
  evaluation itself is the broken component. Fix the meter before the machine.

## Evidence map (auditable)
| Claim | Source |
|---|---|
| 9% draws / 16% deal-ins real | tools/ladder_report.py over others/ladder_top30_score1216/ |
| 3-arm defense verdict | paper/evidence/b5060/safe_ab.res, fold_ab.res |
| PIMC nulls | paper/evidence/ssh8/pimc_spike.log, b5060/pimc_par*.res |
| champion-clone tie | paper/evidence/ssh8/g8b.log (clean head-to-head) |
| grid (§5) | paper/evidence/ssh1/grid_train.log, grid_eval.res |
| noise floor (§6) | paper/evidence/b5060/noisefloor.res |
| contamination incidents | docs/ANALYSIS_2026-06-11.md §eval-infra + session logs |
| historical record | docs/ANALYSIS_2026-06-01.md, docs/HANDOFF_TO_WIN.md |

# Paper plan — "The Evaluation Gap" (IEEE Transactions on Games)

Concrete, submit-oriented plan. Detailed section structure lives in [`TOG_SKELETON.md`](TOG_SKELETON.md);
this file tracks **what's done, what's missing, and what it takes to submit.**

## One-paragraph pitch
A complete, audited competition campaign (IJCAI-2026 Chinese-Standard-Mahjong, Botzone) in which ~18
reasonable interventions passed in-house evaluation and failed reality. We turn that record into (1) a
**failure taxonomy** of how each in-house evaluation layer silently inverts ground truth in imperfect-
information games, (2) a controlled **imitation-ceiling** study, (3) a **noise-floor** quantification, and
(4) a **verified evaluation protocol + open toolkit**. The negative results are the contribution.

## Why this is a real paper (not just a loss report)
- Every number is traceable to a log (`paper/evidence/`), with released code, walls, and configs.
- The thesis is falsifiable and replicated: in-house ⇄ real-field sign flips occur repeatedly and we
  explain the mechanism each time.
- It is constructive: the verified protocol + toolkit is a usable artifact, not just a critique.
- It ties to a public sibling repo (`honest-rl-bench`) — a credibility anchor.

## Section status

| § | Content | Status | Gap to submit |
|---|---------|--------|---------------|
| 1 | Intro + contributions | drafted in skeleton | tighten to 1 page |
| 2 | Related work (Suphx, **Tjong**, PKU-MCR, PIMC/IS-MCTS, negative-results lit) | **partial** | **add Tjong + fan-backward properly (see Lesson 1)**; 1 day |
| 3 | The campaign — timeline of ~18 interventions, each w/ in-house vs real verdict | data exists | assemble the master table from CHANGELOG + evidence; 1 day |
| 4 | **Evaluation-gap taxonomy** (the core) | strong | self-play artifacts ✓, clone circularity ✓, agreement≠play ✓✓ (now 2 incidents), noise-floor ✓, harness contamination ✓, **+ the scoring-bug-as-null (new §4.x)** |
| 5 | Imitation-ceiling controlled experiment | partial run | finish the teacher-strength × data-scale grid; **the strong-teacher distill null (this session) is a clean add** |
| 6 | Noise-floor quantification | data started | finish 5-wall-set replication; produce min-N tables |
| 7 | Verified protocol + toolkit | built | write it up; point to repo |
| 8 | Limitations / "is Mahjong a good testbed" | new | write from autopsy Lesson 3 |

## New, paper-ready material from 2026-06-14 (this session)
1. **Scoring-bug-as-null (→ §4).** A "win8=0, RL can't convert" conclusion was a `verbose=False` fan-sum
   typo; fixed, the warm-start wins 53%. *Lesson for the field: a null can be instrumentation. Audit the
   measurement before believing the negative.* Strong, concrete, reviewer-friendly.
2. **Measured RL-infeasibility (→ §5/§8).** 558 s/rollout, ~50 min/iter for the full deploy net — quantifies
   "RL won't be unlocked by compute" with numbers, motivating the small-net + fan-backward path.
3. **Strong-teacher distill null with real beat-us data (→ §5).** Distilling toward the *actual stronger
   bots* (mythos +9.73, TypeC +8.02), best result ties (−24/144g), higher-agreement-loses-more. The
   cleanest imitation-ceiling datapoint yet: you cannot out-imitate a stronger bot's *observable* moves
   when its strength comes from search/LLM reasoning you can't see.
4. **Field taxonomy finding (→ §3/§8).** Several top bots are LLM-API players — relevant context for "what
   is the field" and for the testbed discussion.

## Minimum path to a submittable draft (~2 weeks of focused work)
1. **Lit review + §2** (Tjong/fan-backward, Suphx, PKU-MCR). [2 d]
2. **§3 master timeline table** from CHANGELOG + `paper/evidence/` (one row per intervention,
   hypothesis / in-house / real). [2 d]
3. **§4 taxonomy** with the 5–6 mechanisms, each = mechanism → quantified incident → general lesson;
   add the scoring-bug subsection. [3 d]
4. **Finish §5 grid + §6 noise floor** (compute-light; reruns on a single box with the rebuilt judge). [3 d]
5. **§7 protocol write-up + artifact polish** (the gauntlet, collector, JAX env, the small-net RL recipe as
   "the feasible path"). [2 d]
6. **§8 limitations / testbed reflection** from autopsy Lesson 3. [1 d]
7. Figures via `paper/scripts/make_figures.py`; assemble `~/tog-evaluation-gap/main.tex`. [2 d]

## If we can't do the full paper now
The autopsy (`docs/phase1_autopsy.html`), findings (`docs/FINDINGS_2026-06-14.md`), README, and CHANGELOG
preserve every result and recipe. The skeleton + this plan make the paper a *resume-able* task, not a
restart. The evidence directory (`paper/evidence/`) and the data assets (README "Data assets") are the
raw material; nothing needed for the paper has been lost.

## Risks
- **Field intelligence is incomplete** (Lesson 2): we don't have winning-team write-ups; the §3 narrative
  leans on our own logs. Mitigation: frame as a single-team forensic case study, which is honest and still
  novel.
- **§5/§6 need clean reruns** on the rebuilt judge to be citable; the older numbers came from boxes since
  reassigned. Compute-light, but must be redone for traceability.

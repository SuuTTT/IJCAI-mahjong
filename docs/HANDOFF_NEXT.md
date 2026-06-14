# Handoff — what to do next

Written 2026-06-14 at Phase-1 close (SIM-8 27/38, submission = `lad_chunjiandu` + net-PIMC `[Claude]aaa`).
Three scenarios: **(A)** we advance to Round 2, **(B)** we re-enter next year, **(C)** the ToG paper.
Read the **[Phase-1 Autopsy](phase1_autopsy.html)** first — the lessons below assume it.

## Where everything lives now
- **Code + docs + result logs:** github.com/SuuTTT/IJCAI-mahjong (master). Rebuild recipes in `CHANGELOG.md`.
- **Models + decision datasets:** HF `Dannibal/ijcai-mahjong-phase1` (private) — `lad_chunjiandu`, `distill100b`,
  all distill candidates, strong-teacher npzs.
- **Raw SIM-8 game logs:** HF dataset `Dannibal/ijcai-mahjong-sim8-games` (per-bot `tar.gz`).
- **Compute:** the eval box (`40664158`) was destroyed. **`40833388` (ssh2.vast.ai:33389, A4000, 128-core,
  cap ~20 cores — shared)** is assigned for next mahjong work. First step on any fresh box: rebuild the C++
  judge (recipe in `CHANGELOG.md` 2026-06-14) and `pip install torch PyMahjongGB optax`.
- ⚠️ **Rotate the exposed tokens** (HF `hf_TkQF…`, the GitHub `ghp_…`) before reusing.

---

## A · If we advance to Round 2 (Final, ~July 7)

The submission is already the proven best. **Don't ship a tied-or-worse model.** Highest-value moves, in order:

1. **Close the teacher gap (cheapest, highest-confidence).** We sit at +2.39; our own teacher `chunjiandu`
   sits at +5.16 — we lost ~2.8 pts in distillation. Re-distill `lad_chunjiandu` from `chunjiandu`'s
   **current** ladder games (collect fresh, version-filter with `extract_top30 --since`), 600–800 steps,
   gauntlet-gate at **≥144 games** vs `lad_chunjiandu`. This is the one lever that targets a known, real gap.
2. **Implement Tjong fan-backward, then small-net warm-started RL.** Distill the 40-block net into a
   ~64-ch/3-block net (the full net is forward-bound — `train_ppo_ws.py` measured ~50 min/iter), BC-warm-start
   it (the warm-start machinery in `train/jax_env/` is built and validated), then PPO with **fan-backward
   reward shaping** (the published fix for our `win8=0` sparse-reward wall — we never implemented it). This is
   the only RL path with a real chance.
3. **Net-PIMC tuning.** The deploy search (`deploy/caiest_cnn/`) was net/tied; with the rebuilt judge you can
   now properly A/B PIMC depth / determinization count / the `value_search` rerank (`CAIEST_VNET`, was +57,
   inside noise — re-test with a better V on more data).
4. **Decision rule:** ship a candidate **only if** it beats `lad_chunjiandu` by a margin clearly outside the
   noise floor (±537/144g for identical bots; the decisive historical gaps were ~+181/144g). Otherwise ship
   `lad_chunjiandu`. Verify the uploaded zip via its debug `[md5]` line.

**Do NOT bother re-running** (conclusive nulls): champion clone, AWBC, Q-rerank, model soups, 8-fan mask,
defense knobs, full-net self-play RL, strong-field-bot distill. See `docs/FINDINGS_2026-06-14.md` and the
README results table.

---

## B · If we re-enter next year (apply the 3 lessons)

1. **Literature review is step 0.** Before any modeling: Suphx, **Tjong (fan-backward)**, PKU-MCR/NAGA,
   PIMC/IS-MCTS. One page each: what they did, what we'll borrow. We wasted weeks re-deriving Tjong's recipe.
2. **Stand up field intelligence from round 1.** A standing job that, every Simulation: ranks the field by
   duplicate net/game (`extract_top30` profile + the ranking script in `CHANGELOG`), version-filters, and
   taxonomizes the top 16 (SL / RL / search / **LLM-API**). Know who you're chasing before you build the chase.
   (This year we learned the top bots include LLM players — un-clonable — only at the very end.)
3. **Build the eval harness before the model.** Our biggest leverage was the trustworthy duplicate gauntlet +
   real-field collector, not the models. Start there: persistent-bot bench, rebuilt judge, noise-floor
   calibration (know your minimum publishable margin on day 1), real-ladder A/B as the only ground truth.
4. **Pick the architecture the field already validated:** strong SL on a large expert corpus, distilled from
   a single coherent strong teacher (coherence > diversity held every time), fine-tuned in the first ~600–800
   steps (recipe law). RL only with fan-backward + a small net. Don't try to out-imitate a reasoning model.
5. **Reuse this repo's infra** — it's all in git/HF and rebuilds in ~30 min. Don't start from scratch.

---

## C · ToG paper — "The Evaluation Gap"

Plan and section status: **[paper/PAPER_PLAN.md](../paper/PAPER_PLAN.md)**; skeleton: `paper/TOG_SKELETON.md`;
LaTeX: `~/tog-evaluation-gap/main.tex`. The negative results **are** the contribution.

**Immediate next steps (~2 weeks focused):**
1. **Lit review + §2** (Tjong/fan-backward, Suphx, PKU-MCR) — the gap reviewers will flag. [2 d]
2. **§3 master timeline table** — one row per intervention (now ~18) from `CHANGELOG.md` +
   `paper/evidence/`: hypothesis / in-house verdict / real-field verdict. [2 d]
3. **§4 failure taxonomy** — 5–6 mechanisms, each *mechanism → quantified incident → general lesson*. Add the
   new **scoring-bug-as-null** subsection (a null that was a typo) — concrete and reviewer-friendly. [3 d]
4. **Finish §5 imitation-ceiling grid + §6 noise-floor** — compute-light reruns on `40833388` with the rebuilt
   judge (the old numbers came from now-destroyed boxes; redo for traceability). The 2026-06-14 strong-teacher
   distill null (agreement 0.737 → worst play) is a clean §5 datapoint. [3 d]
5. **§7 verified protocol + artifact** (gauntlet, collector, JAX env, small-net RL recipe) and **§8 limitations /
   "is Mahjong a good testbed"** (autopsy Lesson 3). [3 d]
6. **Figures** via `paper/scripts/make_figures.py`; assemble `main.tex`; internal review. [2 d]

**Paper-ready material already banked this session:** the scoring-bug null, the measured RL-infeasibility
number, the strong-teacher-distill null with real beat-us-bot data, the LLM-in-the-field observation.
**Risk:** §5/§6 need clean reruns to be citable; everything else is written or logged. Nothing is lost.

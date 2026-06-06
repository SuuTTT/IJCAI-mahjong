# Methods Ledger
## 8-DAY COMPETITION PLAN (deadline 2026-06-14 23:55)
Winning axis = strong SL ResNet (field-confirmed). Autonomous work strengthens the base; data lever
(re-distill on top-player games) runs when data arrives. distill100b = locked floor.
- Phase 1 (now): P-A1 stronger SL (AdamW+cosine+suit-aug, val_acc metric) + P-A2 trustworthy eval.
- Phase 2: P-B1 curriculum-RL v2 (fix overfit) + pMCPA test-time adaptation.
- Phase 3 (data-gated): re-distill on top-player data (collect_winners.py ready) — highest ceiling-raiser.
- Phase 4: P-D1 pure-NumPy deploy fallback + final large-N gauntlet/ladder bake-off.
- Phase 5: lock submission + buffer.
 — tried, untried, and the measurement gap

Complete checklist of every method from `deepresearch.md` + `deepresearch-gemini.md` (Suphx + PKU/
league winners), plus what we've run. **Read `TODO.md` for the prioritized roadmap; this is the full
method inventory.** Deadline 2026-06-09.


## 🎯 GAUNTLET RESULT (M1 done) — the parity verdict was a MEASUREMENT ARTIFACT
Two independent runs (G=6 @seed50000, G=16 @seed60000) vs 5 diverse-arch opponents (cnn16, cnnattn,
resbn24, resbn56, w192). Total net vs gauntlet (avg of both runs):
  dense +142 · distill +113 · rl3 +47 · league +47 · **base (resbn40) -94 (LAST, both runs)**
=> The RL/distill models that "tied base" in self-vs-twin eval are actually STRONGER vs varied
opponents — RL improved ROBUSTNESS, which monoculture eval is blind to. (illegals in bench = PASS-on-
timeout contention artifacts; greedy-masked bots can't emit illegal.) Gauntlet opponents are CNN-family
(same data) so it's a proxy, not the ladder — but far less blind than self-vs-base.
**DEPLOY UPGRADE: ship distill100b (fused, deploy-ready: deploy/cnn_distill100b.pkl) over base.**


## PROGRESS (Tier-2 build + incremental data pipeline)
- DATA PIPELINE (incremental): collect_winners.py — drop top-player log batches under a folder, it
  auto-extracts decisions (4-same self-play -> ALL seats; mixed -> WINNER seat only), dedups by
  match-id, grows one cumulative npz (data/topwinners.npz). Re-distill: distill.py finetune_frac.
  COLLECTION GUIDANCE: lead with 4x the #1 bot (渡鸦) self-play (4x more decisions/game, strongest
  teacher, replaces mid-pack chunjiandu); supplement with 4-different-top-bot games (diverse). Aim 200+.
- F1 safe-discard (#28): built safe_discard.py (genbutsu/furiten = a tile in opp's discards can't ron
  them). AGGRESSIVE version tanks offense (1-6, fights the CNN's learned defense). CONSERVATIVE
  (turn>=12, 2+ melds, top-2) = offense-NEUTRAL (+12/12g noise). Opt-in (SAFE_DISCARD=1). Small upside:
  our CNN already defends (0 deal-ins over 4 ladder games). Ladder-A/B only; not shipped by default.

## ⚠️ THE BINDING CONSTRAINT: measurement
Every RL/distill method ties in LOCAL eval because our eval pool is a **monoculture** (all resbn40
variants → near-twins trade evenly). It is NOT a fundamental limit: resbn40 vs 16-block CNN = +973,
vs MLP = huge — *diverse* opponents separate cleanly. The SOTA systems solve this with a **diverse
opponent yardstick** (Suphx→Tenhou ladder; league→diverse population). **Until we fix measurement,
trying more methods is shooting in the dark.** So measurement is P0.

Status: ✅ done · 🔄 running · ⏳ queued · ❓ needs data/user · ✗ tried, didn't beat base

---

## P0 — MEASUREMENT (unblocks everything)
- [✅] **M1. Diverse eval gauntlet — DONE (see GAUNTLET RESULT above).** Score every candidate by net across DIFFERENT-architecture
  opponents — 16-block CNN (base_16x128), old r18 MLP, a rule-based/heuristic bot, the
  chunjiandu-imitation (distill100b), caiest 16-block. Net-vs-gauntlet separates strong policies
  where vs-base ties. Build `eval/gauntlet.py`. **This is how we measure without the ladder.**
- [ ] **M2. Botzone ladder signal (USER).** Deploy fused resbn40 + distill100b as two bots, play
  ranked games. The real external yardstick (the project has never had one). Even ~20 games informs.
- [ ] **M3. Fixed held-out reference opponents in RL eval-gate.** Replace "vs own base" promotion
  gate in rl_league with "net vs the M1 gauntlet" so league promotion tracks real strength.

## P1 — DEPLOY (competition-critical, in hand)
- [✅] BatchNorm fusion + legacy serialization → resbn40_fused (deploy/caiest_cnn, +973, verified).
- [❓] **Verify fused resbn40 on Botzone (USER).** Re-upload zip + cnn.pkl.
- [✓~] **Distill100b A/B on ladder (USER).** deploy/cnn_distill100b.pkl as a 2nd bot vs fused resbn40.

## P2 — DATA (the proven lever for distillation)
- [❓] **D1. chunjiandu vs DIVERSE top opponents (USER).** More valuable than 4×chunjiandu self-play:
  covers more situations + matches the ladder distribution. Breaks the distill 0.73 plateau
  (which is situation-diversity-limited, NOT sample count — suit-aug 6× gave nothing).
- [❓] **D2. More chunjiandu self-play games (USER).** Diminishing returns vs D1, but still raises
  coverage. ~300-500 games → meaningfully past 0.73.
- [⏳] **D3. Re-run distill (finetune_frac) on the richer D1/D2 data.** Pipeline ready (distill.py).

## P3 — UNTRIED RESEARCH METHODS (validate via M1 gauntlet / ladder)
- [ ] **R1. Look-ahead / fan-potential INPUT features.** Add distance-to-8-fan, fan-potential,
  effective-shanten-under-8-fan as feature planes (NOT just the action mask, which was null).
  Cheap; changes representation. Retrain SL head. Upside limited (CNN already converts at 3% draw).
- [ ] **R2. Dynamic entropy regularization.** Suphx: anneal the entropy coef to keep self-play
  exploration in a useful band. Cheap add to rl_league (currently fixed 0.01).
- [ ] **R3. Oracle guiding.** Train a perfect-info teacher (sees opponents' tiles + wall) → distill
  to the public-info student via KL, decaying the hidden features. The biggest untried Suphx lever.
  BIG build + Suphx used it at scale; attempt only after M1/M2 confirm RL can move the needle.
- [ ] **R4. pMCPA test-time adaptation (#21).** At the deal, sample 10^5 hidden states, rollout,
  a few gradient steps to adapt to the dealt hand, then play. We have ~6s/turn. Measure gain vs latency.
- [ ] **R5. League exploiter (vs historical average).** We have MAIN exploiters; add a LEAGUE
  exploiter targeting systematic biases across all checkpoints. Only if M3 shows league gains.
- [ ] **R6. Curriculum phases.** Heuristic→challenge→league. Mostly for from-scratch; we already
  have a strong SL base, so low priority.

## P4 — DEFENSE (research-listed; our deal-in already <2%, low upside)
- [ ] **F1. Opponent-hand estimation + safe-discard.** Secondary net predicts opponents' tiles from
  public info; enforce safe discards vs a tenpai opponent. Refinement, not a ceiling-breaker.

## TRIED — tied base in BLIND self-vs-twin eval, but most BEAT base on the diverse gauntlet (see top)
- ✗ Plain self-play RL (single frozen base) → parity.
- ✗ Pool + KL-to-SL (rl_actors.py) → parity (39-39 vs base).
- ✗ League: main-exploiter + PFSP (rl_league.py, 400it/13 promotes) → parity (33-36).
- ✗ Global reward prediction / dense reward (Suphx Φ, R²≈0.43, rl_league_dense.py, 600it) → parity (44-52).
- ✗ 8-fan look-ahead action masking → null (+74/60g; CNN already converts).
- ✗ SL distillation (100 chunjiandu games, champ-fraction) → agreement 0.68→0.73, local parity (needs ladder/more data).
- ✗ Suit-permutation augmentation (6×) for distill → no gain (0.726→0.731; game-diversity-limited).
- ✗ Wider/deeper-conv/attention/GNN architectures → all tie-or-below resbn40.
- ✗ High-fan human-data BC fine-tune → broke policy. Test-time fan-rollout planner → hurt.

## Binding facts
- **resbn40 is NOT the best deployable — it's LAST on the diverse gauntlet.** RL/distill (dense,
  distill100b, rl3) beat it; the earlier "parity" was monoculture-eval blindness. Deploy distill100b.
  RL is NOT exhausted — it improved robustness. Next: M2 ladder (final word), D1 diverse data, R1/R3.
- #1 bot chunjiandu = SL+RL; we match ~0.73 of its discards. Botzone = py3.6/torch1.4/~512MB/~6s.

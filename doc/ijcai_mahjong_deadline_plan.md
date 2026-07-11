# IJCAI-2026 Chinese Standard Mahjong (MCR) — Deadline Plan & Status

**Goal:** rank #1 in the Round-2 Final (deadline **2026-07-07**) — **against the actual
16-bot finalist field, NOT "beat moyu."** Our bot **moyu** (`cnn_lad_chunjiandu`, CNN
imitation of top humans) ranked **11/16** in Round 1.

> **⚠️ Objective correction (2026-06-22):** the prior gate (`verify_lever.sh`,
> cand-vs-moyu) tests the WRONG objective. Beating moyu ≠ winning the competition.
> moyu-vs-moyu is two elite defenders → low win-rate, lots of 2nds, so everything "ties
> moyu." But the real field includes bots WEAKER than moyu (we're 11/16), which feed
> tiles and under-defend. A strategy that beats the *field* without beating *moyu*
> (exploits opponents moyu doesn't resemble) scores "parity" on the moyu gate and gets
> falsely killed. **New primary gate = real-field placement-points vs the actual field;
> moyu gate demoted to a cheap sanity check known to miss field-specific gains.**

**Living document.** Status markers: ✅ done · 🔄 in progress · ⬜ todo · ❌ killed.
Last updated: **2026-06-22**.

---

## 1. Current standing (measured, not assumed)

Real-field measurement vs the actual finalists, **N=242**, every figure from a fetched
Botzone `/match` log (durable: `realfield_BIG_RESULTS.verdict_latest.txt`, ssh3):

| metric | moyu |
|---|---|
| top-half rate | **81% [76–86]** |
| 1st / 2nd / 3rd / 4th | **28% / 54% / 0% [0–2] / 19%** |
| sole outright win | 24% |
| TLE / RE | **0 / 0** |
| mean score | CI crosses 0 (competitive / break-even) |

**Diagnosis:** moyu is a **steady-2nd specialist with elite defense** — it never busts
(0/242 third) but under-converts 2nd→1st. Under the duplicate placement-point format
(4/3/2/1 summed), consistently parking at 2nd vs a beatable field is exactly how a bot
lands mid-table (the 11/16).

---

## 2. The decisive strategic finding (lever audit, 2026-06-22)

Full audit: `lever_audit.txt` (ssh3). **The "early candidates died on weak proxies, so
the new Botzone real-field A/B re-opens them" hypothesis is NOT supported.**

- Every lever was killed by one of: **(b)** the *strong-moyu* gate (`verify_lever.sh`:
  bias-corrected cand-vs-moyu minus matched moyu-vs-moyu null, N≥400, 2 seed families —
  *harder* than the real field, removes wall/seat luck); **(c)** a real-field A/B; or
  **(d)** mechanism / replay analysis. **No lever died on a weak proxy (class a).**
- it80 confirmed empirically that **self-play-vs-moyu tracks the real field** (null in
  both) — so the strong gate is a valid cheap filter; we don't need to burn games to
  re-test generic-strength levers.
- **Only one lever was a genuine bugged no-op: AWR** (EDGE exactly 0.000 → never
  actually tested). That is the single defensible re-screen.

### Lever ledger (verdict + why it stays dead)

| lever | kill-basis | verdict |
|---|---|---|
| SL / imitation | — | **incumbent (ship)** |
| PPO | (b) strong | ❌ 39-39 parity trap |
| DMC | (b) strong | ❌ EDGE −10.14/g |
| Suphx-RL (GRP+oracle) | (b)+(d) | ❌ parity; oracle-critic luck baseline |
| Fan-Backward RL | (b) strong | ❌ FAIL both seed families |
| Self-play League-RL | (b) strong | ❌ eval random-walk ≤0 |
| Placement-RL it80 | (c)+(b) | ❌ **double-killed** (real-field 1st 25% vs 28%; gate null) |
| PIMC online | (b)/(d) | ❌ null; off by design |
| Offline deep-PIMC | (b)+(d) | ❌ teacher +1.78/g < +2.0 floor |
| Ensemble | (b) strong | ❌ all mixtures fail |
| **AWR** | **(e) BUGGED no-op** | ⬜ **the one fair re-screen** |
| Defense (fold-v1/v2, meld-ceiling) | (d)+(b) | ❌ MCR-structural null (pay −12 when anyone wins) |
| Human-knowledge (Wait/DealinNet) | (d) | ❌ predictors accurate, every *use* null |
| Wait-quality overlay | (d) | ❌ 0 overrides (byte-identical to moyu) |
| HL-loop / rank-risk overlay | (d) | ❌ undeployable on Botzone + oracle-negative |
| Construction-fix | (d) | ❌ winners build the SAME waits (no headroom) |

### Why the brainstorm is genuinely thin (structural constraints)
- **8-fan hard minimum** → cannot trade hand value for speed (cheap hands can't win).
- **~−12 paid whenever anyone wins** → defense banks too little to forfeit win equity.
- **24-perm duplicate format cancels variance** → risk-posture / "gamble from behind"
  is a strict EV loss (proven even with an oracle standing signal).
- **Winners build the same thin/sub-floor 8-fan waits moyu does** (paired width Δ≈0) →
  no construction target to learn toward.
- **SL base is near-ceiling** → RL finds no moyu-beating gradient.

---

## 3. Plan to 2026-07-07

### Phase 0 — Lever audit ✅ DONE (2026-06-22)
Classified all 16 levers by kill-basis. Result: lever space is exhausted except AWR;
Botzone does not re-open it. (`lever_audit.txt`)

### Phase 1 — AWR fair re-screen ❌ NULL
Load verified (md5≠moyu, 7.5–9% argmax-flip = genuine policy). Gate fam3.0M: be8 EDGE
−5.48/g, be16 −2.44/g — both FAIL +1.0/g (can't pass). The one bugged-eval lever, now
fairly tested = null. (`awr_rescreen_RESULTS.txt`)
The one lever never truly tested (bugged no-op). Locate/retrain the AWR best-response
fine-tune; run through `verify_lever.sh` at **N≥400 × 2 seed families** on the
*fixed* code path (not the ignored `CAIEST_MODEL` path).
- **PASS** (clears +1.0/g both families) → graduate to a real-field A/B (distinct name).
- **NULL** → the imitation ceiling is confirmed by exhaustion; proceed to Phase 3.
- Expected: low payoff (the fixed-codepath retry already read parity-to-below at N=160),
  but it is the one genuinely-unfair kill, so it earns a clean test.

### Phase 2 — Field-exploitation (the reframed direction) ❌ REFUTED (H_NOHEADROOM)
Tested the corrected-objective flagship: does moyu under-farm beatable opponents?
Diagnostic (`field_exploit_diagnostic.txt`, ssh3) → **NO.** (1) moyu's 1st-rate is flat
across all 4 finalists (25–30%, CIs overlap); player152 is its *hardest* matchup, not a
farm. (2) In 15/16 tenpai-but-lost games moyu had no agency (opp self-draw / a *3rd*
player fed the winner); it stayed on **live waits (13/15)** and lost the race fairly to
developed (fan-15.6) hands — not folding winnable hands. (3) 73% of moyu's wins are
opponent feeds — it already cashes the looseness. (4) it80 (N=79) already gained 0 firsts.
**Third independent confirmation (with construction-gate + it80) that the gap to 1st is
race variance on equivalent hands, NOT exploitable passivity.** Exploitation-training
launched then KILLED on this finding (don't train toward a refuted premise).

### Phase 2b — Imitation-capacity (FIRST POSITIVE SIGNAL) 🔄 VERIFYING
The one GPU axis the audit never tried: does moyu's CNN UNDER-FIT the champion data?
- **Champion SL data RECOVERED** (was lost in prior data-loss): found on box `3060tkde`
  (91.150.160.38:11708) — `union_chun_top30.npz` (79,898 champion decisions) + 3,546 raw
  Botzone rank-1 logs. Staged to China box. (`champion_data_recovery.txt`)
- **256×40 scaled imitation BEATS moyu on held-out top-1: 0.767 vs 0.736** (+2.8 over a
  same-budget 128×40 control) → **moyu's 128ch CNN was under-fit.** FIRST non-null signal
  in the campaign.
- **BUT — not verified as strength.** Accuracy is a weak proxy; moyu still LEADS on top-3
  (0.947 vs 0.929); the strength gate wasn't run. This project's recurring failure is
  exactly proxy-gains-evaporating-at-the-gate. So:
  - ❌ **Strength gate → VERIFIED NULL** (`imit256_strength_gate.txt`): 256×40 EDGE
    −36.9/−36.3/g, 128×40 control −40.8/−40.4/g vs moyu (N=400×2, ~21 SE below +1.0/g).
    Accuracy gain *reversed* at play. No-op-trap ruled out; monotonic capacity ordering
    confirms it's real, not a bug. **moyu (`ckpt/suphx/` = SL+Suphx-RL on more than the
    recovered 80k set) can't even be reproduced by from-scratch BC on recoverable data.**
    Capacity sweep killed (refuted premise). Imitation-capacity axis DEAD.

### Phase 3 — Finalize 🔄 STANDING
- **moyu (lean, 0 TLE) is the locked Round-2 submission** until something *passes a
  real-field A/B* (1st-rate up, 4th-rate not up). Nothing has.
- Deliverables: the negative-result write-up + the 2 blogs + the real-field-eval infra.
- Housekeeping: rotate Botzone account passwords; release the
  China-box GPU when AWR finishes.

### Standing rules (verification discipline — this project fabricated ~8×)
- Every number from a file (fetched log / ledger / JSON); n=0 = FAIL, never a pass.
- A candidate replaces moyu ONLY by passing a real-field A/B, after passing the
  strong-moyu gate first (cheap filter before spending games).
- Game volume: moderate + exponential backoff (we already trip 429s); distinct bot
  names per candidate; never burst (account ban = losing the entry).
- Do not touch the TDMPC project's GPUs/CPU/assets on ssh3.

---

## 4. Where we are RIGHT NOW (2026-06-22)
- ✅ Real-field measurement complete (N=242) — diagnosis locked.
- ✅ Placement-RL it80 built, deployed (`moyupp`), real-field A/B'd (N=79) → NULL.
- ✅ Lever audit complete → Botzone doesn't re-open the space; AWR is the one loose end.
- 🔄 **Launching the AWR re-screen** (Phase 1) on the China box.
- 🔄 moyu remains the standing submission.

---

## 5. Changelog
- **2026-06-22** — Found the project's git repo (`SuuTTT/IJCAI-mahjong`, master) — full June
  campaign + recipe. CORRECTION: moyu = top-30 ladder single-teacher distill (not BC-finetune);
  it's reproducible (repo + `~/data.zip` = official 98k). **Capacity lever → NULL (already in
  repo**: `docs/ARCHITECTURES.md` — 256ch & deeper all tie-or-below resbn40; "agreement≠play").
  Repo verdict: "algorithmic search complete; moyu is the Round-2 submission." Launched durable
  backup (docs→repo, data.zip+weights→HF). Net new spend on capacity: $0 (checked repo first).
- **2026-06-22** — AWR re-screen → **NULL** (be8 −5.48/g, be16 −2.44/g vs moyu; load
  verified). The last fair-test gap closed. **Lever space exhausted; recommend finalize.**
- **2026-06-22** — Imitation strength gate → **VERIFIED NULL** (256×40 −36.9/g, 128×40
  control −40.8/g vs moyu). The accuracy lead reversed at play; not a bug (monotonic
  capacity ordering, moyu null=0.000). Finding: moyu (SL+Suphx-RL, more data) can't even
  be reproduced by from-scratch BC on recoverable data. Killed the sweep. First positive
  signal was a proxy mirage — the recurring pattern.
- **2026-06-22** — **FIRST POSITIVE SIGNAL (later NULLed).** Recovered the lost champion
  SL dataset (box `3060tkde`). 256×40 scaled imitation beat moyu on held-out top-1 (0.767
  vs 0.736). Launched the decisive strength gate + capacity sweep.
- **2026-06-22** — Field-exploitation diagnostic → **H_NOHEADROOM** (no farm target, not
  passivity, already cashes feeds). Killed the exploitation training (refuted premise).
  Launched imitation-capacity flyer (Phase 2b, modest prior) to use the GPU. 3rd
  independent confirmation the gap to 1st is race variance, not strategy.
- **2026-06-22** — **OBJECTIVE CORRECTION**: gate reframed from "beat moyu" → "win vs
  the actual field." Launched field-exploitation diagnostic (Phase 2). moyu gate demoted.
- **2026-06-22** — AWR re-screen launched (Phase 1).
- **2026-06-22** — Lever audit done (`lever_audit.txt`): no weak-proxy kills; AWR the
  only fair-test gap; brainstorm thin (structural). Plan reframed from "re-test on
  Botzone" → "AWR check, then finalize."
- **2026-06-22** — Placement-RL it80 real-field A/B final **N=79: NULL** (1st 25% vs
  moyu 28%, 4th 20% vs 19%); caught+fixed the `model.cfg`-drop no-op trap; deploy clean.
- **2026-06-22** — Placement-rank-reward RL **NULL** (self-play gate, 5 ckpts, all z<1.1).
- **2026-06-21** — Construction-viability gate: **H_ceiling** (winners build same waits).
- **2026-06-21** — Real-field measurement converged N=69→242 (top-half 91%→81% honest).
- **2026-06-21** — Wait-quality overlay: **NO-OP** (0 move overrides).
- *(earlier)* — SL/PPO/DMC/Suphx-RL/Fan-Backward/League-RL/PIMC/ensemble/defense/
  human-knowledge/HL-loop all null vs the strong-moyu gate or by mechanism.
</content>

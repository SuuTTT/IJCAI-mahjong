# Plan to Win — post-2025-gauntlet (2026-06-08, deadline 06-14 23:55)

**Situation (read from results, not memory):** distill100b / V1 / champ2025 are statistically
equivalent — "best" flips by opponent pool (non-transitive). In-house training levers (RL ×5,
distill, soups, 37.5k-decision data, 2025-champion distill) have all hit the SL ceiling at our
compute. The one demonstrated points-win was the WH bug fix. So the plan attacks what is **not**
ceiling-bound, ranked by expected ROI before the deadline.

## P0 — Ground truth (only the user can do)
- Deploy the **WH-fixed zip** (`deploy/caiest_cnn_bot.zip`, md5 `064a49cb…`) on the live bot — pure win.
- A/B **distill100b vs V1** on the real ladder. Even ~20 ranked games > any local gauntlet. This is
  the only non-proxy signal; everything below is judged against the finalist gauntlet until we get it.

## P1 — Correctness & robustness (guaranteed points, un-ceilinged)  ← highest ROI
The WH fix was 5×−30 in one sim. Hunt the rest with a **replay regression suite** over the 39k 2025
games + our own logs:
- Re-run every game through the bot in replay mode; flag any state where our action ≠ a legal/safe move,
  any phantom HU, any desync. (We already have `eval/test_wh_fix.py` / `eval/diff_bots.py` as the harness.)
- Targeted audits of penalty-prone rules: 抢杠和 (rob-kong), 海底/河底 (last-tile HU), 杠上开花,
  七对 edge cases, BUGANG response, fan-validation off-by-one, AnGang-vs-Gang in replay.
- Ops: cold-start TLE (measured 1.84s — ok, but watch), RSS near 512MB (measured 468MB — thin margin),
  the pure-NumPy fallback path actually triggering.
**Deliverable:** a green replay suite + any fix is a deploy candidate with zero ceiling risk.

## P2 — Inference-time mixture of the non-transitive models  ← exploits the finding
We have ≥3 diverse models (distill100b, V1, champ2025, + 2 sl2/dense if rebuilt) that beat each other
on different opponents. A mixture is more robust than any single one against a varied field.
- **v1 (cheap):** per-decision **logit averaging** across 2–3 models (all run in <6s on CPU). Decorrelated
  errors → fewer blunders. Build `deploy` flag `ENSEMBLE=model1,model2,...`.
- **v2:** per-game random selection from the pool (diversifies what opponents can model).
- Gate on the finalist gauntlet — a mixture that beats all 3 singletons is the first robust win.
**Deliverable:** ensemble bot, gauntlet net vs each singleton.

## P3 — Test-time search (the untried ceiling-breaker)
~6s/turn is mostly unused. Determinized shallow search at the discard decision:
sample opponent hands consistent with discards/melds → roll the policy a few plies → pick highest-EV
(duplicate-score) discard. Measure strength gain vs latency; cap to stay under budget.
**Deliverable:** `SEARCH=1` discard mode, gauntlet net + p95 turn time.

## P4 — Stronger on-distribution SL base (winning axis, slower payoff)
Retrain the base on **official (5.87M) + 2025-final (39k games)** combined, modern recipe
(`supervised_v2.py`, AdamW+cosine+suit-aug, val 0.875 proven), then the proven 600-step distill.
The 2025 games are the actual final distribution. Judge on the finalist gauntlet, NOT agreement.
**Deliverable:** `base2025` candidate; ship only if it beats distill100b on the on-distribution gauntlet.

## Sequencing to 06-14
- Now → 06-10: P0 deploy + ladder A/B running; P1 replay suite (fast, high-ROI); P2 ensemble v1.
- 06-10 → 06-12: P3 search prototype; P4 base train (background on free GPU); fold ladder signal in.
- 06-13: `lock_check.sh` cron → bake-off on the finalist gauntlet + ladder → lock the single strongest.
- 06-14: buffer. Freeze after final upload.

## Honest guardrails (from this campaign)
- Trust the **finalist gauntlet** (on-distribution) over self-vs-twin and over agreement metrics.
- A candidate ships only if it wins the **on-distribution** eval AND isn't worse on defense/ops.
- Small N + one wall-set lies (V1's +515 was wall-luck). Use ≥200 games on disjoint walls before claiming.
- Every number gets read back from a result file. This project has a documented history of proxy-metric
  self-deception; the discipline is the moat.

---
layout: default
title: "Engine Validation Feedback — Ludus MCR engine"
---

# Engine Validation Feedback — Ludus MCR engine

**Date:** 2026-07-12
**To:** the Ludus platform (botzone) developer
**From:** the mahjong-campaign side (owners of the test set + validator)
**Verdict up front:** the engine's rule content is **perfect** — 12,288/12,288 on every
rule-level check. One protocol field (`canHu`) was missing from its display events, which
made strict byte-exact replay fail; a fix exists on `fix/canhu-strict` and strict mode now
also passes 12,288/12,288. **Merge requested:**
[SuuTTT/ludus#1](https://github.com/SuuTTT/ludus/issues/1).

---

## 1. What was validated

The **Ludus pure-Python MCR engine** — specifically the validator adapter
`mahjong/validate_adapter.py` (`MyEngine`, the documented `reset(wall, quan, srand)` /
`step(responses)` interface) — was replayed against the published correctness oracle:

- **Test set:** [Dannibal/mcr-final2026-testset](https://huggingface.co/datasets/Dannibal/mcr-final2026-testset)
  — all **12,288 official IJCAI-2026 Final Stage-2 games** (full ordered walls, judge
  `srand`, the verbatim per-seat request/response protocol stream, expected terminal
  fan/score blocks), plus the 221-game golden edge-case subset.
- **Validator:** the stock `validate_engine.py` shipped with the dataset (stdlib-only;
  its `--self-test` passes 12,288/12,288). See also
  [Platform Developer Guide §5.5](PLATFORM_DEVELOPER_GUIDE.html) ("Acceptance Suite A").

The validator drives the engine turn by turn with the recorded responses and requires it
to reproduce the judge's event stream: every per-seat request, every display event
(draws, claim resolutions, gang discrimination), and the terminal fan/score block.

## 2. Finding

**Rule content: perfect.** On all 12,288 games the engine reproduced every claim
resolution (HU > PENG/GANG > CHI priority, multi-HU nearest-downstream), every gang
discrimination (AnGang/BuGang/qianggang), every fan calculation, and every four-seat
score — **12,288/12,288** on each of those checks. That includes the 221-game golden edge
subset. This is the hard part of an MCR engine, and it is right.

**Strict byte-exact mode: initially 0/12,288.** The official judge attaches a per-seat
win-eligibility array **`canHu[4]` to every display event**: a value ≥ 0 is the exact fan
total that seat could win with on the tile currently in flight (reported even below the
8-fan legal minimum), −3 means the seat cannot form a winning hand on it, −4 means not
applicable. `MyEngine` never emitted this field, so *every* display event differed
byte-for-byte from the judge's and *every* game failed strict comparison — while every
rule-level check passed.

Two things worth being direct about:

- **The original "byte-exact" claim had not been run in strict mode.** The engine's docs
  claimed byte-exact replay on the basis of the rule-level checks; strict mode was first
  actually executed during this validation, and it failed 0/12,288. The claim was made
  ahead of the measurement — exactly the failure mode the validator exists to catch.
- **`canHu` is not cosmetic — it is safety-critical for RL.** It is the per-step
  win-eligibility oracle: it is what tells a training environment whether a seat's Hu
  action is *legal* right now, and the winning event's `canHu[winner]` equals the terminal
  fan count. An RL environment built on an engine without it either recomputes win
  eligibility itself (and can get it wrong independently) or exposes an action space
  where illegal Hu is representable. Since the Ludus engine feeds `MahjongEnv`/PPO
  self-play, this field is load-bearing.

## 3. Fix — merge requested

Branch **`fix/canhu-strict`** (commit `1a9b231`, "validate_adapter: emit canHu on every
display event (strict validation 0/12288 → 12288/12288)") implements `canHu` emission by
reusing the engine's existing fan path (`calc_fan_raw`) — DRAW events check the drawer
(zimo), PLAY/PENG/CHI events check the three other seats for ron on the embedded discard,
with the judge's exact conventions (INIT all 0; DEAL/GANG/HU/HUANG all −4; wall-last and
4th-tile flags honoured).

**Strict re-verification on the unmodified validator now passes 12,288/12,288.** No
change to the validator, the test set, or any rule logic was needed — only the missing
field.

**Please merge `fix/canhu-strict` → `master`.** Tracking issue with details:
**<https://github.com/SuuTTT/ludus/issues/1>**.

## 4. Lessons for platform game implementations

Offered collegially — the engine is excellent, and this is precisely the process working
as designed:

1. **Run the strict validator before claiming exactness.** Rule-level passes and
   byte-exact passes are different theorems. The strict run costs minutes; an unbacked
   "byte-exact" in the docs costs trust and, downstream, silent divergence.
2. **Treat every oracle field as semantic until proven cosmetic.** `canHu` looked like
   display metadata; it is the win-legality oracle. The safe default for any field the
   judge emits on every event is: reproduce it, then argue about whether it matters.
3. **The validator + golden subset catch this class of bug in minutes.** 0/12,288 with
   all rule checks green is an unambiguous, localizable signal — the discrepancy diff
   pointed at the missing key immediately, and the 221-game golden subset alone would
   have shown it. Wire Acceptance Suite A (and the strict flag) into CI for every future
   game implementation.

**Bottom line:** with `fix/canhu-strict` merged, the Ludus MCR engine is, to the full
resolution of the official final corpus, a byte-exact replica of the IJCAI-2026 judge —
rule content *and* protocol stream — and is cleared as the substrate for the RL pilot.

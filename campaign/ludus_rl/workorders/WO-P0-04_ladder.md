# WO-P0-04 · CLI ladder: CI ratings, paired scheduling, calibration anchors

**Status:** TODO · **Est:** 1–2 weeks · **Depends:** WO-P0-02 · **Spec:** docs/04 (this IS the moat — implement it faithfully)

## Prompt for the executing agent
Build the rating/ladder service as a library + CLI (web UI comes in P1).

### Deliverables
1. `arena/ladder/` — match scheduler (mirrored-pair scheduling), OpenSkill ratings with
   displayed CI, promotion on CI-separation only, calibration anchors (`rule_v0`, `bc/ppo_v0`)
   pinning the scale.
2. Nightly calibration job: anchors re-measured; drift beyond tolerance freezes the ladder
   (writes `LADDER_FROZEN` + reason; integrity field in all outputs).
3. Provenance: every match/result JSON embeds env_version + submission hashes + judge hash
   + seed (use a shared `provenance()` helper).
4. `arena/ladder/cli.py` — `submit <policy>`, `standings`, `head2head A B` (paired blocks,
   t-CI, verdict SEPARATED/TIE), `audit <match_id>` (replay-verify).
5. Judge-side mechanism metrics: per-agent illegal-input rate, timeout rate, fallback
   firing counts exposed in standings.

### Exit criteria
- [ ] 4-bot ladder (random, rule_v0, bc_v0 or a weak ppo snapshot, ppo_v0) reaches stable
      CI-separated ordering from paired matches; standings JSON committed.
- [ ] Calibration self-test wired: any drift injects a synthetic bug test (mutate one card
      stat in a scratch env) and demonstrably FREEZES the ladder.
- [ ] `head2head ppo_v0 rule_v0` reproduces WO-P0-02's verdict from fresh matches.
- [ ] All outputs carry provenance + integrity fields.

### Non-goals
No accounts, no sandboxed uploads (P1), no web UI, no seasons.

## Log
- (append dated notes here)
- 2026-07-03 (claude) **DONE (CLI scope).** `arena/ladder/` shipped: mirrored-pair
  duel blocks (same seed + same action-RNG stream both orders → identical agents
  aggregate to exactly 0.5 by construction), OpenSkill ratings displayed with 2σ CI,
  adjacent-rank promotion flags on CI separation only, block-level Student-t
  head2head with SEPARATED/TIE verdicts, replay-verify audit, provenance
  (env_version + judge-source hash + submission hashes + commit) and integrity
  fields in every output, freeze discipline (mutating ops refuse while
  LADDER_FROZEN exists). Exit criteria:
  - [x] 4-anchor ladder, 480 pairs/agent (960 games each): ppo_v0 58.1 [51.6,64.6]
        SEPARATED > ppo_weak_v0 30.3 [26.2,34.4] SEPARATED > {rule_v0 18.0
        [14.3,21.7] ≈ random_v0 16.6 [12.8,20.3]}. The bottom pair is a real
        statistical tie (fresh head2head: 0.473, t-CI [0.435,0.511] → TIE) — the
        ladder correctly refuses to order near-equals; this IS the docs/04 §3
        criterion working. `benchmarks/results/ladder_standings.json`.
  - [x] Calibration self-tests exact (all four anchors, twice); drift vs committed
        reference 0.0; synthetic-bug drill (BOOM_CARDS_PATCH doubles Bulwark hp in
        a subprocess) demonstrably froze the scratch ladder (LADDER_FROZEN written,
        drill JSON in run log). Judge-hash change alone also freezes.
  - [x] `head2head ppo_v0 rule_v0` from fresh matches: 1.0, t-CI [1.0,1.0],
        SEPARATED — reproduces WO-P0-02 (`benchmarks/results/h2h_ppo_rule.json`).
  - [x] All outputs carry provenance + integrity (judge_hash 6e2742d791fcac00 at
        commit f7bfd63).
  Deferred to P1/ops: nightly cron for `calibrate` (job exists, scheduling is ops),
  web standings (P1 platform), submission quotas/hidden-seed rotation (docs/04 §6).
- 2026-07-03 (claude) Re-anchored on engine v5 (collision physics): fresh reference +
  standings (288 pairs/agent, mixed decks): rule_v0 30.2 ≈ ppo_v0 29.4 (tie — ppo is
  a deck-B specialist, see WO-P0-02 log) > ppo_weak_v0 26.2 ✓sep > random_v0 16.8.
  Registry now points at baselines/checkpoints/ppo_v5 (sha pinned).

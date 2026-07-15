# 04 · Evaluation & ladder spec — the moat

Distilled from a competition campaign that killed ~10 fake wins and 1 fake null.
Full narrative: the "Evaluation Infrastructure That Catches Its Own Lies" post
(suuttt.github.io) and the companion ToG paper draft.

## 1. Calibration traps
- Every game ships **calibration bots** (scripted / BC / RL) whose pairwise results are
  re-measured nightly; drift beyond tolerance freezes the ladder and pages ops.
- Self-test: a bot playing itself in duplicate mode MUST score the exact tie value by
  construction (e.g., 2.500 placement in 4-player duplicate; 0.5 in mirrored 1v1 pairs).
  Any deviation = harness bug, ladder frozen.

## 2. Paired / duplicate scheduling
- 1v1 games: matches are scheduled as **mirrored pairs** (same seed, sides swapped;
  same decks/spawn RNG mirrored). A "result" is the pair aggregate.
- N-player games: same seed, candidate rotated through every seat.
- Open-world score races (Craftax): same world seed for all entrants in a bracket.
- Effect: seat/wall/deck luck cancels; small true differences become measurable with
  100× fewer games.

## 3. Ratings with uncertainty
- Base: OpenSkill/TrueSkill per game. Displayed rating always carries its CI.
- Promotion/demotion and "beats X" badges trigger on **CI separation**, not point means.
- Head-to-head claims (e.g., tournament seeding) use block-level Student-t CIs over
  paired blocks, exactly as in the campaign gates.
- Rating anchors: calibration bots pin the scale so ratings are comparable across seasons.

## 4. Provenance & integrity
- Every submission stored with content hash; every match record embeds
  (env_version, submission hashes, judge script hash, seed).
- Aggregations assert expected counts and write an `integrity` field — partial results
  cannot masquerade as complete.
- Replays are append-only; outcome hashes verifiable by third parties (audit endpoint).

## 5. Mechanism-engagement metrics
- Judge-side overlays (timeouts, fallbacks for illegal actions, disconnect handling)
  expose firing rates per match. An agent whose "illegal-action fallback rate" is high is
  flagged — its rating reflects the fallback policy, not its intended policy.

## 6. Anti-cheat / anti-overfit
- Hidden seed pools rotated per season; public seeds for practice only.
- Weights-only tier: architecture registry prevents payload smuggling; code tier:
  no network, syscall allowlist, per-move budgets, memory caps (published, unlike Botzone).
- Ladder-probing throttles: submission quotas + rating provisional until N paired blocks.

## 7. Data schema (the product)
One canonical record for human AND agent decisions:
```
match_id, env_version, seed, tick, player_id(anon), is_human, obs_ref (tensor shard),
legal_mask_ref, action, action_latency_ms, outcome (appended at end), consent_flags
```
Consent-first: per-user toggle; anonymized exports; data dividend credits for opt-in.

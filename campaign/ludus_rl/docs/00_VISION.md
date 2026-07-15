# 00 · Vision

## One-liner
Agents compete, humans play the same games, training happens on-platform and is gamified —
and every match (human or agent) becomes consented, well-structured training data.

## The gap
Botzone proves the demand (universities + competitions still run on it) but is ~10-year-old
web: no live spectating, opaque judging, storage quotas, no data API, no training story,
fixed game list. Kaggle Simulations, CodinGame, Lux, Screeps each own a slice; none combine
**agent ladder + human play + on-platform training + developer games** in one loop.

## The thesis
Competitive game agents hit the **imitation/data ceiling**: past a point only better data
(especially human data) and better evaluation move the needle. A platform that owns the
human-play stream, the agent-match stream, and the training telemetry owns the three assets
that matter in the agent era. Labs currently scramble for exactly this.

## Flywheel
```
better games → more humans play → human data → better built-in agents/benchmarks
     ↑                                                        ↓
dev uploads games ← devs come for opponents+data ← agent ladder gets interesting
```

## Pillars
1. **Ladder & tournaments done right** — sandboxed (or weights-only) submissions; ratings
   with confidence intervals; calibration bots as fixed anchors; paired/duplicate match
   scheduling; replay for everything; one-click hosted competitions.
2. **Human play & data** — every game playable in-browser (humans are just another client);
   one canonical replay format shared by human play and agent training; consent-first ToS
   with a data dividend (credits for opted-in contributors).
3. **Training, gamified** — one-click BC-from-your-own-replays ("train an agent that plays
   like YOU", then watch it climb); PPO/league templates on rented GPU; quests, XP,
   curriculum ladders, public "hatchery" pages; trained-on-platform-only seasonal leagues.
4. **Developer games** — the game SDK is a pure-functional JAX state machine + a renderer
   component; review pipeline; revenue share.

## Monetization (ordered by realism)
1. Compute margin on on-platform training/judging (credits over spot GPU).
2. Hosted competitions for universities/labs/companies ($2–10k/event).
3. Data licensing of cleaned, consented human+agent corpora.
4. Pro tier (private leagues, submission quota, analytics API).
5. Later: game-dev rev share, sponsorships, cosmetics.

## Publications unlocked (the ToG line)
1. Platform/benchmark paper: GPU-vectorized Boom env + CI-honest paired ladder.
2. Imitation-ceiling-at-scale: BC-of-each-human vs the human's own rating, N thousand users.
3. Evaluation methodology: calibrated, paired, CI-honest ladders (9 best practices).
4. Seasonal dataset releases (balance patches = natural experiments).

## KPIs (alpha)
Weekly active agents on ladder · human matches/week · consented-replay volume ·
time from signup → first trained agent on ladder (target <1h) · tournament NPS.

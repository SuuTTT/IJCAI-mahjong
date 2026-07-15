# 07 · Platform parity plan (Botzone-class functionality, original implementation)

Goal: match the *functionality* of established university competition platforms,
then differentiate. All code, content, and design are original; the reference is
a feature checklist only.

## Feature checklist (reference platform surface -> Ludus status)
| Function | Reference behavior | Ludus status |
|---|---|---|
| Game list + rules pages | per-game wiki, sample code | PARTIAL: /cards, docs in repo -> need per-game pages with quickstart bots |
| Bot submission | upload code, versioned, multi-language | TODO (P1): weights-only tier first (msgpack + arch registry), container tier later |
| Auto ladder matches | continuous rated matches, Elo | PARTIAL: CI ladder (CLI, WO-P0-04) -> needs scheduler daemon + web standings (live /api/league exists) |
| Match records + replay viewer | list, filter, visual replay | DONE for Boom: /replays + watch mode + video render |
| Game tables (human vs bot / bot tests) | create table, pick opponents | PARTIAL: /play?bot=ppo|rule|random|gen:K; need bot-vs-bot spectate tables |
| Rankings | per-game + global user ranks | PARTIAL: /ladder page; needs user accounts to be meaningful |
| Groups / courses | homework, private contests | TODO (P1.5) |
| Contests | scheduled tournaments, brackets | TODO (P1): weekly automated tournament (PRIORITIES #7) |
| Notifications / announcements | site news | TODO (trivial once accounts exist) |
| Datasets | match data downloads | PARTIAL: replays are (seed, action_log) JSON; add export endpoint |

## Our differentiators (beyond parity)
- Weights-only submissions judged fully vmapped: thousands of rated matches in
  seconds, no sandbox risk (docs/01).
- Ratings with confidence intervals + calibration freeze (docs/04) — no
  unfalsifiable leaderboards.
- Generational self-play league with public lineage (/league).
- Consent-first human replay datasets (docs/04 §7).

## Implemented in this pass
- /api/league + /league lineage page (live curve, CI gates, fight-any-generation).
- /blog devlog page.
- Bot registry accepts league generations (?bot=gen:K).

## Next slice (P1 alpha)
1. sqlite accounts (name + token), per-user match history.
2. Weights upload endpoint -> validated vs arch registry -> auto ladder entry.
3. Ladder scheduler daemon (continuous paired matches, web standings).
4. Bot-vs-bot spectate tables.

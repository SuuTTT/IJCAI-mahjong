# Accounts, a living ladder, an SDK, and bots you can build without code

*Ludus devlog #3 — 2026-07-05*

## The night's build

Four features shipped and — the real headline — every one smoke-tested by
acting as a user before being called done:

1. **Accounts (alpha)**: claim a name at `/api/register`, get a token; uploads
   carry ownership. No emails, no passwords — competition-alpha tier.
2. **A living ladder**: a supervised daemon continuously picks the
   least-played pair from the rated pool (league champion, recent generations,
   every user upload, scripted anchors), plays seat-swapped mirrored blocks —
   the same CI-honest protocol as league gates — and publishes OpenSkill
   ratings. `/ladder` now renders live standings: rating = μ − 2σ, the
   pessimistic bound, because leaderboards should be claims, not vibes.
3. **`boom.sdk`**: the four-call surface from the Agent Challenge —
   `Env(seed).observe()/step()` plus `submit_check(path)`, which validates a
   submission exactly like the platform will: load it against the published
   architecture, play it, count illegal actions. Entrants find out locally,
   not after uploading.
4. **The Composer (Tier 3 v0)**: build a bot with zero code — an ordered rule
   list (IF elixir > 6 THEN play cheapest at left_bridge…, six board presets,
   five sensors) compiled server-side into a playable, table-able bot. The
   on-ramp for players who will never write Python — and composed bots make
   honest scripted anchors for the ladder.

## The smoke test that earned its keep

The platform owner set a rule this week: *nothing ships until the builder has
used it like a user.* Tonight's user journey — register, validate a checkpoint
with the SDK, upload it over HTTP, confirm it's fightable — failed on first
run: the form-parsing dependency for multipart uploads had never been
installed. The upload page had shipped earlier, looked fine, returned 200 on
GET, and could never have accepted a single file. Pages lie; journeys don't.

The journey now passes end-to-end, and its script lives in the repo as the
regression test for the whole submission pipeline.

## Meanwhile, on the GPU

The v2 league (elixir-overflow penalty + opponent-pool training) grinds
through its window. Early observation: fresh-initialization explorer
generations lose their first gates badly, as expected — their job is not to
win immediately but to seed the opponent pool with play styles the mirror
never produces. The bet is that resume-generations trained against that
diversity finally dethrone the long-reigning champion. The lineage page tells
the story either way, failures included.

## Next

- Ladder entry for composed bots and clones (the daemon currently rates
  checkpoint bots only).
- Capped equal-compute training slots — the Kaiwu-style fairness core.
- Per-user pages: your bots, your matches, your clone's rating curve.

*Everything above is in the repo, with the smoke logs. The platform:
one Vast 3090, a supervisor, and a growing pile of hard-won ops rules.*

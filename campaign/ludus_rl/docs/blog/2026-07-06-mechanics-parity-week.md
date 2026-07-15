# Mechanics-parity week: five engine versions, two new cards, and a replay archive

*Ludus devlog #7 — 2026-07-06*

## The loop that drives everything now

The owner plays, notices something un-CR, reports it — often with a replay
link and once with a full mechanics dossier. We reproduce it in the engine,
fix it, pin it with a test, bump the version. v11 → v15 in four days:

- **v12**: units deadlocked against tower footprints (radial push exactly
  canceled the march). Tangential slide → units path around towers.
- **v13**: knockback could strand ground units in the river (water rule now
  applies to spells) and heavies (mass ≥ 10) are knockback-immune, like CR.
- **v14**: the big one — **locks hold only while engaged**. Walking troops
  re-evaluate targets, which makes kiting, ice-golem pulls, and tank/support
  splitting real tactics. Found from a replay: a mega-minion ignored an
  interposed tank. Now it doesn't.
- **v15**: king-tower activation verified on ALL damage sources (spell chip
  included) and pinned; the king now visibly sleeps/wakes in 3D. Plus the two
  cards that weaponize activation: **Whirlgale** (tornado-family pull — drags
  even golems, buildings anchored) and **Harpooner** (fisherman-family hook —
  drags victims adjacent, cross-river hooks land on the bank).

42 pinning tests, all green. The mechanics-audit wiki page now embeds the
owner's full CR targeting/pathfinding + king-activation dossiers as the
reference contract.

## Replays became an archive

The "all my replays are gone" bug was a policy bug: the list silently hid
records from older engine versions — five bumps made history invisible.
Records were never lost (deterministic (env_version, seed, action_log) since
day one). Now: per-game tabs, full history with engine-drift badges, watch in
2D or full 3D (models/sound/projectiles), mp4 export, bit-exact JSON export,
and both players' hands + elixir rendered in replay — proper match review.

## Sound, self-made

MusicGen on our own 3090 for the battle loop; seven SFX synthesized from
numpy oscillators (CC0 by construction) after the free-sound route proved
JS-gated. Card thumps, bolt whooshes, tower-collapse bell, victory fanfare —
throttled so swarm fights stay musical.

## SMAX grew up too

Game #3 went from replay viewer to full RTS: interactive squads (click
orders, attack-move, control groups, minimap, SC2-style unit cards), MAPPO
baselines at 100% on 2s3z/3s5z, and the auto-reset gotcha documented in the
challenge doc after it cost us an evening.

## Next

Warpath (the RA2-class slice) engine skeleton; per-card sight ranges;
league-on-v15; SMAX hard-scenario training recipe; gomoku replay playback.

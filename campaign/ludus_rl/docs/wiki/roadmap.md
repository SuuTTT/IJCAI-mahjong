# 13 · The Ludus Roadmap — everything we're building, and where we are

*Written 2026-07-07 at the four-games milestone. This is the canonical plan;
kanban tracks the active slice. Progress bars are honest estimates of "done
vs the ambition", not of effort spent.*

**Vision**: a Botzone-class competition platform where one deterministic JAX
core per game powers everything — agents train on it at millions of steps/sec,
humans play the exact same engine in the browser, every match is a
bit-replayable record, and ratings ship with confidence intervals or not at
all. Original IP throughout; owner-generated or self-made assets only.

---

## A · Platform functions

### A1 Core loop `[█████████░] 90%`
Accounts, rated ladder (TrueSkill + CI, seat-swapped pairs, calibration
anchors), bot upload (weights tier + composer tier), spectate tables, replay
archive (full history, 2D/3D watch, mp4, bit-exact export), per-game
challenge docs, blog/wiki/kanban, SDK.
**Left**: container-tier bot sandbox (arbitrary code), rating decay policy,
account profiles/match history pages.

### A2 Tournaments & divisions `[██░░░░░░░░] 20%`
Weekly automated tournaments (fixed seed pools, bracket + round-robin),
per-game divisions (Gomoku swap2, Boom capped-compute slots), seasonal
resets with hall-of-fame. Design exists (docs/05); ladder machinery ready.
**Next**: scheduler + bracket runner + results pages; first weekly Boom cup.

### A3 Social & growth `[█░░░░░░░░░] 10%`
Public landing polish, per-user replay sharing links (exists), profile pages,
notifications, docs-site quality pass, demo videos (cinema pipeline exists).
**Next**: profile pages; embed-able replay player.

### A4 Multi-agent / API play `[████░░░░░░] 40%`
WS protocol solid across 4 games; SDK for Boom; SMAX/Warpath agent specs in
challenge docs.
**Left**: unified per-game arch registry for uploads, Python client SDK
covering all games, rate-limited public API tokens.

## B · Render & asset infrastructure

### B1 Universal scene schema + three.js renderer `[████████░░] 80%`
Proven across Boom (CR-tilt), SMAX (top-down), Warpath (RA2 iso): entities +
events + stage profiles, procedural-first models, GLB pipeline with fallback
chains, WebGL-failure resilience, headless screenshot verification loop.
**Left**: formal schema doc v1 freeze (docs/09 draft → spec), PixiJS 2D
backend for low-end devices, camera controls (pan/zoom) for RTS games.

### B2 Asset pipeline `[█████████░] 90%`
Manifest + provenance + audit CI, owner generation loop (2D/portraits/GLB
rigged), thumbnails, procedural synthesis (SFX), MusicGen BGM on own GPU,
rolling prompt queue (batch at 10 — currently 5/10).
**Left**: batch roll of pending queue; per-game asset packs for SMAX/Warpath
hero units (procedural is fine but hero-quality art elevates).

### B3 Cinematics `[███░░░░░░░] 30%`
mp4 render pipeline live for Boom replays (server-side).
**Left**: hand/elixir overlays in renders, SMAX/Warpath cinema adapters,
auto-highlight detection (tower falls, big fights) for shareable clips.

## C · Games

### C1 Boom (flagship) `[█████████░] 88%`
Engine v15: 42 pinning tests, deep CR parity (targeting, kiting, pulls, king
activation, knockback classes, charge, auras, spells), 63 cards, full
owner-generated asset pack (board/units/towers/projectiles/61 portraits/
audio), 3D+2D clients, dual-hand replays, generational league (champion gen
166, first v15-era monarch), My Clone BC, deck builder + composer.
**Left**:
- per-card sight ranges (last ⚠ audit row)
- stun-retarget nuance + splash shapes audit notes
- Whirlgale/Harpooner art (in pending queue) + deck-pool entry
- balance telemetry page (per-card winrates from replays)
- weekly cup (A2)

### C2 Gomoku `[██████░░░░] 60%`
Rapfi (Gomocup winner) as house bot black/white, freestyle rules page,
records exported.
**Left**: in-browser replay playback, swap2 division + forbidden-move
(Renju) engine variant, rating ladder vs Rapfi levels.

### C3 SMAX `[████████░░] 80%`
Interactive RTS play (SC2 squad UI: cards, minimap, control groups,
attack-move), replay viewer, per-scenario board (2s3z 100 · 3s5z 100 ·
3s5z_vs_3s6z 83 · 10m_vs_11m 30 · 5m_vs_6m 10), challenge doc, MAPPO
baselines.
**Left**: hard-scenario training recipe (curriculum/longer horizon/tuned
hypers) for the marine maps, weights-upload eval harness, self-play
two-sided division (JaxMARL supports it), hero art pack (optional).

### C4 Warpath (RA2-class) `[██████░░░░] 55%`
Deterministic RTS engine (economy/production/combat/prereq tech graph/power),
playable vs scripted commander, RA2 sidebar + selection + minimap, 6 engine
tests. Milestones M1-M7+M9 done.
**Left** (tutorial ladder):
- M8 fog of war + shroud (visibility per player, affects targeting)
- M10 replays into the shared archive + watch mode
- queued production, placement ghosts, footprints/collision, A* pathing
- naval/air lane, more units (artillery, engineer), defense structures
- PPO/league baseline + challenge doc + upload tier
- M11 lockstep MP (engine is already deterministic — protocol work)

### C5 Hold'em `[█░░░░░░░░░] 8%`
Design notes only. JAX-native HUNL engine, CFR/deep-CFR baseline, bankroll
ladder with BB/100 + CI. **Next after Warpath P1.**

### C6 Mahjong `[█░░░░░░░░░] 8%`
Chinese Standard engine port (owner has Botzone-hardened codebase + IJCAI
work to draw on), 4-player tables, duplicate-format scoring for variance
control.

### C7 Catalog breadth `[░░░░░░░░░░] 5%`
PGX classics (Go/chess/shogi) for instant depth, Craftax long-horizon
benchmark, HoK-class MOBA (hok_env study → own slice, the Kaiwu direction),
M&B-like tactics slice. Each rides the same platform machinery.

## D · Training & research

### D1 Boom league `[████████░░] 85%`
Generational league with promotion gates + CIs, PFSP-lite pool, explorer
gens, resume-adapt across engine versions (survived v8→v15 with zero
resets — gen 166 proves adaptation works), 8h window automation with quiet
night-watch monitoring.
**Left**: bigger-net league track (obs capacity already frozen at 96),
population-based hyperparameter exploration, public lineage visualization.

### D2 SMAX slate `[███████░░░] 70%`
MAPPO baselines across 5 scenarios; asym scenario strong.
**Left**: hard-map recipe; self-play division; upload eval.

### D3 Warpath RL `[░░░░░░░░░░] 0%`
Macro action space is agent-ready by design.
**Plan**: PPO vs scripted commander curriculum → self-play league (reuse D1
machinery wholesale) → the third ladder division.

### D4 Research outputs `[██░░░░░░░░] 15%`
Devlogs #1-7 published; honest-benchmarking practices baked in (CIs,
determinism contracts, pinning tests).
**Ambition**: a benchmark/platform paper once 5+ games have ladders + upload
tiers ("Ludus: deterministic multi-game agent benchmarking with replayable
evaluation"); per-game baseline reports.

## E · Ops & infrastructure

### E1 Serving `[████████░░] 80%`
Single 3090 box runs: 4 game servers, league training, SMAX slate, audio/gen
jobs (isolated venv), supervisor + Caddy + token auth.
**Left**: monitoring dashboard (GPU/queue/uptime), backup automation for
replays+checkpoints (HF mirror exists for models), second-box failover plan.

### E2 Determinism CI `[██████░░░░] 60%`
Pinning tests per engine (42 Boom + 6 Warpath), replay contracts.
**Left**: replay-checksum regression suite in CI (record golden replays per
env_version, verify bit-identity on every commit), cross-device (CPU vs GPU)
equivalence tests in CI.

---

## The near plan (next ~2 weeks, in order)

1. **Warpath M8+M10**: fog of war, replays in the archive, queued production
   + placement ghosts (the RA2 feel trio).
2. **Warpath PPO baseline** (D3) — reuses the Boom league harness; the
   scripted commander becomes the calibration anchor.
3. **SMAX hard-map recipe** — curriculum from 8m → 5m_vs_6m, 40M steps,
   ent-coef sweep; goal ≥60% on both marine maps.
4. **Boom sight-range parity** + balance telemetry page.
5. **Weekly Boom cup #1** (A2 scheduler + bracket page).
6. **Art roll** when pending queue hits 10 (5/10 now).
7. **Replay-checksum CI** (E2) — locks determinism forever.

## The quarter plan (through 2026-Q3)

- Hold'em engine + CFR baseline + bankroll ladder (C5).
- Mahjong engine port + tables (C6).
- Gomoku swap2 division + Renju variant (C2).
- Warpath lockstep MP + bigger maps + naval/air (C4).
- Container-tier bot uploads (A1) + public API tokens (A4).
- PGX classics catalog row (C7).
- Benchmark paper draft (D4).
- MOBA-slice design doc (C7, the Kaiwu direction) — engine spike only.

*Cadence unchanged: owner playtests → reports → repro → fix → pin → version.
Every milestone gets a devlog; every mechanic gets a test; every match gets
a replay.*

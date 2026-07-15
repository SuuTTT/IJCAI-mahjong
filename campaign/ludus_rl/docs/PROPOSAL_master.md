# AGENT ARENA — a modern Botzone: proposal & execution plan
**Date:** 2026-07-02 · **Owner:** SuuTTT · **Status:** PROPOSAL (independent of the IJCAI competition; executable by separate agent sessions/GPUs)
**One-liner:** a modern game-AI platform where developers upload agents (and games), humans play the same games, training happens on-platform and is itself gamified — and every match, human or agent, becomes a licensed, well-structured dataset.

---
## 1. Why this, why now

**The gap.** Botzone (the incumbent we know intimately from the IJCAI campaign) proves the demand — universities and competitions still run on it — but it is ~10-year-old web: no live spectating, cookie/login quirks, 256MiB quotas, opaque judging, no data API, no training story, a fixed game list, zero mobile. Kaggle Simulations, CodinGame, Lux, Screeps each own a slice; none combine **agent ladder + human play + on-platform training + developer-uploaded games** in one loop.

**The thesis (from our own research).** Our Mahjong campaign's central finding is that agents hit the **imitation/data ceiling**: past a point, only better *data* (especially human data) and better *evaluation* move the needle. A platform that owns the human-play stream, the agent-match stream, AND the training telemetry owns the three assets that matter for game AI in the 2026 agent era. Every LLM-agent lab currently scrambles for exactly this kind of interactive environment + demonstration data.

**The flywheel.**
```
better games → more humans play → human data → better built-in agents/benchmarks
     ↑                                                        ↓
dev uploads games ← devs come for opponents+data ← agent ladder gets interesting
```

## 2. Product pillars

### P1 — Agent ladder & tournaments (the Botzone core, done right)
- Upload agent (any language) → sandboxed match runner → ELO/TrueSkill ladder per game.
- **Evaluation methodology as a feature** (our moat, straight from the campaign): duplicate/paired evaluation (same seed/wall/deck rotated across seats), block confidence intervals on ladder deltas, calibration bots at known ratings, provenance hashes on every submitted binary, replay for every match. "Your rating has a CI" — no other platform does honest uncertainty.
- Tournament engine: Swiss, round-robin, duplicate formats; one-click "host a competition" for classes/labs/companies.

### P2 — Human play & data collection
- Every game is human-playable in the browser (same protocol as agents — humans are just another client).
- Humans can queue vs agents, vs humans, or into mixed ladders; agent-vs-human is a first-class mode ("can your bot beat its author?").
- **Every match is recorded in one canonical replay format** (obs, legal actions, chosen action, timing, outcome) — the SAME format agents train on. Consent baked into ToS + per-user data toggle; contributors can get credit/compute back for opting in (data dividend).

### P3 — On-platform training, gamified
- "Train" tab: pick env, pick algo template (BC from your own replays / PPO self-play / league), rent platform GPU or bring your own key; training curves render live in the UI.
- **Gamification of training itself:** quests ("reach 60% vs tier-3 bot", "beat your own imitation"), XP for curriculum stages, public "hatchery" pages where spectators watch an agent learn, seasonal leagues for trained-on-platform-only agents (levels the field, drives compute revenue).
- Killer loop unique to us: **"train an agent that plays like you"** — one-click BC on your own human replays, then watch yourself-the-bot climb the ladder. Emotionally sticky + generates paired human/imitation data (a research goldmine: imitation-ceiling studies at scale).

### P4 — Developer-uploaded games (the platform bet)
- Game SDK: a game = deterministic, seedable state machine implementing `reset(seed) / step(actions) / observe(player) / legal(player) / result()` + a renderer (web component) + a rules doc. Runs server-side headless for matches, compiled to WASM where possible for client replay/play.
- Review pipeline + revenue share for popular games. First-party games seed the catalog.

## 3. First flagship: "LaneClash" (Clash-Royale-like RTS micro)

**Why CR-like:** real-time (200ms ticks), imperfect info (hidden hand/cycle), resource economy (elixir), combinatorial deck-building, short matches (3 min), enormously legible to spectators, and *seriously underserved* as a research benchmark (existing work: ClashRoyaleBuildABot, KataCR screen-scrape the real game — no clean sim exists). A faithful open sim would become the reference environment overnight — that's a paper AND a community.

**IP note (important):** mechanics are not copyrightable, but names/art/characters are. Ship **original theme + assets** ("LaneClash": towers, lanes, elixir→"energy", original units with CR-inspired roles), no CR name in marketing. Same posture as OpenTTD/Mindustry. Get this reviewed before launch.

**Scope (v1):** 2 lanes + 2+1 towers, 60-card original set covering the CR archetype space (tank/swarm/splash/air/spell/building), fixed 8-card decks, deterministic tick sim at 5 ticks/s, full replay determinism from (seed, action log). Ship with 3 built-in bots (scripted / BC-from-playtests / PPO) as calibration anchors at known ratings.

**Research hooks from day one:** balanced-seed duplicate mode (both players get mirrored deck/elixir RNG — our duplicate-Mahjong trick, novel for RTS eval), per-tick obs tensors versioned for training, weekly "balance patch as natural experiment" datasets.

## 4. Architecture (concrete, boring-on-purpose)

| Layer | Choice | Notes |
|---|---|---|
| Game engine | **Rust core → WASM + native** | one impl for server matches, client play, and Python training bindings (PyO3); determinism enforced (integer/fixed-point math, no wall clock) |
| Agent runtime | Docker (nsjail/gVisor) for uploaded code; WASM fast-path for compiled agents | per-move CPU/mem/time budgets like Botzone but transparent + local-testable via the same image |
| Agent protocol | JSON/msgpack over stdio or WebSocket; obs schema versioned | identical for humans (browser client) and agents |
| Matchmaker/queue | Redis + workers | paired/duplicate scheduling built-in |
| Backend | FastAPI (or Go) + Postgres + S3 (replays) | replays are the crown jewels: append-only, hashed |
| Frontend | Next.js + PixiJS/WebGL replay+play canvas | live spectate = replay stream; mobile-usable |
| Training | K8s GPU pool; template repos (BC/PPO/league) | meter GPU-seconds → credits |
| Ratings | OpenSkill/TrueSkill + our CI layer | calibration bots as fixed reference points |

**Reuse from the Mahjong campaign:** the entire eval discipline (calibration traps, paired blocks, CIs, provenance, preflight, mechanism-engagement stats) ports directly into the ladder/judge; the numpy-parity habit becomes the WASM/native determinism test suite; the Mahjong game itself becomes catalog game #2 for free (we already have engine + agents + the official fan library integration).

## 5. Publications this unlocks (the ToG direction)

1. **Platform/benchmark paper (ToG or NeurIPS D&B):** "LaneClash + Arena: a paired-evaluation platform for real-time card-battler AI" — the environment, the duplicate-RTS eval protocol, calibration-anchored ladders, baseline agents. (Precedent: Lux, Griddly, MicroRTS papers.)
2. **Human-agent data study (ToG):** imitation ceiling at scale — N thousand humans, BC-of-each-human vs each human's own rating; directly generalizes our Mahjong §"agreement ≠ strength" finding.
3. **Evaluation methodology paper:** "Ratings with confidence: calibrated, paired, CI-honest ladders for game AI" — the 9 best practices, now with platform-scale evidence.
4. Ongoing: every season's balance patches + match corpus = citable dataset releases.

## 6. Monetization (ordered by realism)

1. **Compute margin** on on-platform training + tournament judging (charge credits, buy spot GPU — we already know the Vast.ai economics cold).
2. **Hosted competitions** (universities, labs, companies): white-label tournaments with our eval rigor — Botzone shows unis pay in adoption; labs pay in cash. $2–10k/event.
3. **Data licensing:** cleaned, consented human+agent corpora for agent-training labs (the hot market in 2026). This is the big one if traffic comes.
4. **Pro tier:** private leagues, more submissions/day, replay analytics, API.
5. Later: game-dev revenue share, sponsorships, cosmetics.

## 7. Phased plan (each phase = a work order another agent/GPU can execute)

**P0 — Engine + vertical slice (6–8 wks, 1–2 agents)**
- Rust LaneClash core (fixed-point, deterministic, replay = (seed, actions)); property tests: replay determinism, WASM≡native bit-exactness.
- Python bindings + gym-style env; scripted bot + PPO baseline trains to >90% vs scripted.
- Minimal web: play vs bot in browser (WASM), watch replays. CLI ladder w/ TrueSkill.
- EXIT: 3-min match playable in browser; 1k matches/hr/core headless; deterministic replays.

**P1 — Platform alpha (8 wks)**
- Accounts, agent upload (docker sandbox), per-game ladder w/ CI ratings + calibration bots, replay browser, human queue, data-consent flow + replay export API.
- Port Mahjong as game #2 (proves the SDK is real).
- EXIT: 50 invited users (start with Botzone/IJCAI community — we know them), first weekly tournament runs unattended.

**P2 — Training + gamification beta (8 wks)**
- BC-from-my-replays one-click; PPO template on rented GPU; quests/XP; "plays-like-you" ladder.
- First paper submission (platform/benchmark).
- EXIT: 20 agents trained fully on-platform; training→ladder loop under 1 hour.

**P3 — Open + monetize**
- Public launch, hosted-competition product, game SDK docs public, data-licensing pilot with 1 lab.

**Effort/cost ballpark:** P0–P2 is ~5–6 agent-months of focused build + $200–500/mo infra (one modest VPS + spot GPU on demand). No fine-tuned-model risk; all engineering.

## 8. Top risks & mitigations
| Risk | Mitigation |
|---|---|
| CR IP (biggest) | original assets/name/units from day 1; legal read before public launch; mechanics-only inspiration |
| Sandbox escape | gVisor/nsjail + no-network + syscall allowlist; WASM path preferred; bug bounty later |
| Cold start | seed with the community we already have (Botzone/IJCAI, uni courses); Mahjong port brings existing bot authors; calibration bots make a 10-user ladder still fun |
| Data privacy | consent-first ToS, per-user toggle, EU-style export/delete, anonymized releases |
| Scope creep | P0 exit criteria are contractual; CR-like FULL fidelity explicitly out of scope for v1 |
| Determinism drift (the silent killer) | our preflight discipline: bit-exactness tests in CI on every engine commit |

## 9. Immediate next actions (can start today, zero GPU)
1. Name/domain check ("LaneClash" + platform name TBD — 30 min).
2. Repo skeleton `arena-platform/` + `laneclash-engine/` with the P0 exit criteria as README + CI determinism test stubs (1 evening — good first work-order for another agent session).
3. LaneClash design doc: 60-card list w/ archetype coverage matrix + tick-sim spec (1–2 days; I can draft the spec from CR mechanics literature).
4. Legal sanity pass on the CR-like posture (async).
5. P0 work-order prompt written for the executing agent(s) (I draft it when you say go).

---
---
# V2 ADDENDUM (2026-07-03) — Boom Arena: JAX-first, env-as-game, costs

## A. Name & flagship
Platform + flagship game: **Boom Arena** (the CR-like battler = "Boom"). LaneClash naming dropped.

## B. ARCHITECTURE PIVOT: JAX-first (replaces Rust→WASM as the core bet)
User requirement: games must be **fast-RL-trainable** (MuJoCo Playground / JaxMARL / SMAX style). This is the right call for a platform whose product is *training* — and it resolves cleanly:

- **Game core = pure-functional JAX** (`reset/step/observe/legal` as jit-able pure fns, integer/fixed-point state, explicit PRNG keys). `vmap` → **thousands of parallel matches on ONE GPU** (SMAX shows 10k+ envs/GPU; Boom's tick sim is comparable complexity). Training loop = PureJaxRL/Mava style, whole rollout on-device.
- **Human play = server-authoritative**: the SAME JAX core runs server-side (CPU jit for a single match is microseconds/tick); browser is a thin PixiJS renderer over WebSocket. No WASM port needed, no dual implementation, replay = (seed, action log) exactly as before. 200ms ticks tolerate 50-80ms RTT fine.
- **Agents**: for training, users get the env directly (pip package, gymnasium + PettingZoo adapters). For ladder matches, sandboxed agent processes speak the same obs/action protocol; a JAX-native fast path lets pure-policy agents (params + apply_fn) be evaluated at 1000× speed server-side — **"submit weights, not code"** tier = cheap judging + no sandbox risk.
- Determinism preflight (bit-exactness across jit/no-jit, CPU/GPU, versions) stays in CI — JAX float nondeterminism is the known trap → keep game STATE in int32/fixed-point; floats only in observations.

## C. Catalog strategy: build 1, adopt many ("env-as-game")
Launch catalog = our flagship + curated existing JAX envs wrapped as ladder games (each gets: page, ladder w/ CI ratings, replays, human-play where sensible, calibration bots):
| Source | Games | Notes |
|---|---|---|
| **Ours** | Boom (CR-like) | flagship, human-playable, the paper |
| **JaxMARL** | SMAX (SC2-micro-like), Overcooked, Hanabi, MPE | instant credibility w/ MARL researchers; SMAX = the "jax starcraft" ask |
| **PGX** | Go/chess/shogi/backgammon family | classic board ladder |
| **Ours, cheap** | Texas Hold'em (JAX, trivial state), Chinese Standard Mahjong (port ours) | huge audiences, poker-AI scene; mahjong reuses campaign assets |
| **Benchmark-as-game** | gymnax/MuJoCo-Playground tasks as "sport" leaderboards | training IS the game; quests = curriculum |
License check per env (JaxMARL/PGX Apache-2.0 ✓). Contribution: our paired-duplicate eval wrapper upstreamed = free marketing.

## D. Genre roadmap for the wishlist (what each demands from the SDK)
| Family | Wishlist items | Plan |
|---|---|---|
| Real-time card battler | (CR) | **Boom v1 — P0** |
| Turn-based imperfect-info card | YGO-DuelLinks-like, Hold'em, CS Mahjong | **P1** — cheap engines, big audiences; YGO-like needs card-DSL design (do a 200-card original set later) |
| RTS micro | SC2, RA2 | **P1**: SMAX adoption covers micro; **P3**: "Boom RTS" = RA2-flavored macro-lite (base+economy+2D combat) on the Boom engine core |
| MOBA | Dota2, LoL, HoK | **P4**: start 1v1 mid-lite (HoK-1v1-style, ref: Tencent's Honor of Kings Arena env); 5v5 MOBA-lite only after MARL infra proven on SMAX |
| Open-world / action-RPG | GTA, M&B II | not ladder-able v1; long-term "scenario/agent-playground" category (LLM-agent quests, scripted scenario packs); explicitly out of scope P0-P3 |
Rule: every new family enters as the SMALLEST competitive slice (1 lane, 1v1, micro-only), never full fidelity.

## E. COST ESTIMATES

### E1. Prototype/dev on Vast.ai (now → P1)
Best practice: **Vast boxes are ephemeral (random ports, no static IP, can vanish) — never host state there.** Split:
| Component | Where | $ |
|---|---|---|
| Web/API/Postgres/redis (control plane) | cheap stable VPS (Hetzner CX32 / Contabo 4-core) | **$8–15/mo** |
| Replay/object storage | Cloudflare R2 (free egress) | ~$0–5/mo |
| GPU dev+training worker | Vast 3090/4090 interruptible, $0.15–0.35/hr, **on-demand only** (destroy-idle policy... rent-idle policy: STOP when idle) | 4h/day ≈ **$25–45/mo**; 24/7 burn ≈ $110–250/mo |
| Judging (match runner) | same VPS (CPU jit) until >10k matches/day | $0 |
| Domain/TLS/CDN | Cloudflare free | ~$10/yr |
**Prototype total: ~$40–70/mo typical; <$300/mo worst-case heavy month.** Connect VPS↔Vast via Tailscale (free tier) + job queue pull model (worker pulls from Redis — survives box churn; exactly our mahjong-campaign pattern).

### E2. Production (P2+): honest best practice
**Recommendation: do NOT move to AWS until revenue or a compliance/investor reason exists.** Interim prod = Hetzner dedicated (AX42 ~€46/mo: 8c/64GB) + R2 + Vast/RunPod GPU burst ≈ **$100–200/mo** for thousands of users — 5–10× cheaper than AWS equivalent.

**When AWS (P3+, hosted-competition customers / SLA):**
| Tier | Setup | Est./mo |
|---|---|---|
| Control plane | ECS Fargate (2 svc) + ALB + RDS Postgres (t4g.small→r6g) + ElastiCache + S3+CloudFront | **$180–320** |
| GPU judging | g6.xlarge SPOT ($0.35–0.45/hr) scale-to-zero via queue depth | $50–200 usage-based |
| Training rental (revenue side) | keep on Vast/RunPod even in prod (3–5× margin vs AWS GPU) OR AWS spot resell w/ markup | net-positive if priced right |
| Observability | CloudWatch basics + Grafana Cloud free | $20–40 |
**AWS prod floor ≈ $250–550/mo; scale linearly with match volume.** Best practices: everything behind a queue (scale-to-zero workers), spot+checkpointing for all GPU, S3 lifecycle → Glacier for old replays, Savings Plan only after 3 stable months, IaC (Terraform) from day 1 so Hetzner→AWS is a re-point not a rewrite, budgets+alerts at $100/$300 (rogue-GPU protection — we know this failure mode from fleet ops).

## F. Plan deltas
- P0 flagship build switches to **JAX Boom core** (+ PPO baseline via PureJaxRL; exit adds: ≥5k parallel envs on one 3090, >1M env-steps/s aggregate).
- P1 adds JaxMARL/PGX adoption + Hold'em + Mahjong port (catalog of 6+ at alpha).
- Publication #1 gains an angle: "Boom: a GPU-vectorized CR-like battler + CI-honest paired ladder" (env speed is itself a headline number, MuJoCo-Playground-style).

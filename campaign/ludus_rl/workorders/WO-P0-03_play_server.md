# WO-P0-03 · Server-authoritative play + browser client

**Status:** TODO · **Est:** 2–3 weeks · **Depends:** WO-P0-01 · **Spec:** docs/01

## Prompt for the executing agent
Make Boom humanly playable in a browser against a bot, with recorded replays.

### Deliverables
1. `arena/match_server.py` — FastAPI + WebSocket. Authoritative loop: steps the JAX core
   (CPU jit) at 5 ticks/s; receives player inputs `(card_slot, x, y)`; broadcasts
   state-frame diffs; enforces legality server-side (reject+count illegal inputs — no
   silent correction, per AGENTS.md §2).
2. `web/` — minimal Next.js app: lobby (pick opponent bot), PixiJS board renderer
   (placeholder shapes/sprites are fine — readable > pretty), card hand UI, energy bar,
   result screen, replay viewer (scrub bar; replays fetched by id).
3. Replay pipeline: every match writes `(env_version, seed, action_log, outcome_hash)` to
   storage; a CLI `arena/replay_verify.py` re-simulates and checks the hash.
4. Bot adapter: `rule_v0`/`ppo_v0` (from WO-P0-02) callable in the server loop.

### Exit criteria
- [ ] A human can complete a full 3-min match vs `rule_v0` in Chrome with no desync;
      input→effect latency <150 ms on localhost.
- [ ] 20 concurrent matches on a 4-core VPS-class machine without tick overruns
      (load test script committed).
- [ ] Every match auto-saves a replay; `replay_verify` passes on 100/100 matches.
- [ ] Human matches produce decision records in the docs/04 §7 schema.

### Non-goals
No accounts/auth (P1), no matchmaking, no human-vs-human yet, no art pass, no mobile.

## Log
- (append dated notes here)

# 01 · Architecture — JAX-first, server-authoritative

## Core decision
**Game cores are pure-functional JAX.** Rationale: the platform's product is *training*;
a jit/vmap core gives thousands of parallel matches per consumer GPU (SMAX /
MuJoCo-Playground class). One implementation serves training, judging, human play, and
replays — no dual-engine drift.

```
             ┌────────────────────────────────────────────┐
             │  boom/ (JAX game core, pure functions)     │
             │  reset(key) → state                        │
             │  step(state, actions, key) → state         │
             │  observe(state, player) → obs (versioned)  │
             │  legal(state, player) → mask               │
             │  result(state) → outcome | None            │
             └──────┬───────────────┬───────────────┬─────┘
        vmap (GPU)  │        single-match CPU jit   │ same fns
      ┌─────────────▼──┐    ┌───────▼────────┐  ┌───▼─────────┐
      │ training envs  │    │ match runner   │  │ replay      │
      │ (pip package,  │    │ (judge, paired │  │ verifier    │
      │ gym/PettingZoo)│    │ scheduling)    │  │ (CI + audit)│
      └────────────────┘    └───────┬────────┘  └─────────────┘
                                    │ WebSocket (obs→, actions←)
                          ┌─────────▼─────────┐
                          │ browser client    │  humans = just another client
                          │ (PixiJS renderer) │  agents = same protocol
                          └───────────────────┘
```

## Determinism rules (enforced by CI — see AGENTS.md §1)
- State in int32/fixed-point (e.g., positions in 1/256 tile units, HP integer).
- Explicit PRNG keys; fold_in per subsystem; no wall clock.
- Replay = `(env_version, seed, action_log)`; outcome hash recorded; CI replays matches.
- Observations may be float; the state trajectory may not.

## Human play = server-authoritative
The same core steps server-side (CPU jit: µs/tick; 200 ms game ticks). Client renders
interpolated state and sends inputs. Benefits: no cheating, no WASM port, replays and human
matches share the agent data schema bit-for-bit. Latency budget 50–80 ms RTT is fine at
5 ticks/s.

## Agent submission tiers
1. **Weights-only (preferred):** submit params for a published policy architecture
   (flax module registry). Judged fully vmapped server-side — 1000× cheaper, zero sandbox
   risk, instant high-volume paired evaluation.
2. **Containerized code:** any language, stdio/WebSocket JSON protocol, gVisor/nsjail
   sandbox, per-move CPU/mem/time budgets, no network. Local-testable via the same image.

## Platform services (control plane)
| Service | Choice | Notes |
|---|---|---|
| API | FastAPI + Postgres | accounts, games, submissions, ratings |
| Queue | Redis (streams) | match jobs, training jobs; workers PULL (survive box churn) |
| Storage | R2/S3 | replays append-only + hashed; obs-tensor shards for training |
| Ratings | OpenSkill + CI layer | see docs/04 |
| Web | Next.js + PixiJS | play, spectate (replay stream), ladder, hatchery |
| Training workers | rented GPU (Vast) | pull queue; checkpoint to storage; killable anytime |
| Observability | Prometheus/Grafana + billing alarms | rogue-GPU protection |

## Game SDK (developer-facing)
A game = a Python package exporting the five pure functions + `render_frame(state) → dict`
(consumed by a generic PixiJS scene) + `meta.yaml` (players, tick rate, obs schema version,
calibration bots). Acceptance = determinism suite passes + review.

## Anti-goals (v1)
No 3D, no client-side simulation, no custom netcode beyond WebSocket, no blockchain,
no full-fidelity clones of commercial games.

# PRIORITIES — ordered backlog (top = take next)

Status: TODO | CLAIMED(agent,date) | DONE(date, evidence-link)

## P0 — Boom vertical slice (target: 6–8 weeks)
| # | Work order | Status | Depends |
|---|---|---|---|
| 1 | `WO-P0-01_boom_engine.md` — JAX Boom core (deterministic, vmapped) | DONE(2026-07-03, benchmarks/results/{throughput_v1,sanity_10k}.json @85bd3d1) | — |
| 2 | `WO-P0-02_baselines.md` — scripted bot + PPO baseline (PureJaxRL) | DONE(2026-07-03, benchmarks/results/eval_v2_*.json + baselines/checkpoints/ppo_v2) | 1 |
| 3 | `WO-P0-03_play_server.md` — server-authoritative play + PixiJS client | TODO | 1 |
| 4 | `WO-P0-04_ladder.md` — CLI ladder: TrueSkill+CI, paired scheduling, calibration bots | DONE(2026-07-03, benchmarks/results/ladder_standings.json + h2h_*.json) | 2 |

**P0 exit (all):** a stranger can `pip install`, train a PPO agent in <1h on one 3090,
play Boom in a browser vs that agent, and see a CI-rated 4-bot ladder from reproducible replays.

## P1 — Catalog + platform alpha (8 weeks)
| # | Work order | Status | Depends |
|---|---|---|---|
| 5 | `WO-P1-05_catalog.md` — adopt SMAX(JaxMARL), PGX, Craftax; build JAX Hold'em; port CS Mahjong | TODO | 4 |
| 6 | `WO-P1-06_platform.md` — accounts, agent upload (sandbox + weights-only tier), web ladder, replay browser, consent-first data export | TODO | 3,4 |
| 7 | First weekly automated tournament (invite Botzone/uni community) | TODO | 6 |

## P2 — Training + gamification beta (8 weeks)
| 8 | One-click BC-from-my-replays; PPO template on rented GPU; quests/XP; "plays-like-you" ladder | TODO | 6 |
| 9 | Paper #1 submission: Boom env + CI-honest paired ladder (ToG / NeurIPS D&B) | TODO | 5,8 |

## P3 — Genre expansion
| 10 | Boom-RTS macro-lite (RA2-flavored: economy + base + 2D combat on Boom engine) | TODO | 2 |
| 11 | `docs/06` Hero-Commander battle layer (M&B/MOBA/RTS hybrid, 1v1, JAX) | TODO | 2 |
| 12 | Hosted-competition product (white-label tournaments) | TODO | 7 |

## P4 — Long horizon
| 13 | Hero-Commander conquest campaign (persistent async meta-layer = the open-world loop) | TODO | 11 |
| 14 | MOBA 5v5-lite (only after SMAX-scale MARL proven on platform) | TODO | 5,11 |
| 15 | 3D Minecraft scenario category (MineRL/MineDojo showcases, LLM-agent quests) | TODO | 6 |
| 16 | Data-licensing pilot + monetization hardening | TODO | 7,8 |

## Standing rules
- Every new genre enters as its SMALLEST competitive slice (1 lane, 1v1, micro-only).
- Nothing ships to the ladder without the determinism suite + calibration bots.
- Original IP only — see README license posture.

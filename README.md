# IJCAI-2026 Chinese-Standard-Mahjong

An honest campaign to build a competitive Chinese-Standard-Mahjong (MCR) bot for the IJCAI-2026 / Botzone
contest — and, increasingly, a research artifact about **why in-house evaluation lies** in imperfect-
information games.

> **Phase-1 status (2026-06-14):** Submission is `lad_chunjiandu` + the net-PIMC bot `[Claude]aaa`.
> Simulation-8 result: **27/38** (net +2.39/g). ~18 modeling interventions tried; the only demonstrable
> win across the campaign was a bug fix. See the **[Phase-1 Autopsy](docs/phase1_autopsy.html)** and
> **[Findings](docs/FINDINGS_2026-06-14.md)**.

---

## Start here

| Doc | What |
|-----|------|
| [docs/phase1_autopsy.html](docs/phase1_autopsy.html) | **Phase-1 post-mortem** — what we built, what died, 3 lessons, the plan |
| [docs/index.html](docs/index.html) | "Chasing the Ceiling" — the campaign log / blog |
| [docs/FINDINGS_2026-06-14.md](docs/FINDINGS_2026-06-14.md) | Latest findings: the scoring bug, RL infeasibility, strong-teacher distill null |
| [paper/TOG_SKELETON.md](paper/TOG_SKELETON.md) | ToG paper skeleton (the evaluation-gap thesis) |
| [paper/PAPER_PLAN.md](paper/PAPER_PLAN.md) | Concrete, submit-oriented paper plan + what's done/TBD |
| [CHANGELOG.md](CHANGELOG.md) | Chronological record |
| [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md) | R1–R6 forward research items |

---

## The deployed bot

- **Model:** `lad_chunjiandu` — 40-block ResNet (128 ch), distilled from the single strongest coherent
  ladder teacher (`chunjiandu`), 12× suit-augmented. The lock.
  - `deploy/ship/cnn_lad_chunjiandu.pkl` (fused, deploy) · `.npz` (weights) · md5 `d517e6a9`
  - Safe-floor fallback: `deploy/ship/cnn_distill100b.pkl` (the proven multi-teacher BC floor)
- **Runtime:** `deploy/caiest_cnn/` — pure-numpy **net-PIMC** search (no torch; 139 MB, fits Botzone's
  512 MB / ~6 s envelope). Auto-enables via `model.cfg` → `.npz`. Verifiable via a debug `[md5]` line.
- **Submission packaging:** `deploy/ship/bot_lad_chunjiandu.zip`, `bot_distill100b.zip` (shared-Storage
  A/B: each zip picks its model via `model.cfg`). Build: `bot/make_submit.sh`.
- **Deploy constraints:** Botzone py3.6 / torch1.4 / ≤512 MB / ~6 s → fused BN-free + legacy serialization,
  or pure-numpy. Only the Storage `cnn.pkl` swaps between bots.

---

## Infrastructure we built (the durable part)

### Self-play & search
- `train/jax_env/` — **GPU-vectorized JAX self-play env**. Agari via per-suit feasibility tables
  (`agari_jax.py`, `build_agari_tables.py`), terminal MCR scoring (`fan_reward.py`), win-aware step
  (`csm_selfplay.py`). **Validated 0-mismatch vs MahjongGB** (20k+ hands). ~589k games/s.
  - `resnet_jax.py` — JAX forward of the deploy ResFused-40 (warm-start; byte-exact vs numpy, argmax 16/16)
  - `obs38.py` — the 38-plane CAIEST observation, byte-exact vs `feature.py`
  - `train_ppo_ws.py` — warm-started PPO (the Tjong path; **infeasible at 40 blocks**, see findings)
- `deploy/caiest_cnn/csm_rollout.py`, `pimc_search.py`, `determinize.py`, `numpy_resfused.py` —
  the pure-numpy net-PIMC deploy search.

### Evaluation (the trustworthy yardstick)
- `eval/bench_vs_bot.py`, `run_match_kr.py`, `run_match.py` — **persistent-bot duplicate-format gauntlet**
  (2v2 rotated, same walls, thread-reader IPC, played/stuck validity gate). Needs `BOTZONE_JSON=0`.
- `eval/duplicate_eval.py`, `run_gauntlet.py`, `gate_candidate.py` — drivers.
- **Official C++ judge** — rebuild recipe (see CHANGELOG 2026-06-14): `git clone ailab-pku/Chinese-Standard-Mahjong`,
  `g++ -O2 -std=c++14 -D_BOTZONE_ONLINE -Iinc -Igbinc -I/usr/include/jsoncpp main.cpp -lboost_system -ljsoncpp -o judge`
  (jsoncpp shim + `MahjongGB`→`mahjong-algorithm` symlink).
- `eval/replay_harness.py`, `replay_audit.py` — replay Botzone's full-history path (found PHANTOM_HU=0).
- `tools/ladder_report.py`, `tools/pull_claude_ab.py`, the hourly collector — real-field telemetry.

### Training / distillation toolkit
- `train/caiest_repro/distill_kl.py` — KL-leashed BC + **AWBC** (`--aw`, advantage-weighted by duplicate
  score); `--student-blocks` for cross-size distill.
- `extract_top30.py` — decision extraction, `--player` (comma-list), `--since` (ObjectId date filter),
  `--scores` (per-seat final score for AWBC).
- `fuse2bn.py` / `bn2fuse.py` — exact fused↔non-fused conversion (RL needs non-fused; deploy needs fused).
- `value_head.py`, `q_head.py`, `deploy/caiest_cnn/value_search.py` — value/Q reranking (opt-in `CAIEST_VNET`/`CAIEST_QNET`).
- `rl_league.py`, `rl_curriculum.py` — PFSP self-play RL league.

---

## Data assets

| Path | Contents |
|------|----------|
| `others/sim8_*` | Full Simulation-8 duplicate sets for the field (incl. the 5 strong bots: `mythos`, `aidenh/hhhhhhhhh`, `infunus/TypeC青雀`, `xxxxltt/dl_v3`, `小试强化`) |
| `others/global_mythos_aidenh_typec_qingque/` | 460 global games for TypeC青雀 |
| `others/strong5.npz` / `strong5_full.npz` | 8,888 / 24,401 strong-teacher decisions (AWBC scores) |
| `others/typec_full.npz` / `mythos_full.npz` | 7,733 / 4,104 single-coherent-teacher decisions |
| `data/agents2025/` | 16 per-agent BC npzs from the 2025 final (on-distribution gauntlet) |
| `deploy/incoming/gauntlet2025/imit_*.pkl` | 6 BC imitations of 2025 finalists (eval opponents) |
| `paper/evidence/` | Every traceable bench log behind the paper's numbers |

**Field ranking (SIM-8 duplicate net/game):** `[Claude]aaa` +2.39 · our teacher `chunjiandu` +5.16 ·
strongest non-LLM `mythos` +9.73 / `TypeC青雀` +8.02 / `hhhhhhhhh` +7.82 · LLM-API bots `kimi_k2` +8.63,
`gpt_5_mini`, `glm`, `opus` (not clonable).

---

## Results summary (vs `lad_chunjiandu`, 144-game duplicate)

| Lever | Verdict |
|-------|---------|
| SL distill from coherent strong teacher | **WON** — *is* `lad_chunjiandu` |
| Self-play RL (league/curriculum) | NULL ×4+ |
| Warm-started self-play RL (full 40-block net) | INFEASIBLE (~50 min/iter, measured) |
| Value-of-state search rerank | +57 (inside noise) |
| Q-rerank / AWBC / champion clone / soups / 8-fan mask / defense | NULL ×8+ |
| Strong-teacher distill (best: TypeC β0.3) | NULL (−24, ties; bigger data didn't help) |
| **Net bug fixes** | the only demonstrable point gains |

Noise floor: ±537 net / 144 games for *identical* bots. Most "effects" live inside it.

---

## Reproduce / re-run

- **Gauntlet a candidate:** ship `deploy/caiest_cnn/` + the candidate `.pkl` + the judge to a box; run
  `eval/bench_vs_bot.py "<cand cmd>" "<base cmd>" 144 cand lad` with `CAIEST_PIMC=0` for plain-net,
  `MAHJONG_JUDGE=…/judge`, `BOTZONE_JSON=0`, `PYTHONPATH=<base>`. (Needs `eval/__init__.py`,
  `data/__init__.py`, `data/log_collector.py`.)
- **Distill:** `distill_kl.py --base cnn_lad_chunjiandu.pkl --champ <teacher>.npz --aw --beta 0.5 --steps 700 --out cand.pkl`.
- **Extract teacher data:** `extract_top30.py --root others --player "[author]bot" --scores --out t.npz`.
- **GPU notes:** see `CHANGELOG.md` for the JAX cuDNN/numpy pin and the never-mix-torch+jax lesson.

---

## License / provenance
Official assets (judge, fan-calculator, sample bot) are from
[ailab-pku/Chinese-Standard-Mahjong](https://github.com/ailab-pku/Chinese-Standard-Mahjong) (license
unspecified upstream — use with care). `PyMahjongGB` is MIT. Our code is for the contest + the companion
research paper.

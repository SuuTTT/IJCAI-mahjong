# IJCAI-2026 Mahjong Campaign — Backup Manifest

This repository is the **code + results** backup for the IJCAI-2026 MCR Mahjong
competition campaign (entry: **kdens3**, a 3×KD-student mean-softmax ensemble).
Large binaries (checkpoints, corpora, self-play data) live on Hugging Face — see
links below. This manifest is the index of *what lives where* so the project is
fully recoverable.

Last updated: 2026-07-15.

## Where everything is

| Artifact | Location |
|---|---|
| **Code + result JSONs** | GitHub: `SuuTTT/IJCAI-mahjong` (this repo) |
| **Trained models / checkpoints** | HF model repo: `Dannibal/ijcai-mahjong-ckpts-2026` |
| **Training corpora** | HF dataset: `Dannibal/ijcai-mahjong-corpora-2026` |
| **12,288-game evaluation testset** | HF dataset: `Dannibal/mcr-final2026-testset` |

- GitHub: https://github.com/SuuTTT/IJCAI-mahjong
- HF models: https://huggingface.co/Dannibal/ijcai-mahjong-ckpts-2026
- HF corpora: https://huggingface.co/datasets/Dannibal/ijcai-mahjong-corpora-2026
- HF testset: https://huggingface.co/datasets/Dannibal/mcr-final2026-testset

## Repository layout (code + results)

```
train/caiest_repro/      Main experiment code (~225 .py, 45 .sh) + ALL result JSONs
  ├── *.py *.sh          trainers, gating harnesses, aggregators, deploy builders
  ├── results/           427 verdict JSONs (PIMC gates, sweeps, value/fold/dealin,
  │                       teacher-curve, RL pilot gates) — the paper's evidence
  ├── kd_blocks/         per-block placement-point JSONs for kdens gating
  ├── ckpt/**/*.json     checkpoint manifests + traininfo (weights on HF)
  └── audit_final/ seproper_gate/ fold_blocks/ sampeff_gate/  gate outputs
campaign/                Cross-domain + RL infrastructure (our-authored code)
  ├── ludus_rl/          JAX RL env + league trainer
  │   └── baselines/     mahjong_t2_jax*.py (league trainer), ppo, pool_eval, ...
  ├── rl_sweep/          RL KL/entropy sweep drivers + jsonl summaries (msgpack on box)
  ├── rl_league/         League configs + seedN_jax_results.jsonl (policies on HF)
  ├── poker_domain/      Cross-game generality: poker distillation results
  ├── othello_domain/    Cross-game generality: Othello depth sweep results
  ├── crossgame/doudizhu/ Cross-game generality: Doudizhu results
  ├── synth_coherence/   Synthetic coherence-cell ablations
  ├── e1_cifarn/ e2_chess/ e3_robomimic/ e4_rldistill/  Domain experiment code+results
  ├── final2_harvest/    Final-2 corpus build + analysis code (corpora → HF)
  └── mcr_champion/       Champion numpy inference (feature.py, numpy_resfused.py)
```

## Models on Hugging Face (`Dannibal/ijcai-mahjong-ckpts-2026`)

See `MODEL_CARD.md` in the HF repo for load instructions and metrics. Summary:

| Path | What | Notes |
|---|---|---|
| `champion/kdens_s{0,1,2}_fp16.npz` | **DEPLOYED CHAMPION** kdens3 | fp16 storage, fp32 compute, numpy-only. Load with `champion/numpy_resfused.py` + `feature.py` |
| `ckpt/kd/` | 6 KD students (128×40 resnet) `.pkl`+`.bn.pkl` | source nets for the champion ensemble |
| `ckpt/aug/` | 14 aug baselines (`aug_s0` = the A/B comparison anchor) | |
| `ckpt/placeval/` | placement+value heads `placeval_s0..2` | |
| `ckpt/value/` | value critic ensemble `VALUE_C_60K(_s1..s7)`, `value_e2e_ckpt` | |
| `ckpt/oppbelief/` | opponent-belief models (incl. `_more60k`, `_big`) | |
| `ckpt/dealin/ dealin_pc/ dealin_pc_v2/` | deal-in predictors v1 + pc + v2 | for coherent-fold defense |
| `ckpt/kdcurve/ kd10/ paperx/ danger/` | teacher-count-curve + ablation nets | |
| `rl_league/<L>/latest.msgpack` | representative RL league policies | opponent pools NOT backed up (regenerable) |

## Corpora on Hugging Face (`datasets/Dannibal/ijcai-mahjong-corpora-2026`)

| File | What |
|---|---|
| `cooked_single.npz` (184M) | base BC training corpus (2025 agents) |
| `cooked_quarter.npz` (59M) | quarter-size subset |
| `final2_cai_corpus.npz` (23M) | final-2 CAI (champion-imitation) corpus |
| `final2_bc_corpus.npz` (30M) | final-2 BC corpus |
| `final2_all.jsonl.gz` (46M) | raw final-2 game replays |
| `final2_games_summary.jsonl.gz` | per-game summaries |

## NOT backed up (regenerable) — how to recreate

| Data | Size | How to regenerate |
|---|---|---|
| `caiest_repro/data/oppbelief/{full,full2}/*.npz` | 4.3G | self-play shards for belief training; regenerate by running the belief self-play generator in `train/caiest_repro` against the aug/kd nets |
| `caiest_repro/data/oppdealin/full/*.npz` | 1.8G | deal-in self-play shards; same pipeline, deal-in label target |
| `rl_league/*/pool.pkl` | ~1.1G each | league opponent pools; rebuilt during league training from `latest.msgpack` policies + `mahjong_t2_jax*.py` |
| `rl_sweep/*/`msgpack | ~170M each | RL sweep policy snapshots; rerun sweep drivers |
| domain venvs, `__pycache__`, logs, `.html` scrapes | — | ephemeral |
| `ckpt/aug/` full 1.5G, dealin smoke, kd10 | partial | aug s0–s13 + all listed nets are on HF; smoke variants dropped |

## Headline results (evidence JSONs in `train/caiest_repro/results/`)

- kdens3 vs aug_s0: **2.5054** (ci_lo 2.5012) + replication 2.5057 (ci_lo 2.5018). DEPLOYED.
- Nothing beats kdens3 across ~200k paired games (kdens6/mix6/kd10/kd14/kdT4ens all n.s.).
- Teacher-count curve FLAT (N=1..14): the distillation *operator*, not committee size, is the mechanism.
- Value-guidance E14: genuine null. Rule-overlay defense loses; coherent-fold trades deal-ins for points.
- Real-field A/B (325 matches): fold deal-in 25.7% vs plain 29.1%. Eval-wall: real-field deal-in 16.6%/game vs 2–3% in-house.
- Sim-11: kdens3 official 2nd of 25.

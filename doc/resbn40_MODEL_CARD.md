# MODEL CARD — resbn40 (the base architecture behind moyu) · IJCAI-2026 MCR

> **Purpose:** durable provenance for the **resbn40** architecture — the project's
> definitive policy network and the base moyu (`lad_chunjiandu`) is built on. Pairs with
> `moyu_MODEL_CARD.md` (the deployed model) and the repo `github.com/SuuTTT/IJCAI-mahjong`.
> Written/verified 2026-06-22.

## 1. Architecture
- **resbn40** = ResBNCNN: 38 tile-feature planes over a **(38,4,9)** tile grid → 3-conv stem
  → **40 residual bottleneck blocks, 128 channels, with BatchNorm** → FC head to the 235-action
  space. **~14.3M params** (moyu 14,362,172; our reproduction 14,330,987 — same arch).
- BatchNorm is load-bearing: it fixes the divergence that killed the un-normalized 32-block net.
- Input encoder: `feature.py` — (38,4,9) planes + legal-action mask; **HU fan-gated** (only offered
  when fan ≥ 8) so play is legal by construction.

## 2. Why resbn40 (the architecture search — `docs/ARCHITECTURES.md`, VERIFIED)
The project's headline: **representation mattered far more than RL/algorithm tuning** (a plain SL
CNN beat the most-tuned MLP 0-of-60 through the official judge). The search converged on resbn40:

| arch | val-acc | strength (head-to-head) |
|---|---|---|
| **resbn40 (128×40 BN)** | **0.894** | the champion base; beats 16-block CNN **+973 (52–25)** |
| 16-block CNN (`base_16x128`) | 0.863 | was the deployed model; +2826 vs old MLP r18 |
| resbn24 (24 blk) | 0.890 | **ties** resbn40 (+90) — lighter, good deploy alt |
| resbn56 (56 blk) | 0.894 | **ties** (+128, 30–29) — deeper does NOT help |
| resbnw192 (192ch×24) | 0.895 | −71 (~tie, slightly below) |
| wide_16x256 (256 ch) | 0.830 | **no better, far more params** |
| cnn_attn hybrid | 0.881 | ties (+109) |
| attn transformer (d192) | **0.897** (highest) | **ties-or-below at play** (val-acc≠play) |
| GNN | 0.76 | clearly worse |
| attnbig (d256) / deep_32x128 (no norm) | 0.38 / 0.23 | failed / diverged |
| flat MLP (r18 lineage) | — | lost 0-of-60 to a 1-epoch CNN; DEPRECATED |

**Verdict: resbn40 (40×128) is the definitive sweet spot.** Wider/deeper/attention/GNN all
tie-or-below. KEY LESSON: **val-acc ≠ play strength** (the transformer has the highest val-acc yet
doesn't win more) → the bottleneck is the imitation/data ceiling, not the network. The remaining
lever after architecture is distill/RL finishing, not more arch search.

## 3. Training recipe (base resbn40)
- **Data:** official 98k-game Botzone MCR dataset (`data.txt`, 98,209 rounds — public; archived as
  `data.zip` on HF `Dannibal/ijcai-mahjong-moyu-binaries-public`). Cook with the repo's official
  cooker (`parse_official_caiest.py`/`preprocess_single.py`, all 4 players) → **~5.1M decisions**
  (obs (38,4,9) uint8 / mask (235) bool / act int16; NO `ret`).
- **Train:** `bc_any.py --channels 128 --blocks 40 --bs 512 --lr 3e-4 --aug 0.8` (suit-aug, cosine,
  AdamW wd 1e-4), train to convergence (repo's resbn40 reached val 0.894; ~10+ epochs).
- **Serialization for Botzone:** save legacy torch-1.4 (`_use_new_zipfile_serialization=False`).
  ⚠️ OPEN ISSUE: resbn40's BN weights saved by a modern torch **crash Botzone's torch 1.4 on load**
  (status 120) — the deployed lineage uses a 1.4-loadable fused/no-BN form. Resolve before deploying
  a fresh resbn40.

## 4. Reproduction status (2026-06-22)
- Our reproduced base **`base_official_128x40.pkl`** (resbn40, sha256 `69b6199e…`, 10 epochs on the
  5.1M): val-acc **0.879** (slightly under the repo's converged 0.894), and gates **~moyu−12/g** in
  bias-corrected duplicate self-play. → A correctly-architected (resbn40) SL base reproduces *most*
  of moyu but lands ~12/g short; **moyu's remaining edge is the distill/RL finishing**, not the net.
- To reach parity: converge the base fully (→0.894) + a strong ladder distill (`distill_kl.py` toward
  the top-30 champion) + possibly sim6 self-play distill / RL (`sim_cnn.py`/`rl_league.py`).

## 5. Reproduce-from-scratch (one path)
1. `data.zip` → repo cooker → cooked base (~5.1M), verify count + sha256.
2. `bc_any.py --channels 128 --blocks 40` to convergence → base resbn40; gate vs moyu (`gate_imit256.py`,
   N≥400×2 families, no-op-trap proof).
3. Strong ladder distill (+ sim6/RL if pursuing exact parity); gate at each stage.
4. For deploy: re-save 1.4-loadable + verify on a Botzone debug match (256MB/turn, py3.6).

## 6. Artifacts
- Repo (recipe/scripts/arch doc): `github.com/SuuTTT/IJCAI-mahjong` (master).
- Data + weights (HF, sha256-verified): `Dannibal/ijcai-mahjong-moyu-binaries-public`
  (`dataset/data.zip`, `weights/moyu_bn_128x40.pkl`, `champion_data/*`).
- Reproduced base + gates: chinabox `/root/mahjong/train/caiest_repro/ckpt/base_official_128x40.pkl`,
  `/root/base_reproduce_RESULTS.txt`, `gate_base_vs_moyu.json`.
</content>

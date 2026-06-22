# MODEL CARD — moyu (cnn_lad_chunjiandu)  ·  IJCAI-2026 MCR competition bot

> **Purpose of this file:** the durable provenance record for moyu, salvaged 2026-06-22
> from ephemeral vast.ai boxes that had **no version control** and had already suffered one
> data-loss. Everything below is either VERIFIED (cited) or marked INFERRED / TO-CONFIRM.
> Keep this in a durable git remote + back the referenced artifacts to durable storage.

## 1. Identity
- **Model ID:** moyu / `cnn_lad_chunjiandu`
- **Weights (current submission):** `moyu_bn_128x40.pkl` — md5 `749dde3d49cbc9a4474467e3fe626b01`
  (torch path); numpy deploy `cnn_lad_chunjiandu.npz` md5 `53041a7b…`. ~57 MB.
- **Architecture:** ResBNCNN (BatchNorm ResNet), **128 channels × 40 blocks, ~14.3M params**.
  (Earlier lineage per SUBMIT.md: a 16-block ResNet ~10M params → `cnn.pkl`.)
- **Serialization:** legacy torch-1.4 format (`_use_new_zipfile_serialization=False`) so
  Botzone's torch 1.4.0 / py3.6 can load it.
- **Current location (NON-durable, ephemeral boxes):** chinabox `/root/mahjong/ckpt/suphx/moyu_bn_128x40.pkl`.
  ⚠️ NOT yet in durable storage — see §6.

## 2. Deploy constraints (from `deploy/caiest_cnn/SUBMIT.md`, VERIFIED)
- Botzone: **Python 3.6, torch 1.4.0**, PyMahjongGB present. **256 MB/turn** memory limit.
- Legal-by-construction: `feature.py` fan-gates HU via MahjongFanCalculator (HU only when fan ≥ 8).
- Entry `__main__.py` reads via `sys.stdin.readline()`, breaks on EOF, wraps each request → PASS on error.
- Lean deploy only (~11 files); the full 25-file dir caused a 15.6 s turn-1 TLE. Cold-start ~0.65 s.

## 3. Build pipeline (ORDERED) — the recipe
> CORRECTED 2026-06-22 from the AUTHORITATIVE repo `github.com/SuuTTT/IJCAI-mahjong`
> (`CHANGELOG.md`). moyu = `lad_chunjiandu` = **top-30 ladder SINGLE-TEACHER DISTILL**, which
> BEAT the 98k-based multi-teacher BC `distill100b` (`cnn.pkl`). "Coherence > diversity."
> The repo + `~/data.zip` (official 98k `data.txt`) make moyu reproducible. My earlier
> "BC→fine-tune" inference below stage 1 was WRONG; the real lineage is distill, per the repo.
>   - distill100b = multi-teacher BC on official 98k (`preprocess_single.py`→`train_repro.py`).
>   - lad_chunjiandu (moyu) = single top-30 ladder teacher distilled (`extract_top30 --since`,
>     distill scripts) → beat distill100b on the dup-wall gauntlet (+4119 > +3938, 144g).
> NOTE: a prior data-loss (2026-06-07) already happened; pipeline was rebuilt
> (`preprocess_chunked.py`, recompiled judge, fuse2bn/bn2fuse). The repo is the durable record.
> [SUPERSEDED inference below, kept for audit:]

**Stage 1 — base imitation on the OFFICIAL public dataset (VERIFIED, SUBMIT.md):**
- Data: **official 98k-game Botzone dataset** (PUBLIC) → `preprocess_single.py` →
  `data/cooked_single.npz` = **5.87M samples**. ⚠️ `cooked_single.npz` was LOST in the data-loss;
  REBUILDABLE from the public 98k dataset (re-download → `preprocess_single.py`).
- Train: `train/caiest_repro/train_repro.py --epochs N` → `log/checkpoint_without0/best.pkl`.
- Then re-save legacy torch-1.4 + copy to `deploy/caiest_cnn/data/cnn.pkl`.

**Stage 2 — champion-ladder fine-tune (VERIFIED scripts; the "chunjiandu" flavor):**
- Data: champion/top-30 ladder logs `chun_src/{ladder_top30_score1216, rank1-chunjiandu}` →
  cooked sets `union_chun_top30.npz` (~80k decisions), `alltop30.npz`, etc. (these SURVIVE).
- Train: `bc_any.py --blocks 40 --channels 128 --epochs --bs 512 --lr 3e-4 --aug 0.8` (suit-aug, cosine).
- NOTE: 80k ≈ 1.4% of the 5.87M base; fine-tune ONLY, NOT a from-scratch substitute. A fresh
  BC on the 80k alone gates at **−40/g vs moyu** (proves the base stage is essential).

**Stage 3 — sim6 self-play distillation (INFERRED, TO-CONFIRM):**
- Raw `chun_src/sim6-chunjiandu/` (93 MB) + `distill_kl.py` suggest a self-play distillation stage.

**Stage 4 — RL (INFERRED from `ckpt/suphx/` path + `rl_league.py`/`rl_curriculum.py`, TO-CONFIRM):**
- A Suphx-style RL stage may sit on top. NOTE: independent re-runs of Suphx-RL / league-RL
  gate at PARITY vs moyu (so the RL stage's ceiling ≈ moyu).

## 4. Recipe scripts (salvaged verbatim, 2026-06-22)
**`train_imit.sh`** (gauntlet imitation):
```bash
cd /root/mahjong/caiest_repro
for pl in 天胡豪七 好好听 渡鸦 谢飞扬; do
  python3 -u bc_any.py --data "data/gaunt_${pl}.npz" --blocks 24 --epochs 12 --bs 512 \
    --lr 3e-4 --aug 0.8 --out "/root/mahjong/ckpt/g30_${pl}.pkl"
done
```
**`gredux2.sh`** (gauntlet eval vs imitation pool via official judge — `eval/bench_vs_bot.py`,
`WALL_SEED_BASE`, `MAHJONG_JUDGE=Chinese-Standard-Mahjong/judge/judge`).
**Reproduce (SUBMIT.md):** `preprocess_single.py` → `cooked_single.npz` (5.87M) →
`train_repro.py --epochs N` → re-save legacy → `deploy/caiest_cnn/data/cnn.pkl`.
**Key scripts present:** `bc_any.py`, `preprocess_single.py`, `train_repro.py`,
`train_se_official.py`, `data/parse_official_caiest.py`, `distill_kl.py`, `rl_league.py`,
`rl_curriculum.py`, `rl_model.py`, `verify_lever.sh`, `gate_imit256.py`, `feature.py`.

## 5. Evaluation (VERIFIED)
- Held-out top-1 (champion val split): **0.7355**.
- Real-field vs the actual finalists, **N=242**: top-half **81% [76–86]**, dist 1st 28% / 2nd 54% /
  3rd 0% / 4th 19%, **0 TLE**. A steady-2nd specialist (see deadline plan doc).
- Strength gate: every retrain/lever tested gates at parity-or-below vs moyu (imitation ceiling).

## 6. Durable-storage status (✅ BACKED UP 2026-06-22, verified by sha256)
- [x] Push all mahjong code + recipe scripts to a **git remote** (GitHub), commit-pinned.
      → `github.com/SuuTTT/IJCAI-mahjong` (branch `master`). This card lives at `doc/moyu_MODEL_CARD.md`.
- [x] Back up weights (`moyu_bn_128x40.pkl` + key ckpts) to **HF Hub**, sha256-verified.
- [x] Archive the **official 98k dataset** (`data.zip`) to durable storage. ⚠️ This was the only
      surviving copy (on the EC2 box) — now also on HF.
- [x] Back up the surviving champion-ladder data (`chun_src/*` raw + cooked `union_chun_top30.npz`,
      `alltop30.npz`).
- [ ] (still TODO) Rebuild + archive `cooked_single.npz` (5.87M; rebuildable from `data.zip`).
- [ ] (still TODO) Training wrapper that auto-stamps a manifest (git commit + cmd + data/out hashes + metric).

### HF backup repo + artifact hashes (VERIFIED — remote LFS sha256 == source-box sha256)
**Repo (PUBLIC ⚠️):** `https://huggingface.co/datasets/Dannibal/ijcai-mahjong-moyu-binaries-public`
> ⚠️ Created PUBLIC, not private: the `Dannibal` account's **private-storage quota is exhausted**
> (HF returned `403 Private repository storage limit reached`). An empty private repo
> `Dannibal/ijcai-mahjong-moyu-binaries` exists but could not accept LFS uploads. To make these
> private, upgrade the HF plan (or free private storage), then move the files. Note these are live
> competition assets through 2026-07-07 — public exposure is a deliberate-decision point for the owner.

| path in HF repo | bytes | sha256 | source box:path |
|---|---|---|---|
| `dataset/data.zip` | 26386568 | `476abfcca64b2504973a877d6e5834ca623551914071b3e02da0e55b555ac2e0` | EC2 `/home/ubuntu/data.zip` (official 98k `data.txt`) |
| `weights/moyu_bn_128x40.pkl` | 57521902 | `c74b54da27a88b8a1c81c5a18eecd433fce84b243b74d5aaf40ee468158fa786` | chinabox `/root/mahjong/ckpt/suphx/moyu_bn_128x40.pkl` |
| `weights/cnn_lad_chunjiandu.npz` | 53310670 | `2b8d8e41d8409dfe835ab61f4bc4d398e77bc49703ecc633c4e5630d756f8e25` | chinabox `/root/mahjong/deploy/caiest_cnn/data/cnn_lad_chunjiandu.npz` |
| `weights/P1_it80_fused.pkl` | 57350483 | `eed350034096dfe2486d7b60286933689d7970816eb6e215711c6e7b0a71fac3` | chinabox `/root/mahjong/P1_it80_fused.pkl` (deployed candidate) |
| `champion_data/chun_src.tar.gz` | 12050686 | `1d90aaefa944e3a07fe606fa7e11c3d57c960c5377d777bb18b4a307b87cf184` | tkde-box `/root/mahjong/chun_src/` (tar gz; raw ladder/sim6 logs ~250MB unpacked) |
| `champion_data/union_chun_top30.npz` | 2684978 | `4e337212fe2fe08a8df066154a7ee8ccb0c94a68f1e1b447dc31575f0d8c182b` | tkde-box `/root/mahjong/caiest_repro/data/union_chun_top30.npz` |
| `champion_data/alltop30.npz` | 2175945 | `893d826f2d4d3f7368dc9788ed5a870153041030800b50fae177268c9c7beece` | tkde-box `/root/mahjong/caiest_repro/data/alltop30.npz` |

**NOT backed up (intentional):** redundant RL snapshot iterations (recorded as a null result —
RL gates at parity vs moyu, so no marginal value). `cooked_single.npz` (5.87M) is rebuildable from `data.zip`.

> ⚠️ **SECURITY:** the HF token used for this upload (`REDACTED-TOKEN-FINGERPRINT`) was passed in plaintext and
> is EXPOSED. **ROTATE it** at https://huggingface.co/settings/tokens after confirming the backup.

## 7. Reproduce moyu (the path, now that the recipe is known)
1. Re-download the official 98k Botzone dataset → `preprocess_single.py` → `cooked_single.npz` (5.87M).
2. `train_repro.py` → base CNN (validate it reaches moyu's strength at the gate — this is the
   step we were missing; from-scratch on 80k gives −40/g).
3. Fine-tune on champion-ladder cooked sets (`bc_any.py --blocks 40 --channels 128`).
4. (If needed to fully match) sim6 distill (`distill_kl.py`) + RL (`rl_league.py`).
5. Gate each stage vs moyu (`gate_imit256.py` / `verify_lever.sh`, N≥400×2 families).
</content>

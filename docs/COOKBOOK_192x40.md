# Cookbook — `big192x40`: the first model to beat moyu (IJCAI-2026 MCR)

Reproduction recipe for the **192-channel × 40-block ResBNCNN** that beats the
deployed champion `moyu` (`cnn_lad_chunjiandu`, 128×40) on a bias-corrected
duplicate-self-play gate **+3.27/g** and shows a **positive real-field signal**
(+1.55/g, mean placement 2.116 over N=121 real Botzone games vs the finalists).

This is the campaign's first non-null after a long exhaustion record. The key was
**scaling capacity on the FULL official dataset** — earlier capacity tests were run
on a partial (80k-decision) recovered set and looked dead; on the full 5.87M-decision
set, bigger nets beat moyu, peaking at 192 channels.

---

## 0. Provenance / hashes (verify before trusting)

| artifact | what | hash / id |
|---|---|---|
| `data.zip` | official 98,209-game dataset (`data.txt`) | sha256 `476abfcc…` (== EC2 copy) |
| `data/cooked_single.npz` | preprocessed, 5,865,816 decisions | sha256 `2c36f105b996fa9f73859ec3716713a001f109d5dd6f260afdb3869971be5dc7` |
| `ckpt/big192x40_s0.pkl` | trained model, **BN form** (30,330,044 params) | md5 `7a614bfe0473513d785897103b242f19` |
| `base_192x40.npz` | fused fp16 deploy weights, 56.1 MB | md5 `675e221aa15456f13b6f9572c72032f1` |
| `assets/moyu_bn_128x40.pkl` | the reference to beat (moyu, BN form) | md5 `749dde3d…` |

Box used: 4×3090, `/root/IJCAI-mahjong/train/caiest_repro/`. Repo:
`github.com/SuuTTT/IJCAI-mahjong`. Binaries backed up to HF (see §6).

---

## 1. Data → cooked tensor

```bash
# data.zip = official Botzone top-game dump (data.txt = 98,209 rounds, full event stream)
unzip data.zip                      # -> data.txt
python3 preprocess_single.py        # -> data/cooked_single.npz
#   N = 5,865,816 decisions; arrays: obs, mask, act (act range 0–234); 98,209 matches.
#   Sanity: this is the SAME base data moyu was trained on (the parity target).
```

## 2. Train (behavioral cloning, the capacity sweet spot)

```bash
python3 bc_any.py \
  --data data/cooked_single.npz \
  --channels 192 --blocks 40 \
  --epochs 16 --bs 512 --lr 3e-4 --aug 0.8 \
  --out ckpt/big192x40_s0.pkl
# ResBNCNN, ~30.3M params. best_val_acc ≈ 0.889 (held-out top-1).
# Train a 2nd seed (capB_192x40_s1) to confirm the gate win replicates.
```

**Capacity curve (all BC on the full 5.87M set, gated vs moyu — §4):**

| channels×blocks | params | gate edge/g | both seats positive? |
|---|---|---|---|
| 128×40 | 14.4M | ≈ +1.0 | — (under-fit) |
| **192×40** | **30.3M** | **+3.27** (CI[+2.14,+4.39]) | **✅ seatA +3.9 / seatB +2.7** |
| 256×40 | ~54M | +1.5…+2.2 | ⚠️ seatB collapses (−0.2 on one seed) |
| 320×40 | ~84M | +2.76 | one seed only, < 192 |

→ **192 is the peak**; bigger over-fits the fixed dataset. Don't go past 192 without
more data or RL.

## 3. Fuse for deployment (BN-free numpy bot)

```bash
python3 bn2fuse_192.py              # folds BatchNorm into conv -> base_192x40.npz (fp16, 56.1 MB)
# Verify: numpy(fp16) argmax == torch(fp32) argmax on a batch (we saw 40/40 identical, 0 flips).
# Botzone runs py3.6 / torch1.4 -> modern-torch BN nets crash on load; the numpy bot avoids torch entirely.
```

## 4. Gate (the cheap filter — bias-corrected duplicate self-play vs moyu)

```bash
# BN-form ckpt -> --cand-kind resbn ; FUSED ckpt -> --cand-kind resbn_fused
python3 parity_gate.py \
  --cand ckpt/big192x40_s0.pkl --cand-kind resbn --cand-cfg channels=192,blocks=40 \
  --ref  assets/moyu_bn_128x40.pkl --ref-kind resbn --ref-cfg channels=128,blocks=40 \
  --games 4000 --workers 40 --seed0 50000 --out capg_192_50000.json
# Repeat seed0=70000 (SECOND family — BOTH must clear the bar).
```
**Discipline (this project over-claimed ~9× — every rule earned by a burned result):**
- **Calibration:** moyu-vs-moyu MUST read +0.000/g. Any exact-0.000 candidate edge = a
  load bug (no-op trap), not a tie.
- **Both seed families** (50000 & 70000), **per-seat** (seatA AND seatB positive — an
  average of +2 that's seatA +6 / seatB −2 is a coin flip).
- **Bias-corrected:** edge = (cand-vs-ref) − (ref-vs-ref) on the SAME walls, seats rotated.
- **N≥4000×2** to resolve ~1/g effects. A local gate win is a hypothesis until the
  **real-field A/B** confirms it.

## 5. Deploy + real-field A/B (the decision-maker)

The model is 56 MB — **Botzone caps the source zip at 4 MB**, so:
```bash
# (a) Upload the npz as a USERFILE (mounts under data/ at runtime) — NOT in the source zip:
python3 up.py                       # POST /userfile/uploadfile  (see up.py: cookies_live.json)
# (b) Build a LEAN code-only zip (~30 KB, no npz), PIN the model filename literal
#     (_PINNED_MODEL = "base_192x40.npz") and REMOVE any "load largest npz" fallback
#     (3 npz are mounted -> a glob loads the WRONG one = the no-op trap).
# (c) /mybots/create (multipart: name, game, extension=py36, description[non-empty],
#     enable_keep_running, source zip <=4MB).
# (d) SMOKE: fetch the /match log, confirm debug line shows md5=675e221a + 0 TLE BEFORE any A/B.
# (e) paced_runner (captcha-OCR ~50% miss, backoff on 429) + 15-min md5-verifying harvest.
```
**Result (N=121, 0 TLE, md5 verified every game):** mean score **+1.55/g**, mean
placement **2.116** (even field = 2.500), 1st-rate 29.8%. First positive real-field
translation of a gate win (every prior lever tied). *Caveat: +1.55 is within noise at
N=121; the placement metric is tie-biased (analyzer collapses 3-way −8 score-ties →
0 third places). Grow to N≈300 before shipping.*

## 6. Backup

- **GitHub** (`SuuTTT/IJCAI-mahjong`): this cookbook + `bc_any.py`/`preprocess_single.py`/
  `bn2fuse_192.py`/`parity_gate.py` + model card.
- **Hugging Face** (binaries, too big for git): `big192x40_s0.pkl`, `base_192x40.npz`,
  `cooked_single.npz`, `moyu_bn_128x40.pkl` → `Dannibal/ijcai-mahjong-moyu-binaries-public`.

## 7. Next levers (run ON this 192 base — never tried on a base this strong)

1. **Champion-distill** (moyu's own stage-3): `distill_kl.py --base big192x40_s0_fused.pkl
   --champ assets/union_chun_top30.npz --kind resbn_fused` (plain + advantage-weighted) → gate.
2. **192-seed ensemble** (cheap robustness).
3. **Placement-RL** on the 192 base (RL was null on the 128 base; a stronger base is its best shot).

Status at time of writing: replication gate (seed-1) + champion-distill running; real-field
N growing toward 300. moyu stays the live submission until the 192 clears replication AND
a firmed-up real-field A/B.

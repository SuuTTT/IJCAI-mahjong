# Mahjong policy architectures — what we've tried (IJCAI 2026, Chinese Standard Mahjong)

Living doc. Ranks every architecture/feature we've trained for the discard/claim policy
(235-action space), the lesson from each, and what's next. **The headline finding of the whole
project: architecture (representation) mattered far more than the RL/algorithm tuning we spent a
week on.** A standard SL CNN beat our most-tuned MLP 0-of-60 through the official judge.

Two ways we measure strength:
- **val_acc** — supervised top-1 action accuracy on held-out expert games. Cheap, imitation-only.
- **net vs r18** — points over our old MLP champion through the OFFICIAL JUDGE (the trustworthy
  cross-architecture signal). This is the bar that matters.

---

## Tier 1 — DEPLOYED / strongest

### CNN (caiest lineage) — `(38,4,9)` ResNet  ★ current deploy
- 38 tile-feature planes over a 4×9 tile grid → 3-conv stem → **16 residual bottleneck blocks**
  (128 ch) → FC head to 235. ~9.9M params. PyTorch; deploys on Botzone (torch 1.4, py3.6).
- **`base_16x128_final` (14 epochs): val 0.863, net vs r18 = +2826 (99/120 wins).** DEPLOYED,
  confirmed live on Botzone (344 OK verdicts, an 8-fan self-drawn win, claims actively).
- Why it works: convolution captures tile adjacency (runs, partial melds, triplets) natively —
  exactly the structure the flat MLP threw away. **This was the breakthrough.**

---

### Normalized deep ResNet — `resbn40` (40 blocks + BatchNorm)  ★★ NEW BEST BASE
- The caiest CNN but deeper (40 blocks) **with BatchNorm**, which fixes the divergence that killed
  the un-normalized 32-block net. ~14M params.
- **val_acc 0.894** (converged, above the 16-block CNN's 0.863); **net vs r18 = +1493 (49 wins)**;
  and decisively **beats the deployed 16-block CNN champion HEAD-TO-HEAD: +973 (52–25, 4% draws,
  80 games).** *Depth helps once normalized* — confirmed. This is the new base for deploy + RL.
- Deploy notes: 40 blocks ≈ ~1.5–2 s/turn CPU (fits the 6 s budget) but heavier memory (~57 MB
  weights) — must verify Botzone fit; a 24-block resbn may be the deploy sweet spot if 40 MLEs.

### CNN + Transformer hybrid — `cnn_attn` (CONVERGED) — ties the champion
- Conv stem (local) → transformer (global). ~2.2M params. **val 0.881, net vs r18 +1469 (48–3)**,
  but head-to-head vs the CNN champ only **+109 (tie)**. Strong-and-tiny, but not an upgrade.

### Tile-token Transformer — `attn` (d=192, 6 layers, ~2.9M)
- **val 0.897 (highest val-acc), net vs r18 +1227 (47–4)** — strong but below resbn40/cnn_attn
  head-to-head. The 1–2 "illegal" on attention models are an explore_bot harness artifact (even
  the pure-CNN resbn40 shows 1), NOT the model — deploy uses the clean bot.

---

## Tier 3 — DOES NOT WORK (negative results, with reasons)

### Flat MLP — 240-dim → ResMLP (the old `bc_v3_ft → ppo_vb → poolbig → gen2 → r18` lineage)
- ~3.4M params. Legal, reasonable, and **only ever strong relative to itself.** Lost **0 of 60**
  to a 1-epoch CNN. The flat 240-dim feature discards the spatial tile structure. **DEPRECATED.**

### Tile-graph GNN — `gnn` (34 tile-type nodes, suit-adjacency + honor cliques, 4 GCN layers)
- **val_acc 0.7606 (done) — clearly worse.** Collapsing the 38×4×9 grid into 34 tile-type nodes
  loses positional/quantity detail the conv keeps; the hand-built adjacency is a weaker prior than
  letting convolution learn it. Not worth pursuing in this form.

### Oversized Transformer — `attnbig` (d=256, 8 layers)
- **val_acc 0.378 — failed to train** (unstable; too large / LR-mismatched for this data size).
  Bigger attention ≠ better here; the d192/6-layer `attn` is the right scale.

### Un-normalized deep CNN — `deep_32x128` (32 blocks, no norm)
- **Diverged outright** (val stuck at 0.231). Plain residual stacks this deep don't train without
  normalization — which is exactly what `resbn40` fixes.

### Width / shallow CNN variants (arch search)
- `wide_16x256` (256 ch): val 0.830, net +1141 — **no better** than 128 ch, far more params.
- `deep_24x128` (3 ep): net +975 — worse than base at equal epochs.
- `small_8x128` (8 blocks), `gap_16x128` (global-avg-pool head): tested, not superior.

---

## What's NEXT
1. **Finish + judge-benchmark the fleet candidates** (resbn40, attn, cnn_attn) vs r18 AND the
   deployed CNN champion. val_acc is promising but the judge is the decider. If resbn40/attn beats
   the CNN, re-converge it (more epochs) and redeploy (check Botzone 256MB/torch-1.4 fit).
2. **RL fine-tune the confirmed-best base** (value head + self-play with a league guard to avoid
   the passivity trap). Seeds from whichever SL model wins the judge benchmark.
3. **Promising hybrids to try if time:** resbn (normalized depth) + a couple of attention layers on
   top (local-conv + global-attention done right, unlike the small cnn_attn); a deeper transformer
   at the *right* scale (d192, 10–12 layers) since d192/6 was already competitive.
4. **Feature-side ideas (untried):** richer input planes (explicit discard-history / danger planes),
   and a temporal model (LSTM/transformer over the turn sequence) — needs sequence-format
   preprocessing (current data is per-decision snapshots), so it's a larger build.

## Files
- CNN: `train/caiest_repro/model.py` (deploy), `model_cfg.py` (configurable: channels/blocks/head).
- Exploration archs: `train/caiest_repro/models_explore.py` (resbn / attn / cnn_attn / gnn).
- Train: `train_repro.py`, `arch_search.py` (local search), `remote_train.py` + `fleet_explore.sh`
  + `fleet_pull.sh` (fleet exploration). Benchmark: `eval/bench_vs_bot.py`, `explore_bot.py`.

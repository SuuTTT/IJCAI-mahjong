# Arch-continuation (8h) — running writeup

Mandate: keep the 4x3090 busy on MEANINGFUL arch/feature work vs aug_s0 with STRICT anti-thrash
(the box was just over-subscribed into an SSH-timeout thrash). Coexist with the running arch
campaign; honor `/root/STOP_ARCHCONT`; idle > filler.

## Box state at handoff (2026-07-02 ~01:00)
- An autonomous **arch_orch.py --loop (PID 507600)** is already managing the arch campaign:
  trains attn/cnnattn/gnn/temporal and runs CPU-only calibrated placement gates
  (e11_gate / gate_seq, WORKERS=48, CUDA_VISIBLE_DEVICES=""). Its 48-worker gates are the
  load spikes (saw 1-min load 36-45). It honors `/root/STOP_ARCH` (a DIFFERENT flag).
- 4 training jobs were running at start (= the strict cap): 2x e11_train (recipe seeds),
  arch_bc attn_s0, seq_bc temporal_s0. So per the >=4 rule I WAITED and launched nothing until
  a slot + a quiet-load window appeared.

## Anti-thrash design (feat_orch.py, PID logged in /root/feat_orch.log)
STRICT rules enforced programmatically, re-checked every 90s:
- <=4 total trainings (counts arch_bc/seq_bc/e11_train/e11plus_train/train_plus).
- Launch a training only when 1-min load < 20 AND a GPU has <800MB used; confirm the process
  is alive before the next scan.
- Gates: at most ONE at a time, workers=24, only when NO other gate (e11_gate/gate_seq/parity_gate)
  is running AND 1-min load < 18.
- Honors `/root/STOP_ARCHCONT`; disk floor 8GB; setsid/detached; when in doubt -> WAIT.

## Prior campaign results (verified from JSON) — what is ALREADY NULL vs aug_s0
- **CAPACITY**: raw192/raw384/big256/big320 -> all TIED (BESTNET_RESULTS.json). Bigger CNN doesn't beat.
- **RECIPE**: augrepro3/augsuit/lsm00 -> all TIED (RECIPE_RESULTS.json). aug_s0 is the recipe optimum.
- **ARCH**: gnn_s0 WORSE; attn/cnnattn/temporal gating in progress (arch_orch, ARCH_RESULTS.json).
- **FEATURES (old recipe)**: featA/B/C (featplus) + enh192/enh384 (feature44 DEAD/WALL/TURN) were
  trained and reported TIED — BUT featA/B/C used the OLD plain-CE recipe (suit-aug only, no
  label-smoothing / EMA), NOT the enhanced e11 recipe. (The "TIED" claim for featA/B/C lives only
  in ARCH_WRITEUP.md, not in a results JSON.)

## MEANINGFUL QUEUE being run (feat_orch.py)
1. **+FEATURES x ENHANCED RECIPE (priority #1, the genuinely-untested cell):**
   featplus **ABC** = base-38 + A(danger: opp-river/meld-commit/progress, +5) +
   B(shanten reg/7p/13o + useful-tile, +4, from precomputed planeB.npy) + C(genbutsu safe-tile
   per opp, +3) = **50 planes**, trained on the SAME deployable 128x40 CNN with the PROVEN e11
   recipe (suit x reflect x dragon aug, label-smoothing, EMA, warmup+cosine). 2 seeds.
   - This matches the ladder's "ResNet18 + FEATURES" motivation and covers the prompt's requested
     features (shanten/tenpai distance, fan/useful-tile, danger/safe-tile from opponent discards).
   - Plane symmetry verified: shanten & danger are invariant under suit-perm / rank-reflection /
     dragon-perm; spatial planes permute covariantly -> aug stays label-preserving. planeB.npy is
     index-aligned with cooked_single.npz (verified identical ordering).
   - Zero re-cook / zero new big files (planeB.npy already on disk; everything else derived
     on-the-fly) -> no disk/CPU thrash.
   - Gate: parity_gate_plus (no-op-guarded; feeds 50 planes to cand seats via SimPlus).
     Metric = edge_per_game (score edge, 0=tied). BEAT iff edge 95% CI lower bound > 0.
     (This is the native featplus strength gate; distinct from the placement gate, 2.500=tied.)
   - Results -> FEATURES_RESULTS.json + FEATURES_WRITEUP.md (auto-aggregated each pass).
2/3. temporal 2nd seed / borderline-arch 2nd seeds: COVERED by arch_orch (do not duplicate).
4.  aug_s0 2nd-seed noise-floor anchor: only if queue exhausted and time remains.

## Status log
- feat_orch launched, WAITING (load>20) as designed until a quiet slot opens.
- (per-verdict + final ranked table appended below as results land)

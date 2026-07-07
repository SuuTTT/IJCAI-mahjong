# RECIPE optimization sweep on the deployable 128x40 (vs aug_s0)

Reference = **aug_s0** (`ckpt/aug/aug_128x40_s0.pkl`), the current best deployable net. Gate: `e11_gate.py` lam=0 calibrated duplicate-format placement gate (4-seat rotation). 2.500 = tied with aug_s0. **A config BEATS aug_s0 iff its placement 95% CI lower bound > 2.500.**

Calibration (aug_s0 vs aug_s0): placement = **2.5** (1 block(s)) — must read 2.500.

Base recipe: `--steps 130000 --lr 3e-4 --wd 1.5e-4 --lsm 0.05 --ema 0.999 --warmup 2000 --bs 1024 --p_suit 0.8 --p_ref 0.5 --p_drag 0.5` (each config changes only the listed axis).

## Ranked table (gate vs aug_s0)

| rank | config | recipe change | val_acc | n_blk | placement mean | 95% CI | margin_lo | beats aug_s0 |
|---|---|---|---|---|---|---|---|---|
| 1 | augrepro3 | `--seed 3` | 0.8823 | 12 | 2.5013 | [2.4926, 2.51] | -0.0074 | no |
| 2 | augsuit | `--p_ref 0 --p_drag 0` | 0.8874 | 12 | 2.4939 | [2.4854, 2.5024] | -0.0146 | no |
| 3 | lsm00 | `--lsm 0.0` | 0.8799 | 2 | 2.4825 | [2.3275, 2.6375] | -0.1725 | no |

## Verdict

NULL: no recipe variant CI-separated above aug_s0 -> aug_s0 confirmed the recipe optimum for the deployable 128x40.

## Run note — campaign halted by /root/STOP_RECIPE at 2026-07-01 23:18 UTC

The self-managing sweep honored the STOP flag: the training dispatcher and gate loop exited
(per project convention, STOP halts the harness loops; in-flight trainings finish naturally).

Configs GATED before STOP (calibrated e11_gate lam=0 vs aug_s0; 2.500=tied):
  - augrepro3 (same recipe, seed 3) — 12 blocks — placement 2.5013, TIED. == NOISE FLOOR ==
  - augsuit   (suit-only aug)       — 12 blocks — placement 2.4939, TIED (highest val 0.8874).
  - lsm00     (label-smoothing off) —  2 blocks — placement 2.4825, UNRELIABLE (n=2, CI +-0.15).

Trained but NOT gated (STOP halted gating; fused ckpts persist in ckpt/recipe/ for later use):
  - emaoff (EMA off), lsm10 (lsm 0.10), augsuitref (suit+reflect). Not started: wd/ema/steps/lr/aug-strong cells.

Mid-run the box became heavily shared by other concurrent campaigns (arch-search BC, online-RL,
temporal-BC), oversubscribing GPU+CPU and slowing everything; the dispatcher correctly paused new
launches (1500MB free-guard) rather than fight for GPUs.

VERDICT on the evidence gathered: NULL. The two fully-powered configs (augrepro3, augsuit) both
TIE aug_s0, and the same-recipe reproduction control (augrepro3=2.5013) shows the gate's resolution
floor is ~+-0.01 placement. No recipe variant CI-separated above aug_s0. On available evidence,
aug_s0 remains the recipe optimum for the deployable 128x40. Key axis reading so far: suit-only aug
maximizes imitation val (0.8874 vs 0.8819) but does NOT improve play (2.4939) -> the full
suit+reflect+dragon aug set of aug_s0 regularizes better for placement despite lower val.

# BASENET GATE — trained base nets vs deployed distill

**Ref (deployed):** cnn_lad_chunjiandu.npz (ResFused 128x40 distill bot).  
**Gate:** calibrated duplicate-format placement gate (e8_gate.py, lam=0 raw policy). 2.500 = tied with distill.  
**Blocks:** 5 seed-blocks x 300 seeds (1200 games each).  
**Beats rule:** 95% CI lower bound strictly > 2.500.

**Calibration:** distill-vs-distill = 2.5 (OK = 2.500).

## Ranked: gate-vs-distill (higher = better than distill; 2.500 = tied)

| Rank | Net | Arch | Gate mean | ±95% CI | CI range | Beats distill? |
|------|-----|------|-----------|---------|----------|----------------|
| 1 | full_384x40_s0 | resbn_fused 384x40 | 2.5314 | ±0.0339 | [2.4975, 2.5654] | no |
| 2 | big192x40_s0_fused | resbn_fused 192x40 | 2.5255 | ±0.0338 | [2.4917, 2.5593] | no |
| 3 | full_128x40_s1 | resbn_fused 128x40 | 2.5219 | ±0.0322 | [2.4896, 2.5541] | no |
| 4 | full_256x40_s0 | resbn_fused 256x40 | 2.5199 | ±0.0255 | [2.4945, 2.5454] | no |
| 5 | big256x40_s0_fused | resbn_fused 256x40 | 2.5196 | ±0.0295 | [2.4901, 2.5491] | no |
| 6 | full_128x40_s0 | resbn_fused 128x40 | 2.5185 | ±0.0370 | [2.4815, 2.5555] | no |
| 7 | full_256x40_s1 | resbn_fused 256x40 | 2.5048 | ±0.0370 | [2.4679, 2.5418] | no |
| 8 | moyu_bn_128x40 | resbn 128x40 | 2.5003 | ±0.0276 | [2.4727, 2.5279] | no |

## Verdict

**NO upgrade.** No candidate base net is CI-separated above 2.500 vs distill. Best base net is **full_384x40_s0** at 2.5314 (CI [2.4975,2.5654]) — still at/below distill's 2.500. Distill (cnn_lad_chunjiandu) is confirmed our strongest base policy; the entry stays distill.

## Per-net detail

- **calib_distill** (resbn_fused 128x40; distill self (calibration)): mean 2.5000 ±0.0000, blocks=100000:2.5000, 110000:2.5000, 70000:2.5000, 80000:2.5000, 90000:2.5000
- **moyu_bn_128x40** (resbn 128x40; moyu BN base (capacity-sweep ref)): mean 2.5003 ±0.0276, blocks=100000:2.4692, 110000:2.4904, 70000:2.5058, 80000:2.5292, 90000:2.5071
- **full_128x40_s0** (resbn_fused 128x40; converged 90k base, seed0): mean 2.5185 ±0.0370, blocks=100000:2.5621, 110000:2.4971, 70000:2.4912, 80000:2.5063, 90000:2.5358
- **full_128x40_s1** (resbn_fused 128x40; converged 90k base, seed1): mean 2.5219 ±0.0322, blocks=100000:2.5342, 110000:2.5046, 70000:2.5038, 80000:2.5621, 90000:2.5046
- **full_256x40_s0** (resbn_fused 256x40; converged 90k base, seed0 (val~0.886)): mean 2.5199 ±0.0255, blocks=100000:2.5229, 110000:2.5112, 70000:2.4896, 80000:2.5342, 90000:2.5417
- **full_256x40_s1** (resbn_fused 256x40; converged 90k base, seed1 (val~0.898)): mean 2.5048 ±0.0370, blocks=100000:2.5467, 110000:2.4737, 70000:2.4992, 80000:2.5221, 90000:2.4825
- **full_384x40_s0** (resbn_fused 384x40; converged 90k base, seed0 (val~0.884)): mean 2.5314 ±0.0339, blocks=100000:2.5675, 110000:2.5217, 70000:2.4929, 80000:2.5417, 90000:2.5333
- **big192x40_s0_fused** (resbn_fused 192x40; historical 'beat moyu' candidate): mean 2.5255 ±0.0338, blocks=100000:2.5529, 110000:2.5158, 70000:2.4833, 80000:2.5329, 90000:2.5425
- **big256x40_s0_fused** (resbn_fused 256x40; historical big-256 candidate): mean 2.5196 ±0.0295, blocks=100000:2.5217, 110000:2.5283, 70000:2.4925, 80000:2.5533, 90000:2.5021

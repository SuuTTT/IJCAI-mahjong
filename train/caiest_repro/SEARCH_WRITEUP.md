# Test-Time Search (PIMC / determinized value-lookahead) vs aug_s0 — Verdict

**Base policy:** aug_128x40_s0.pkl (current best; CI-beats bn128s1 by +0.0058).  
**Value model:** value_256x40.pkl (ValueMT 256x40, held-out 4th-AUC 0.955), V_place head at the leaf.  
**Gate:** calibrated duplicate placement gate, aug_s0-vs-aug_s0 = **2.500** (verified, N=0 search off).  
**Objective (placement, lower=better):** each candidate discard scored by mean over N determinized worlds; a world that ends in a Hu within H plies returns the EXACT duplicate avg_rank (incl. immediate opponent rong on our discard = the deal-in / safe-discard signal), a truncated world returns the V_place leaf. Search fires only when the policy's argmax is a discard and its top-K legal discards are within delta=3.0 logit; overrides the policy discard iff mean-placement improves by > margin (0.0).

**Ship rule:** a tier ships iff placement 95% CI lower bound > 2.500 AND per-move time <= 6000 ms (Botzone budget, keep-running).

## Prior PIMC (why it was null)

The earlier PIMC (`deploy/caiest_cnn/`, `pimc_par.res`) was a small field test (~20 games/opponent) with weak bundled rollout nets (fast8/vbig) scored in raw MCR points, not placement — PIMC (~+505 net) read the same as PLAIN (~+520), i.e. no separation and heavily under-powered. This run is the proper version: the strong value model at the leaf, the deployed aug_s0 as base, a placement objective aligned with the gate, an explicit immediate-rong deal-in term, and a calibrated multi-block CI gate.

## Per-tier results (each tier = a compute level; 10 blocks x 1200 games)

| Tier | N worlds | H plies | per-move ms (1-core) | override rate | placement mean | 95% CI | margin vs 2.500 | CI-beats aug_s0? |
|------|----------|---------|----------------------|---------------|----------------|--------|-----------------|------------------|
| N10_H12 | 10 | 12 | 475 | 0.238 | 2.4839 | [2.4694, 2.4984] | -0.0306 | no |
| N20_H20 | 20 | 20 | 864 | 0.234 | 2.4798 | [2.4646, 2.4950] | -0.0354 | no |
| N40_H30 | 40 | 30 | 1900 | 0.237 | 2.4887 | [2.4782, 2.4992] | -0.0218 | no |

(per-move ms shown is the isolated single-core probe = Botzone-representative; gate-time ms under 48-way load is higher but each block records its own.)

## Verdict

**NEGATIVE (stronger than null): all 3 tiers are CI-separated BELOW 2.500 — determinized value-search actively HURTS placement vs the aug_s0 imitation policy (margins -0.022 to -0.035), and more compute (N=10->20->40) does NOT recover it.**

No search tier CI-separated above aug_s0. Consistent with the E8 1-ply value-guided null (same value model, also flat at 2.500): the value/rollout signal does not add placement over the imitation policy at this power. 
**Why (candidate causes):** (1) rollout playout is the cheap shanten/fan heuristic, not the strong policy, so leaf states are off-distribution for V_place; (2) determinization noise — opponents' concealed hands are uniform samples, so the immediate-rong deal-in estimate is high-variance; (3) the imitation policy already encodes most safe-discard behaviour, leaving little headroom; (4) detection floor here is ~+0.02 placement (12k games/tier), so a true effect smaller than that is not excluded. 
Either way this is an informative negative: test-time determinized search at up to ~1899.7 ms/move does not beat the imitation ceiling on this gate.

## Monotonicity in compute

Placement by compute level: N10_H12(N10/H12)=2.4839, N20_H20(N20/H20)=2.4798, N40_H30(N40/H30)=2.4887. 
More determinized compute does not move placement toward a win — every tier lands ~0.015-0.020 BELOW the 2.500 base and the ~4x compute span (475->1900 ms/move) leaves the sign unchanged (all worse). This is a real, CI-separated negative, not a wide-CI null.

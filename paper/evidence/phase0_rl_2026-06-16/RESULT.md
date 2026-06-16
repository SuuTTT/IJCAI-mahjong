# Phase-0: fan-backward warm-started self-play RL — NULL (2026-06-16)

Goal: test whether the published SOTA recipe (Tjong-style SL->RL with fan-backward shaping),
warm-started + KL-anchored, lifts a policy above its supervised warm-start. De-risk on a 3060.

Setup: distill lad_chunjiandu -> 12-block net (warm-start agreement 0.617 vs lad; 12% win8 self-play).
Self-play PPO in the JAX env, FIXED fan scorer, fan-aware completion+flush shaping, KL-leash to the
frozen warm-start (forget-prevention), entropy off, LR 1e-5.

Findings:
- PURE PPO (no KL leash) COLLAPSES the warm-start in ~5 iters: entropy -> 3.5 (uniform), win8 -> 0,
  draws -> 100%. (A documented warm-started-RL instability.)
- KL-leash kl_c=1.0 (tight): no collapse, but win8 HOLDS FLAT ~10-12% (the warm-start level) over 35
  iters -> ties. kl bounded ~0.15-0.23 (policy can't move).
- KL-leash kl_c=0.3 (loose): win8 DEGRADES 12 -> 5.6 -> 6.5% over 10 iters, drifting toward the
  draw-everything failure -> worse than warm-start.

Verdict: NO regime where fan-backward warm-started RL beats the warm-start (tight=ties, loose=degrades).
At feasible (small-net) scale, the published winning recipe does not lift the policy -- the strongest
form of the imitation-ceiling/RL-null result. Caveat: 3060-slow (~10-35 iters/run); no UPWARD signal in
either run though. A faithful Tjong repro would need its larger transformer + more compute (high-risk
given this).

For Paper A: strongest null -- we implemented the published winner and it still tied at feasible scale.
For Paper B: the small-net SL->RL path does not show life; not pursued further.

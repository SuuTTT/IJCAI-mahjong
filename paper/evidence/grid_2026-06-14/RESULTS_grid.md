# §5 Imitation-ceiling + §6 Noise-floor — full grid results (2026-06-14)

Primary box (40833388, A4000). All gauntlets: 144-game duplicate vs `lad_chunjiandu`, plain net, rebuilt
judge. Distill = `distill_kl --aw --steps 700`. Grid = 9 teachers × 6 β (KL leash) = 53 evaluated cells.
Raw: `raw_grid.txt`.

## §6 Noise floor (the calibration)
Identical `lad_chunjiandu` vs itself, 144g × 5 wall seeds: **−8, −95, −20, +242, +82**.
→ **σ = 115 net/144g**, mean +40. **Minimum publishable margin (2σ) = ±231/144g.** Any |net| below this is
noise. (More replicates pending; box2/3 left to their own projects.)

## §5 Headline: agreement does NOT predict play
Across 53 cells, **Pearson r(teacher-agreement, gauntlet-net) = −0.037**. The agreement proxy is essentially
*uncorrelated* with real play strength — the central evaluation-gap claim, quantified on a controlled grid.

## §5 Strength axis: coherence > strength (the imitation ceiling)
Distilling `lad_chunjiandu` toward teachers of known real strength (mean net over β, σ_floor=115):

| teacher | real strength | agreement | mean net | best β | verdict |
|---------|--------------:|----------:|---------:|-------:|---------|
| `mythos` | **+9.73** (strongest) | 0.732 | **−276** | b0.7 +99 | imitating the strongest bot **HURTS** |
| `typec` | +8.02 | 0.712 | −79 | b1.0 +59 | ≈ tied |
| `chunjiandu` | +5.16 (*our own teacher*) | 0.803 | **+222** | **b0.7 +487** | **HELPS** (coherent) |
| `resnet50` | −0.48 | 0.736 | +4 | b0.7 +96 | ≈ tied |
| `damselfish` | −5.58 (weakest) | 0.582 | +80 | b1.0 +563 | noisy (σ 290) |

**Real strength does NOT predict distilled benefit.** The strongest field bots (`mythos`, `typec`) tie-or-hurt;
the only reliable gain comes from our *own coherent* teacher (`chunjiandu`). You cannot out-imitate a bot whose
strength comes from search/LLM reasoning you can't observe — you can only inherit the *style* you can see, and a
foreign style corrupts a competent policy.

## §5 Scale axis: more (incoherent) data is worse
Distilling toward the pooled strong-5 set, by size (mean net over β): 2k **+88** · 4k −13 · 8k **−270** · 16k −235.
**More data of incoherent strong-bot moves makes it worse**, not better — data scale can't rescue an incoherent
teacher. (Contrast: the coherent `chunjiandu` set helps at all sizes.)

## Bonus: a significant Round-2 lead — close the teacher gap
`chunjiandu` (our own teacher) by β: 0.2 +123 · 0.3 +207 · **0.5 +459*** · **0.7 +487*** · 0.9 +180 · 1.0 −121
(* = outside 2σ=231). At a moderate KL leash (β0.5–0.7), distilling `lad_chunjiandu` toward `chunjiandu` beats it
by **+459 to +487/144g (~4σ)** — recovering the ~2.8 pt/game gap we'd lost to our own teacher. β1.0 (over-leashed)
doesn't move; low β (0.2–0.3) drifts. **This is a real, significant, shippable Round-2 candidate** — pending
multi-seed confirmation (each cell is one gauntlet).

## Takeaways for the paper
1. r(agreement, play) ≈ 0 on a controlled grid — agreement is not a valid model-selection signal.
2. Imitation benefit tracks *teacher coherence with the base*, not teacher *strength* — the imitation ceiling.
3. Data scale can't fix an incoherent teacher.
4. The KL-leash β has a sweet spot (0.5–0.7); too tight = no learning, too loose = drift.

## REPLICATION (6 wall seeds/cell) — the single-run grid was noise-dominated

The §5 grid above used ONE 144g gauntlet per cell. Re-running the key cells at 6 independent
wall seeds collapses every effect toward zero:

| cell | single-run (grid) | 6-seed mean ± sd | verdict |
|------|------------------:|-----------------:|---------|
| 0.5_chunjiandu (our teacher) | +459 | **+48 ± 186** | within noise |
| 0.7_chunjiandu (our teacher) | +487 | **+23 ± 379** | within noise |
| 0.5_mythos (strongest +9.73) | -737 | **-77 ± 283** | within noise |
| 0.5_typec (+8.02) | +19 | **+22 ± 373** | within noise |
| 0.5_scale_2000 | +331 | **-24 ± 201** | within noise |

**Every key cell ties `lad_chunjiandu` once replicated** (all |mean| < the σ≈115 identical-bot floor,
and the per-cell sd 186–370 shows non-identical comparisons are noisier still). The single-gauntlet
"+459 (≈4σ)" chunjiandu lead — which a naive pipeline would have SHIPPED — is a noise artifact.

**This is the paper's sharpest result:** the §6 noise floor catching a single-run false positive in
real time. It is the evaluation-gap thesis demonstrated on our own pipeline, end to end: there is no
re-distillation lever that beats the teacher you were distilled from; what looks like one is the noise
floor. Confirms the campaign-wide finding that ~18 interventions all sit inside the noise band.

# Technical Report — Team *moyu* (bot `kdens3`)
## IJCAI-2026 Chinese Standard Mahjong (MCR) Competition

**Final result: 2nd of 16 — runner-up, a statistical coin flip for 1st.**

*Prepared for the competition-hosted technical meeting. This report describes our solution
in detail, the full arc of iterations behind it, our competition story, and the lessons we
draw — including the ones that cost us the most to learn.*

---

## 0. TL;DR

- Our bot **`kdens3`** (submitted as *moyu*) is a **3-network imitation ensemble** — a
  convolutional policy (ResNet-BN, 128 channels × 40 blocks) trained by **supervised
  imitation / distillation** of strong human and bot play, deployed as a
  **mean-softmax-over-legal-moves** ensemble of 3 independently-seeded nets.
- It finished **2nd of 16** in the Stage-2 championship (512 walls × 24 seat permutations =
  **12,288 games**). The margin to 1st (*kong*) was **+0.048 points/game**, a paired-wall
  significance of **t = 0.13, P ≈ 0.56** — i.e. **indistinguishable from a tie**. The top three
  bots were statistically inseparable.
- **Zero rule errors across all 4 finalists × 12,288 games.** We beat the eventual champion
  **on defense** (deal-in rate 16.93% vs 17.37%); we lost the title on **win-value
  conversion** (their self-draw/*zimo* wins paid more).
- Behind that single submission are **two full campaign cycles and dozens of distinct
  levers** — reinforcement learning (five flavors), Monte-Carlo search (online and offline),
  architecture search, ensembling, defense/fold models, opponent modeling, human-knowledge
  injection, and for 2027, test-time PIMC search and league self-play. **Almost every lever
  returned null or parity.** The central empirical finding of our campaign is that
  **imitation of expert play caps the agent at the expert level, and that ceiling is
  remarkably hard to exceed** in MCR.
- Our most valuable contribution is arguably **methodological**: a verification discipline
  that caught **~8 false-positive "improvements"** before they could mislead us, and a
  precise account of *why* the standard in-house evaluation understated defense by 6–8×.

---

## 1. The problem

**Chinese Standard Mahjong (MCR / 国标麻将)** is a 4-player, imperfect-information tile game
with a **≥8-fan floor**: a hand may only be declared a win if it scores at least 8 fan under
the (large, combinatorial) MCR fan table. This floor is the defining strategic constraint —
most locally-legal hands are *not* winnable, so play is a constant trade-off between building
**value** (reaching the 8-fan floor) and building **speed/width** (having enough winning
tiles to actually complete before an opponent does), all under **hidden** opponent hands and
a shrinking wall.

**Platform & format.** The competition ran on **Botzone** (game id
`5e37dcf74019f43051e53201`). The decisive Stage-2 used the **duplicate format (复式赛制)**:
each wall is played under all `4! = 24` seat permutations, and per-wall scores are summed so
that **seat luck and deal luck are engineered out** — the final ranking reflects **relative
strength**, not variance. The per-decision compute budget is generous (**~5000 ms**), which
matters for the 2027 search direction (§6).

**Scoring subtlety that shaped the whole race.** The final ranks bots by **cumulative
wall-score**, not by mean placement. A bot that reliably places 2nd is *not* rewarded the way
a bot that occasionally wins big is. This is exactly where the title was decided (§4).

---

## 2. Our solution in detail

### 2.1 Core: imitation, then ensemble

`kdens3` is deliberately **not** a search or RL agent. It is a **behavior-cloning policy**
distilled from a corpus of strong play, and its entire design philosophy is
**"reproduce expert decisions reliably, at zero error rate."**

**Architecture.** A residual CNN with batch-norm, **128 channels × 40 residual blocks**,
operating on a stack of ~38 feature planes (34 tile types + auxiliary planes encoding the
player's hand, melds/packs, discards, and public game state). The output is a policy over the
discrete action space (discards, and claim actions: Peng/Chi/Gang/Hu). A legal-move mask is
applied so the policy only ever proposes valid actions.

**Training signal — distillation of top play.** The net is trained by supervised imitation of
decisions harvested from strong Botzone players (the campaign's teacher policy is referred to
internally as `chunjiandu`-class top play, later augmented with the Stage-2 finalists'
decisions). We treat this as **policy distillation**: match the teacher's action distribution,
not merely its argmax.

**Deployment — the ensemble.** We train **3 independently-seeded** copies and, at inference,
**average the softmax-over-legal-moves** across the three, then act. This ensembling is what
the name `kdens3` encodes ("**k**-**dens**ity **ens**emble, **3** members"). The ensemble buys
a small but real reliability gain — smoother tails, no single-net blind spots — at
**~1000 ms/move**, comfortably inside the 5 s budget (and **58–96× headroom** versus the
budget on the infrastructure we built).

### 2.2 Why this design won us a podium

Three properties, in order of importance:

1. **Zero-error reliability.** Across 12,288 final games our bot made **0 illegal moves and 0
   timeouts**. On the live ladder, many bots suffer **15–21% error-endings**; those are
   catastrophic in a cumulative-score format. Reliability is table stakes for the final, and
   we treated it as a first-class objective, not an afterthought.
2. **Champion-level defense.** Our deal-in rate (16.93%) **beat the eventual champion's**
   (17.37%). The imitation policy prices danger implicitly — it learned from experts who avoid
   dangerous discards — without any explicit defense module bolted on (we tried bolting several
   on; they all hurt, see §5).
3. **Consistency.** In the duplicate format, the winner won on **consistency, not peak**. The
   Stage-1 champion's four sub-scores were all within a spread of ~13 points; everyone else
   swung 200–400 and suffered catastrophic parts. Our imitation policy is low-variance by
   construction.

### 2.3 The evaluation & verification infrastructure (our real edge)

Equal in importance to the model is the **measurement stack** we built, because the single
hardest problem in this domain is **knowing whether a change actually helped**:

- **Bias-corrected paired-wall gating.** Every candidate is compared to the baseline on
  **identical duplicated walls**, and we subtract a **matched baseline-vs-baseline null** on
  the *same* walls. A lever is "real" only if it clears a **pre-registered threshold** with
  **N ≥ 400** across **two independent seed families**, verdict computed by script and appended
  to a ledger — **never typed by hand**.
- **Null-calibration gates.** Any evaluation harness must first **reproduce the baseline
  against itself at exactly the tie value** (placement 2.500 on the 1–4 scale) before we trust
  a single number it produces. This one discipline caught multiple silent harness bugs
  (see §5).
- **GPU-batched inference (586×).** We built a batched inference server that runs the CNN
  forward passes for many game streams in lockstep, giving a **586× speedup** over a single
  CPU stream — this is what made large confirming gates and, later, search-at-scale feasible.
- **Automated real-field measurement.** Because in-house evaluation against weak proxies
  *lies* (see §5, "the evaluation wall"), we automated **real Botzone matches against the
  actual finalist bots** (Socket.IO + captcha-OCR match creation), the only fully trustworthy
  signal.

---

## 3. How many iterations did we develop?

Honestly: **two full campaign cycles**, and within them **dozens of distinct levers**. The
one-line summary is *"one model shipped, and a very large graveyard of rigorously-killed
ideas behind it."* Here is the honest catalogue.

### 3.1 Cycle 1 — the 2026 competition run (strength & defense)

| # | Lever | Verdict |
|---|-------|---------|
| 1 | **SL / distillation base (`kdens3`)** | **SHIPPED — the winner** |
| 2 | Same-arch ensembles (beat best member?) | Parity |
| 3 | Architecture diversity (Transformer / CNN-attn, Tjong-style) | **FAIL −57/game** — fit labels better, played far worse |
| 4 | Suphx-style RL (GRP return + oracle critic) | Parity (components reproduced faithfully; net zero) |
| 5 | DMC best-response | False positive → **−10.14/game** at scale |
| 6 | Fan-Backward RL (dense shaping) | Parity (critic was broken, then fixed → still parity) |
| 7 | Self-play **league** RL (moving anchor + past selves) | **Conclusive null** — all eval points ≤ 0 |
| 8 | AWR exploitation | −1.4 / −5.7 per game |
| 9 | Online PIMC search | Dead (5 separate false positives caught) |
| 10 | **Offline** unlimited-compute PIMC (32 worlds × depth 24) | Parity (+1.78 < +2.0 floor) |
| 11 | Deal-in rerank / safe-discard | Null — policy already avoids flagged discards |
| 12 | Fold-mode v1 (learned fold value) | Null (+0.0037/game, ~200× under noise) |
| 13 | Fold-mode v2 (tenpai/wait-triggered) | Conclusive null — *worse* as more signal added |
| 14 | Opponent wait/fan/tenpai predictors | Tenpai AUC **0.887** (works!) but does not convert to points |
| 15 | Human-knowledge injection — fan-aware offense | Null (policy already fan-aware, ρ=0.48) |
| 16 | Human-knowledge injection — meld-ceiling / 包牌 defense | Null (matches human danger labels 98.65%) |
| 17 | Wait-quality-aware discard overlay | No-op (0 overrides — argmax already widest wait) |
| 18 | Heuristic-learning (HL) overlay loop | Null |

### 3.2 Cycle 2 — the 2027 investigation (can *anything* beat `kdens3`?)

After the final we asked the sharper question: **is `kdens3` actually optimal, or just our
local ceiling?** We attacked it on every axis we could build a deployable version of.

| # | Lever | Verdict |
|---|-------|---------|
| 19 | Source-conditioned multi-corpus BC | Null (val-acc +0.0012 that **does not transfer** to placement) |
| 20 | Final-corpus fraction sweep | Val-acc rises, placement flat (val-acc ≠ strength, again) |
| 21 | **Trained value head** (leaf evaluator for search) | Built, **r = 0.71** — a genuine asset we previously lacked |
| 22 | Stage-1 greedy value-guided discard | **Loss 2.026** — offense-blind-to-defense |
| 23 | **Oracle** 1-ply discard search (perfect info) | **3.55 vs 2.50 — the ceiling of discard search** (non-deployable; cheats by seeing hidden tiles) |
| 24 | Deal-in defense model v2 (AUROC **0.884**) | Model strong; override gate null (**2.4995**, monotone: more override → worse) |
| 25 | Value-aware action-value (joint EV − λ·deal-in) | Loss 2.19 — no linear blend of two weak static signals recovers the oracle |
| 26 | **Deployable PIMC** (determinized rollout), uniform N=20, value-cutoff | **Tie (2.46)** |
| 27 | PIMC + belief-weighted worlds (opponent belief AUROC 0.765) | **Hurts** (search exploits the belief model's systematic errors) |
| 28 | PIMC + placement-head leaf / deeper rollout K | Both **hurt** |
| 29 | PIMC + more worlds (N=50) — best deployable config | **Tie (2.4898)** at 1920 games |
| 30 | Opponent belief model | Built, AUROC **0.765** (beats hypergeometric baseline) |
| 31 | Wide-explore RL sweep (KL 0.05–0.50 × entropy) | Plateau at anchor |
| 32 | **League self-play RL** (opponent pool, KL-anneal) | **Plateau — 0/10 durable crossings above anchor** |

### 3.3 Cross-domain science (for the papers)

To understand *why* the ceiling is so hard, we replicated the key sub-mechanism (distillation
denoising) across **nine domains**: MCR, Doudizhu, CIFAR-10N/100N, chess (3 rating bands),
Othello 6×6, Robomimic, MinAtar, a synthetic controlled-coherence grid, and Leduc poker. That
work sharpened a publishable theory (§7).

**So: one shipped model, ~32 distinct levers across two cycles, nine cross-domain replications.**

---

## 4. Our competition story

**Round 1 → 11th of 16.** Our first submission ranked mid-pack. Our in-house evaluation
insisted the bot was "at the ceiling," yet reality said 11th. This contradiction became the
most productive thread of the whole campaign (§5, the evaluation wall).

**Stage-1 → 3rd of 16, advancing to the top-4 championship.** Four Swiss-style duplicate
parts; total = mean of parts. We finished 3rd (1126.30) behind QiuQiuR (1232.59, a clear +100)
and player152 (1129.24), in a **three-way tie** with the eventual champion *kong* (1124.91) —
2nd/3rd/4th were within ~5 points. **The decisive lesson was already visible here:** the
Stage-1 leader won on **consistency** (parts all 1214–1247, spread ~13), while everyone else —
us included — swung 200–400 points and had catastrophic parts. Our own 1037 → 1289 swing is
exactly where our points leaked.

**Stage-2 championship → 2nd of 16. A coin flip for the title.** 512 walls × 24 permutations =
**12,288 games**. Final scores: **kong +3923, moyu +3329**, QiuQiuR +1877, player152 −9129.
Paired-wall significance between us and the champion: **t = 0.13, P ≈ 0.56** — statistically a
**dead tie**. In head-to-head placement we were actually **above** kong (36.53% vs 36.25%).

**How the title was decided — the paired-wall decomposition** (kong − moyu = +0.048/game):

| Component | Direction | Magnitude |
|-----------|-----------|-----------|
| Win **rate** | equal | −0.001 |
| Win **value** (zimo composition) | **kong** | +0.078 (self-draw pays 65.6 vs ron 36) |
| **Deal-ins** (defense) | **MOYU** | +0.102 (we deal in *less*) |
| Passive bleed | kong | +0.072 |

We **out-defended the champion** and lost on **win-value conversion**: kong's wins were more
often high-paying self-draws. In a cumulative-score format, that is the whole ballgame. This is
the single most actionable finding for a future team: **the marginal point is in
win-conversion (turning safe 2nds into 1sts), not in defense** — our defense was already
champion-grade.

---

## 5. What we learned — the hard lessons

### 5.1 The imitation ceiling (the central finding)

**Imitating expert play caps you at expert level, and in MCR that cap is extraordinarily hard
to exceed.** We proved this the expensive way, twice:

- **Cycle 1:** every strength lever (RL in five flavors, online and offline search,
  architecture search, ensembling, exploitation) returned **null or parity** versus the
  imitation base. Even **unlimited-compute** offline PIMC only reached parity.
- **Cycle 2 (deployable, rigorous):** BC variants **tie**, deployable PIMC across **every**
  configuration **ties (≤ 2.49)**, belief-weighting **hurts**, wide-explore RL **plateaus**,
  and **league self-play RL plateaus at the anchor** with **0/10 durable crossings**.

The mechanism is precise: **every component we could build — policy, value head, opponent
belief, rollout policy — is itself learned from the same imitation data, so none of them can
exceed the teacher.** The oracle 1-ply search reaches 3.55 (vs 2.50) *only because it cheats*
by seeing the hidden tiles; that signal does not survive any deployable approximation. The
"deeper rollout hurts" result is the tell: the rollout policy (our own imitation net) is the
binding limiter. **To beat `kdens3` you need a signal imitation cannot provide** — from-scratch
superhuman self-play or a genuinely better data source — and neither materialized within our
compute budget.

### 5.2 The evaluation wall

**The gap between "the eval says X" and "the field says Y" was the most dangerous thing in the
project.** Our in-house duplicate gate said the bot was saturated and that defense was
placement-neutral; the real field ranked us 11th and, in the final, defense was worth **+0.102
points/game**. The in-house gate understated real exposure by **6–8×** (measured deal-in
exposure 16.6% vs the ~2–3% the gate implied) because it evaluated against **weak proxy
opponents** and on a **score-EV metric** that could not price variance-reduction. The fix was
to build **real-field measurement** and to treat every in-house number as a hypothesis until
confirmed against real opponents.

### 5.3 val-acc ≠ playing strength (repeatedly)

Higher validation accuracy **did not transfer to placement**, in mahjong *and* in chess. Two
chess student-ensembles with **identical** val-top1 (0.445, 0.445) had playing strengths of
0.208 vs 0.054 — a swing entirely **invisible** to accuracy. We now never gate on a proxy
metric; we gate on the task metric with a confidence interval.

### 5.4 Verification discipline beats cleverness

Our verification machinery **caught ~8 false-positive "improvements"** (5 PIMC, DMC best-
response, fold-v2 head-bias, and the source-conditioning transfer illusion) before any of them
could waste a submission or a paragraph. The specific traps and their fixes:

- **Overlapping / non-disjoint evaluation blocks** manufacture significance (winner's curse +
  correlated blocks). Rule: blocks must be **disjoint** (`step ≥ games`), and the **gate region
  must be disjoint from the selection region**.
- **Estimator self-selection** (a fold head scoring its own chosen folds) fakes a positive;
  **model-free evaluation** revealed it was negative.
- **Silent harness no-ops** (a determinization bug made PIMC secretly do nothing, returning
  exactly 2.500) are caught only by a **null-calibration gate** and a **true-state cross-check**.
- **Pairing and disjointness are separate obligations** — we even caught ourselves applying the
  disjoint-blocks rule on the wrong axis (unpaired cross-policy comparisons), which is what
  dissolved a whole paper's central claim.

### 5.5 A well-trained imitation net already does the "obvious" expert things

Most human-knowledge features we tried to inject (fan-awareness, meld/包牌 defense,
wait-width) were **already present** in the imitation policy — it matched expert danger labels
98.65% of the time and expert achievable-fan choices 98.4%. **Injecting a feature for behavior
the expert already exhibits adds nothing.** (Caveat we hold honestly: this is an empirical
tendency for *in-distribution* behavior, not a theorem — rare/OOD patterns can still have
holes.)

### 5.6 The format rewards consistency and win-conversion

The duplicate cumulative-score format **engineers out luck** and rewards (a) **zero-error
reliability** and (b) **win-value conversion**. It does *not* reward the safe-2nd-place style
that expected-score maximization produces. The champion won on consistency; we lost on
conversion. A future MCR agent should optimize **1st-place rate against a broad field**, not
expected score against weak proxies.

---

## 6. Where a 2027 team should push (the only doors we could not open)

We exhausted the deployable-imitation space. The remaining doors — none of which we could walk
through in-budget — are:

1. **From-scratch superhuman self-play RL** at real scale. Our league plateaued at the anchor,
   but the plateau is *structural* (near-optimal SL policy → tight trust region → weak self-play
   gradient), not obviously fundamental. A larger-scale, longer-horizon run with a genuine
   exploiter population is the highest-upside unexplored lever. **Infrastructure is ready**: an
   oracle-exact vmappable JAX MCR environment (validated 12,288/12,288), a PopArt-normalized
   PPO trainer, and our league trainer with opponent-pool + KL-anneal controls.
2. **A better data source.** Everything we have is capped by the teacher. Higher-tier human
   data, or a self-play corpus that provably exceeds the current expert level, changes the
   ceiling.
3. **Win-conversion as an explicit objective.** The final decomposition names the gap:
   turn safe 2nds into 1sts via higher-value hand construction. This is a
   *hand-construction* problem (set the shape 2–3 shanten earlier), **not** a discard-overlay
   or search problem — both of those fire too late, which is exactly why every overlay was a
   no-op.
4. **Test-time PIMC within the real 5 s budget.** We built and validated a **GPU-batched PIMC
   engine (96×)** but every deployable configuration tied. The 5 s budget permits a far more
   accurate search (many worlds, deep rollouts) than we ran; whether accuracy-first PIMC can
   recover any of the 2.50 → 3.55 oracle gap is the open empirical question. Our honest prior:
   the rollout policy is the limiter, so the answer is probably still "tie" — but it is the one
   search variant left un-maxed.

---

## 7. Research output

The campaign produced a genuine scientific result beyond the competition: a characterization
of **when "distill, then ensemble" beats "ensemble the teachers."** Across nine domains we find
it pays **only when imitating a coherent policy observed through real noise** (CIFAR-N,
Doudizhu); in clean games (chess, Othello, Leduc) distillation matches teacher-ensembling at
**half the inference cost** but does not beat it. A synthetic controlled-coherence grid
reproduces the full picture (an inverted-U in noise, collapsing with incoherence). This, plus
the **evaluation-wall** methodology from §5, form the basis of our paper submissions.

---

## 8. One-paragraph conclusion

We built a **reliable, champion-grade imitation ensemble** and finished **2nd of 16 in a
statistical tie for 1st**, beating the champion on defense and losing only on win-value
conversion in a format that rewards it. Behind that one clean model is a **disciplined
graveyard of ~32 rigorously-killed levers** across two campaign cycles, and the campaign's
real deliverable is the pair of hard-won lessons: **imitation caps you at the teacher, and in
MCR that ceiling survives search, RL, and every static heuristic we could build** — and
**the evaluation you trust is probably lying to you by 6–8× until you measure against the real
field.** We are proud of the podium; we are prouder of the ~8 false positives we caught before
they became claims.

---

*Artifacts, code, models, and a full reproducible handoff accompany this report. See the
project handoff document for access to the champion weights, the batched-PIMC engine, the JAX
RL environment, and every result JSON cited above.*

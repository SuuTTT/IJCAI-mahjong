# HANDOFF — resume the IJCAI-MCR project (papers **or** 2027 competition)

**Timestamp: 2026-07-15. Campaign complete; box being downsized.** This supersedes the
pre-final `HANDOFF.md` (2026-07-01). It is written so that *anyone* can pick up either track —
finishing the papers, or building the 2027 competition entry — without talking to the original
team.

> **Secrets are NOT in this repo.** Box SSH, HuggingFace/Botzone credentials, and account
> logins live in the private `~/HANDOFF_IJCAI_MAHJONG_0707.md` on the owner's machine. Ask the
> owner. This repo's git history was scrubbed (git-filter-repo) of all credentials/emails/
> author identity — keep it that way.

---

## 1. One-screen state of the world

- **Competition result:** bot `kdens3` (as *moyu*) finished **2nd of 16** in IJCAI-2026 MCR —
  a statistical coin-flip tie for 1st (t=0.13). See `doc/TECHNICAL_REPORT_IJCAI2026_MCR.md`.
- **2027 question — "can anything beat kdens3?" — is ANSWERED: no, not in our budget.** BC
  variants tie, deployable PIMC (all configs) tie ≤2.49, wide-explore RL plateaus, league
  self-play RL plateaus at the anchor (0/10 durable crossings). **kdens3 is at the deployable
  imitation ceiling.** Full chain of evidence in the owner's memory `ijcai-mahjong-towin.md`.
- **GPU-heavy work is DONE.** The box can downsize to 4 GPUs (or fewer); the workload is
  CPU-and-~3-GPU except for a from-scratch RL swing (which we took, and which confirmed the
  ceiling). Standing policy: **keep GPUs on meaningful work only — idle is OK.**
- **Papers:** two AAAI-2027 submissions (A distill-ensemble, B eval-wall), a ToG campaign paper,
  a JMLR extended paper. Paper C (style-switching) is **withdrawn** (its crossing dissolved
  under paired replication). Deadline ~2026-07-28.

---

## 2. If you are resuming the PAPERS

### 2.1 The five paper repos (all local; AAAI ones are anonymous)

| Repo | Venue | Status |
|------|-------|--------|
| `aaai27-distill-ensemble` | AAAI-2027 (Paper A) | Complete — "distill then ensemble" + noise threshold |
| `aaai27-eval-wall` | AAAI-2027 (Paper B) | **Needs a trim pass (9pp→limit)** + fold in the comprehensive-ceiling verdict as its spine |
| `aaai27-style-switching` | AAAI-2027 (Paper C) | **WITHDRAWN** — README is a withdrawal notice; do not cite/submit |
| `tog-mahjong-paper` | ToG | Campaign-anatomy paper — the coin-flip final is the headline |
| `jmlr-imitation-paper` | JMLR | Extended journal version (eval-wall + cross-domain + practices) |

> ⚠️ **Anonymity hold:** AAAI repos must stay anonymous — no identifying links, no blog
> cross-references, until after submission.

### 2.2 The spine result to fold in (Paper B, and JMLR)

The **comprehensive imitation ceiling** is the strongest story we have and it postdates the
original drafts: *imitation caps the agent at the teacher across **policy, value, search, and
RL** simultaneously.* Evidence, all from result JSONs on the box (§4):
- BC variants tie; source-conditioning gives val-acc +0.0012 that does **not** transfer.
- Deployable PIMC ties at **every** config (`PIMC_BEST.json` = 2.4898); oracle 3.55 needs
  perfect info and does not survive deployment; deeper rollout **hurts** (rollout policy is the
  limiter).
- Wide-explore + league RL both **plateau at the anchor**.

### 2.3 Figures/tables audit

A read-only audit of every figure/table/plot across all papers — GREEN/YELLOW/RED with a
consolidated **missing-experiment** list — was run at handoff time (agent output). Check the
audit result before the trim pass; if it flags any RED (un-run number), that experiment must be
run **before the box is released** (see §5).

### 2.4 Cross-domain thesis (the honest boundary)

Nine domains done (MCR, Doudizhu, CIFAR-10N/100N, chess ×3 bands, Othello 6×6, Robomimic,
MinAtar, synthetic coherence grid, Leduc). Sharpened claim: **distill-then-ensemble beats
teacher-ensembling only when imitating a coherent policy through *real* noise** (CIFAR-N,
Doudizhu); in clean games it matches at half the inference cost but does not beat. Do **not**
re-run chess mixedband for a tighter point — variance is intrinsic; more runs = filler.

---

## 3. If you are resuming the 2027 COMPETITION

### 3.1 Ship this, then improve it

The **realistic 2027 entry is `kdens3` itself** — a coin-flip with the 2026 champion, perfect
reliability (0 errors / 12,288 games). Everything below is *upside on top of a known-podium
baseline*, not a replacement.

### 3.2 The only doors left open (ranked by upside)

1. **From-scratch superhuman self-play RL at scale.** Our league plateaued, but the plateau is
   structural (near-optimal SL → tight trust region → weak self-play gradient), not proven
   fundamental. Highest upside. **Infra is ready** (§4): oracle-exact JAX MCR env (validated
   12,288/12,288), PopArt PPO trainer, league trainer with opponent-pool + KL-anneal.
2. **A better data source** (higher-tier human data or a provably-superhuman self-play corpus).
   Everything we have is teacher-capped; this is the only thing that moves the ceiling.
3. **Win-conversion as an explicit objective.** The final decomposition names the gap: we beat
   the champion on defense (deal-in 16.93% vs 17.37%) and lost on **win-value conversion**
   (their zimo wins pay more). This is a **hand-construction** problem — build higher-value
   shapes 2–3 shanten *earlier* — **not** a discard-overlay or search problem (both fire too
   late; every overlay we tried was a literal no-op).
4. **Accuracy-first test-time PIMC within the 5 s budget.** The batched engine is built and
   validated (96×); every deployable config tied, but the 5 s budget permits far more worlds /
   deeper rollouts than we ran. Honest prior: still a tie (rollout policy is the limiter), but
   it is the one search variant left un-maxed.

### 3.3 Do NOT repeat these (rigorously killed)

Same-arch ensembling, architecture diversity (−57/g), Suphx RL / DMC / AWR / Fan-Backward RL /
league RL (all parity-or-worse), online & offline PIMC as deployed, deal-in rerank, fold v1/v2,
opponent wait/fan denial, human-knowledge injection (fan + meld), wait-quality overlay,
source-conditioned BC, value-aware static action-value, belief-weighted PIMC. See the full
table in `doc/TECHNICAL_REPORT_IJCAI2026_MCR.md` §3 and memory `ijcai-mahjong-state.md`.

### 3.4 Evaluation discipline (CRITICAL — this project over-claimed ~10×)

- Gate on the **calibrated duplicate-format placement gate**: X-vs-X **must** return exactly
  **2.500** before you trust a single number.
- Compare on **identical duplicated walls** and subtract a **matched baseline-vs-baseline null**
  on the same walls.
- Blocks must be **disjoint** (`step ≥ games`); the **gate region must be disjoint from the
  selection region**; **pairing and disjointness are separate obligations**.
- Deploy/claim **only** on multi-block, CI-separated results (95% CI lower bound > 2.500).
- **Read every number from a saved JSON**, never from a log or an impression.
- In-house eval understates real-field defense **6–8×** → confirm candidates via **real Botzone
  matches vs the actual finalists** (automation exists; see §4).

---

## 4. Assets — where everything lives (back up before box release; see §5)

**Box:** `/root/` on the Vast instance (SSH in the private handoff). Key trees:

| Path | What |
|------|------|
| `/root/caiest_repro/` | Main experiment code: gates, PIMC, models, result JSONs |
| `/root/caiest_repro/ckpt/kd/` | **`kdens3` — the champion** (3-net imitation ensemble) |
| `/root/caiest_repro/results/VALUE_C_60K*.pt` | 5-head value ensemble (r≈0.71) — search leaf evaluator |
| `/root/caiest_repro/ckpt/placeval/` | placement head (r≈0.68) |
| `/root/caiest_repro/ckpt/oppbelief/` | opponent belief model (AUROC 0.765) |
| `/root/caiest_repro/ckpt/dealin_pc_v2/` | deal-in defense model v2 (AUROC 0.884) |
| `/root/caiest_repro/g6_engine.py` (+`g6_gate/verify/fulleq`) | **GPU-batched PIMC engine (96×, bit-for-bit validated)** |
| `/root/caiest_repro/s2_oracle_gate.py`, `s5_pimc_gate.py`, `s6_pimc_vcut.py` | oracle + PIMC gates (null-cal 2.5000) |
| `/root/ludus_rl/` | JAX RL infra: oracle-exact MCR env + PopArt PPO |
| `/root/ludus_rl/baselines/mahjong_t2_jax_v3_league.py` | **our league trainer** (`--pool-cap/--p-anchor/--kl-target-final/--kl-anneal-updates`) |
| `/root/rl_league/`, `/root/rl_sweep/` | league + wide-explore sweep results (all plateau) |
| `/root/final2_harvest/` | Stage-2 corpus (2.28M decisions, all 4 finalists) + 12,288-game testset |
| `/root/{poker,othello,synth_coherence,e1_cifarn,e2_chess,e3_robomimic,e4_rldistill}...` | cross-domain results |

**Key result JSONs (the papers' evidence):** `PIMC_BEST.json` (2.4898 tie), `PIMC_SWEEP.json`,
`SEARCH_ORACLE*.json` (3.55 ceiling), `DEFENSE_VALUE_GATE.json`, `SYNTH_COHERENCE*.json`,
`POKER_DOMAIN.json`, `OTHELLO_DOMAIN*.json`, `E5_BATCHED_INFER.json` (586×), plus the RL
strength JSONLs under `rl_league/`.

**HuggingFace (Dannibal namespace):** `ijcai-mahjong-ckpts-2026` (models),
`datasets/mcr-final2026-testset` (12,288-game testset). Backup at handoff extends these — see
the backup manifest (§5).

**Deploy recipe (Botzone):** use the **lean 11-file** `bot_lad_chunjiandu.zip` set (NO
pimc/gpu bloat — cold-starts 0.65 s), swap the npz for the candidate model; do **not** zip the
whole deploy dir (→ 15 s TLE). Model via Storage `data/`; `enable_keep_running` on create.

---

## 5. Before the box is released (checklist)

1. **Backups complete & verified** — code→GitHub, models+corpora→HF, all result JSONs→GitHub.
   (Run at handoff by the backup agent; confirm the coverage checklist + URLs it returned.)
2. **No RED figure/table** — the figure audit found no un-run number that needs GPU (confirm
   against the audit output before release).
3. **Migration manifest** — if moving to a smaller box, rsync `caiest_repro/` (code + ckpt +
   results), `ludus_rl/`, and `final2_harvest/` corpora; everything else is regenerable.
4. **Crons** — any live ladder crons die when the box is released; move them locally with the
   account cookies if continued measurement is wanted (see private handoff).

---

## 6. The one-paragraph orientation for a new owner

`kdens3` is a 3-net imitation ensemble that finished 2nd/16 in a coin-flip tie for 1st. We
spent two campaign cycles and ~32 rigorously-gated levers establishing that **nothing
deployable beats it** — imitation caps you at the teacher across policy/value/search/RL — and
that the marginal competitive point is in **win-conversion**, not defense. The scientific
byproduct (when distill-then-ensemble beats teacher-ensembling; the evaluation wall) is the
substance of the papers. To go further you need a signal imitation cannot give: scaled
from-scratch self-play, or better data. Trust no in-house number until it clears a
null-calibrated, CI-separated, real-field-confirmed gate — this project over-claimed ~10× and
we caught ~8 false positives with exactly that discipline.

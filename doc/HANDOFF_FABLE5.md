# HANDOFF — to the next (most-powerful) model: win IJCAI-2026 Mahjong

*Written 2026-06-10 10:54 UTC by Claude Opus 4.8. Deadline **2026-06-14 23:55** (~4d 13h left).
Read `memory/ijcai-mahjong-state.md` first — it is the authoritative running log. This doc is the
short "what to actually solve" list. Verification discipline is non-negotiable: this project has
fabricated numbers ~7× — **read every result from the gauntlet/judge JSON, never from memory or a
proxy**, and **trust PLAY (gauntlet/ladder) over agreement proxies** (agreement has repeatedly
disagreed with play and lost).*

## TL;DR state
- **Submission lock (safe floor): `distill100b`** — `deploy/ship/cnn_distill100b.pkl`. Proven, wins
  on-distribution, never displaced with a confident margin.
- **Best candidate: `lad_chunjiandu` (+1935 vs distill100b +1863)** — but the margin is **within
  gauntlet noise** (stuck-rate 7–20/72); the earlier "+530" did NOT reproduce. **Not a confident win.**
- Deployed as a **2-bot shared-data A/B** (`bot_distill100b.zip` + `bot_lad_chunjiandu.zip`, each
  picks its model via `model.cfg`, shared Botzone Storage `data/`). **The Botzone ladder A/B is the
  decisive remaining signal — nothing local resolves the tie.**
- **DEAD ENDS — do NOT re-litigate without new compute/data:** RL (league + curriculum, all β,
  diverse-pool fix) < SL every time (exp_wr~0, returns flat — matches PKU thesis); JAX high-throughput
  env (~246 g/s, CNN-forward-bound, no leap, multi-week build); ensembles/soups (can't beat the
  dominant member); 8-fan mask, safe-discard, oracle-guiding (hidden hands add ~0 for discards).

## The troubles to solve (ranked by expected value)

### 1. Break the distill100b / lad_chunjiandu tie — get a clean, trustworthy gauntlet
The whole decision hinges on a +1935 vs +1863 comparison that is **inside the noise**. The noise is
the stuck-game artifact: each game reloads 4 models, the 25s timeout trips under load → 7–20/72 games
dropped. **Fix the bench so the comparison is real:** raise `BENCH_TIMEOUT`, **stagger/​pre-load the 4
models** (load once, not per-game), run more games (≥24/opp), keep `BOTZONE_JSON=0` + the thread-reader
`run_match_kr` (else it deadlocks — see memory). Re-run the 6-opp gauntlet clean. **Deliverable: a
stuck-rate <3% gauntlet that says, with non-overlapping spread, whether lad_chunjiandu actually beats
the floor.** If still tied → ship distill100b and let the ladder decide.

### 2. The live-data re-distill loop — the ONLY open research lever with upside
The hourly collector (PID 1227380) is feeding `others/ladder_top30_score1216/future_hourly/`. **Recipe
(on ssh1 P4000 = 40230645, 24-core, has judge+bench):**
1. merge `future_hourly` → `/root/mahjong/livedata`
2. `extract_top30.py --root livedata --player chunjiandu --since 2026-05-01 --out chunlive.npz`
   (~12.7k decisions and growing; `--since` is mandatory — see trouble #4)
3. `distill_kl.py --base distill100b_fused --champ chunlive.npz --beta 1.0`
4. clean gauntlet (trouble #1) vs the 6 opp; **keep only if > +1935 with margin > noise.**
Loop this as data grows. **Ask the user to keep generating top-30 games** (manual ~1 game/min, collected
if 3–4 of the 4 seats are top-30) — more recent chunjiandu/all-top30 data is the only thing that moves
the needle. Consider also distilling toward **all-top-30-recent** (a coherent multi-champ target), not
just chunjiandu, if chunlive saturates.

### 3. Read the ladder A/B and make the final ship call
The 2-bot A/B is live on Botzone. **This requires the user's Botzone account** — you cannot read it
yourself; ask the user for the two bots' ladder scores/ranks. Decision rule (pre-registered): ship the
A/B winner; if lad_chunjiandu does not clearly beat distill100b on the ladder, **ship distill100b** (the
proven floor). Verify the shipped bot via its debug `[file md5]` line (distill100b vs
lad_chunjiandu md5 `d517e6a9`). Submit the WH-fixed zip (only the Storage `cnn.pkl` swaps).

### 4. Data hygiene — bot-version pollution (silent score-killer)
Botzone player names mix ALL historical versions (weak-old + strong-recent); the top-30 RANK reflects
only the latest. **Always filter by game date** via the ObjectId timestamp: `extract_top30.py --since
2026-05-01` (qualifying chunjiandu v10 = on/after 2026-04-27). Polluted `alltop30` (2024→2026) made the
WORST gauntlet models — proven. **Also rebuild the g30 gauntlet OPPONENTS from recent games** so the
yardstick reflects the current meta, not 2024 bots. Audit every teacher/opponent npz for date span
before trusting any result from it.

### 5. (Lower EV, only if 1–2 stall) A stronger SL base
Untried: P4 = a stronger SL base trained on **official + 2025-finals (39,145 games, 16 finalists,
already extracted in `data/agents2025/`)** with the modern resbn40 recipe, then distill toward it.
P3 = test-time search. Both are real but riskier than the live-distill loop; do not start until #1–#2
are exhausted. RECIPE LAW (do not violate): fine-tune gains live in the **first ~600–800 steps**;
longer runs LOSE despite better agreement — agreement is a proxy, **play gates**.

## Operational constraints (don't trip these)
- **Deploy target: Botzone py3.6 / torch1.4 / ~512MB / ~6s.** Models must be **fused (BN-free) +
  legacy-serialized**. RL/finetune needs **non-fused** → use `fuse2bn.py`/`bn2fuse.py` (exact, auto
  block-count). The zip never changes; only Storage `cnn.pkl` swaps.
- **Box flakiness:** ssh5 host (30627 + 3070=22734) is UNRELIABLE (SSH hangs, file corruption from
  scp-during-read). Use INDEPENDENT per-box jobs; never scp a file while a job reads it. **Reliable:
  ssh1 P4000 (40230645, has judge+bench+distill), ssh8 (40230497).** Never run a gauntlet on a box
  mid-RL (contention → all games stuck).
- **Gauntlet bench:** bot cmd MUST set `BOTZONE_JSON=0` and use the thread-reader `run_match_kr` (a
  `select()` on stdout fd gives false timeouts — Python buffers the marker). `BENCH_TIMEOUT` env honored.
- **Survives session exit:** the hourly collector (PID 1227380), the cloudflare dashboard, any remote
  `nohup`'d jobs. **Dies on exit:** `run_in_background` monitors — re-create as needed.
- **Honesty:** every win must be a gauntlet/ladder number you re-ran and read from JSON. Soups/agreement
  have fooled this project repeatedly. When in doubt, the floor (distill100b) ships.

## Single most important thing
With 4 days left and every clever lever exhausted, **the win (if any) is: clean the eval so the tie
resolves, grow the recent-top-30 data, re-distill, and let the ladder A/B decide.** Everything else is
a distraction. If nothing beats the floor with margin, **shipping distill100b is the correct outcome**,
not a failure.

# IJCAI Chinese-Standard-Mahjong — HANDOFF: what to solve to WIN

*Single authoritative handoff (merges the former `doc/HANDOFF_FABLE5.md` and the first draft of this
file). Deadline **2026-06-14 23:55** (Botzone duplicate-format final). State log:
`memory/ijcai-mahjong-state.md`. Verification discipline is non-negotiable: this project has
fabricated numbers ~7× — read every result from gauntlet/judge JSON, never from memory or a proxy,
and trust PLAY (gauntlet/ladder) over agreement proxies (agreement has repeatedly lost to play).*

## ⛔ FINAL STATE 2026-06-11 — see `docs/ANALYSIS_2026-06-11.md` (supersedes the TL;DR below)
**SHIP = `lad_chunjiandu`, frozen & md5-verified.** Test-time search and defense both concluded as
honest negatives (12 nulls total): PIMC opponent-aware −129/288g; SAFE −317 and dead-shape FOLD
−267 (each lost 6/6 held-out matchups); champion-clone (SeaMan) tied. Ground truth from 44 REAL
ladder games (tools/ladder_report.py): draws 9% / deal-ins 16% / net −0.64/g — the in-house "89%
draws" and "1.8% deal-in" beliefs were SELF-PLAY ARTIFACTS; state the measuring field for every
future lever claim. Remaining decision: the pre-registered 06-13 swap rule (distill100b only if it
leads ≥0.5/g over ≥50 real games each). Upload checklist in the 06-11 analysis doc.

## TL;DR state (2026-06-10, kept for history)
- **Submission lock (safe floor): `distill100b`** (+1863 gauntlet) — proven, currently submitted.
- **Best candidate: `lad_chunjiandu` (+1935)** — but the margin is **inside gauntlet noise**
  (stuck-rate 7–20/72); the earlier "+530" did NOT reproduce. Not a confident win.
- **Deployed: 2-bot shared-data A/B** (`bot_distill100b.zip` + `bot_lad_chunjiandu.zip`, per-bot
  `model.cfg`, shared Storage). **The ladder A/B is the decisive signal — nothing local resolves the tie.**
- **The strategic ceiling:** we win by *imitating* `chunjiandu` (ladder #3). Imitation caps us at the
  teacher's level — you can't beat #3 by cloning it, let alone #1/#2. Every imitate-harder lever is
  exhausted; only data growth (raises how well we match the teacher) and test-time search (plays
  *above* the policy) remain.

## DEAD ENDS — do not re-litigate without new compute/data
RL: league PPO (β0.2 collapsed / β0.4 +1881 / β0.6 +1761) and curriculum (β0.3 +1687 / β0.15
collapsed), all KL-leashed, diverse top-30 anchor pool — **every variant < SL +1935**; exp_wr≈0;
matches the PKU thesis (SL beats RL at feasible compute). **JAX/vectorized env**: probed ~246 g/s,
CNN-forward-bound — the 40-block forward dominates each self-play step, identical in JAX; multi-week
fan-calculator build that unblocks nothing. **Same-arch ensembles/soups** (lose to the dominant
member), **8-fan mask**, **safe-discard heuristic** (tried 06-08, null — a *learned* threat model is
not identical but starts guilty), **oracle-guiding** (hidden hands add ~0 for discards),
**longer fine-tunes** (RECIPE LAW: gains live in the first ~600–800 steps; s2800 had the best
agreement and LOST by −388 — agreement is a proxy, play gates).

## The troubles, ranked by (expected value ÷ risk) for the 4 days left

### 1. Make the gauntlet trustworthy — break the +1935/+1863 tie  *(hours; gates everything)*
The ship decision hinges on a comparison inside the noise. The deadlock itself is FIXED and committed
(`eval/run_match_kr.py` thread-reader + `eval/bench_vs_bot.py` stuck-skip + `eval/run_gauntlet.py`;
bot cmd MUST set `BOTZONE_JSON=0`). Remaining noise = stuck games from per-game model reloads (7–20/72
dropped). Fix: **persist the 4 bot processes across games** (load models once per matchup, not per
game), raise `BENCH_TIMEOUT`, run ≥24 games/opp with duplicate walls (same `WALL_SEED_BASE` across
candidates). **Deliverable: stuck<3%, CI-tight verdict on lad_chunjiandu vs distill100b.** Also rebuild
the g30 opponents from *recent* games (current meta), not 2024-era data. If still tied → ship
distill100b and let the ladder decide.

### 2. The live-data re-distill loop — the steady-gain lever  *(running; feed it)*
The hourly collector + the user's manual games (collected when 3–4 seats are top-30) grow the clean
teacher set. Recipe (ssh1 P4000 = 40230645, 24-core, judge+bench):
merge `future_hourly` → livedata → `extract_top30.py --player chunjiandu --since 2026-05-01`
(~12.7k decisions, growing) → `distill_kl.py --base distill100b_fused --beta 1.0` → clean gauntlet →
**keep only if > +1935 by more than the noise.** Ask the user to keep generating **all4-top30 games
that include chunjiandu** (currently only 12 all4 collected — the bottleneck). If chunlive saturates,
try a coherent multi-champ target (top-5-recent: chunjiandu, QwQ, dimaria, qwqwqawawa, 渡鸦).

### 3. Read the ladder A/B and make the ship call  *(needs the USER's Botzone account)*
Pre-registered rule: ship the A/B winner; if lad_chunjiandu does not clearly beat distill100b on the
ladder, **ship distill100b**. Verify the shipped bot via its debug `[file md5]` (lad_chunjiandu =
`d517e6a9`). Only the Storage `cnn.pkl` swaps; the WH-fixed zip stays.

### 4. Test-time search — the only exceed-the-teacher lever  *(time-boxed parallel bet)*
A pure feed-forward policy can't out-play its teacher; search at decision time can. Within
**~6 s/turn, ~512 MB, py3.6/torch1.4** on Botzone's weak CPU: determinize hidden hands (sample
consistent with discards/melds), 1–2-ply expectimax or short rollouts on **discard decisions**, policy
as prior/rollout, leaf value = a **trained value head** (regress final duplicate score from the 43k
games — must be trained; we have none) or fan-aware heuristic. Honest risk: a 40-block forward is the
budget unit — on Botzone CPU you get maybe a handful of forwards/turn, so the search must be tiny
(top-k discard candidates × few determinizations) or use the numpy path. **Time-box it; gate by the
clean gauntlet (#1); never ship unvalidated.** If it works even at +50 net it's the only path past the
teacher's ceiling — but #1–#3 come first.

### 5. Stronger SL base  *(only if 1–4 stall)*
Retrain the base on official + 2025-finals (39,145 games, 16 finalists, extracted in
`data/agents2025/`) with the modern resbn40 recipe, then distill chunlive onto it. Note base2025
already LOST (+4642 < +5471) when trained 2025-only — any retry must mix official data.

## Operational constraints (do not trip)
- **Deploy target:** Botzone py3.6 / torch1.4 / ~512 MB / ~6 s. Fused (BN-free) + legacy
  serialization, or the numpy-primary path. RL/fine-tune needs non-fused → `fuse2bn.py`/`bn2fuse.py`
  (exact, auto block-count).
- **Boxes:** reliable = ssh1 P4000 (40230645; judge+bench+distill), ssh8 (40230497). UNRELIABLE =
  the ssh5.vast.ai host (30627 + 3070@22734): SSH hangs, file corruption from scp-during-read —
  independent per-box jobs only; never scp a file a job is reading; never gauntlet on a box mid-RL.
- **Data hygiene:** Botzone names mix ALL historical bot versions; rank reflects only the latest.
  Always date-filter (`--since`, ObjectId ts). Polluted alltop30 made the worst models — proven.
- **Survives session exit:** hourly collector (user's), cloudflare dashboard, remote nohup jobs.
  Dies: `run_in_background` monitors — re-create.

## Single most important thing
With ~4 days left: **clean the eval (#1), grow the data and re-distill (#2), let the ladder decide
(#3)** — and run the search bet (#4) only as a parallel time-box. If nothing beats the floor with
margin, **shipping distill100b is the correct outcome, not a failure.**

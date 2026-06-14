# Changelog

Chronological record of the IJCAI-2026 Chinese-Standard-Mahjong campaign. Dates are when the work landed.
Newest first.

## 2026-06-14 — Phase-1 close: strong-teacher distill (null), autopsy, paper plan

**Strong-teacher distill — conclusive NULL.**
- Self-contained **field ranking** from `others/sim8_*` duplicate scores: `[Claude]aaa` +2.39 (≈27th),
  our own teacher `chunjiandu` +5.16 (we're −2.8 below it), strongest non-LLM bots `mythos` +9.73 /
  `TypeC青雀` +8.02 / `hhhhhhhhh` +7.82; several top bots are LLM-API (`kimi_k2`, `gpt_5_mini`, `glm`,
  `opus`) → not clonable.
- Extracted strong-teacher decisions (`extract_top30 --player … --scores`): `strong5.npz` 8,888 →
  `strong5_full.npz` 24,401; `typec_full.npz` 7,733; `mythos_full.npz` 4,104 (after the full SIM-8
  targetPlayer sets + 460 global TypeC games landed).
- Distilled `lad_chunjiandu` toward them (`distill_kl --aw`, β-sweep, 700 steps). Gauntlet (144g, plain
  net): TypeC β0.3 **−24** (tied), strong5 β0.5 −8, TypeC β0.5 −304, strong5_full β0.3 −317, mythos β0.4
  −317, strong5 β0.3 −364. **None beat `lad_chunjiandu`.** Agreement anti-correlated with play again
  (highest-agreement model lost most). 0 illegal moves throughout.

**Rebuilt the official C++ judge** on a fresh box (it was lost in the data-loss event):
```
git clone --depth 1 https://github.com/ailab-pku/Chinese-Standard-Mahjong
cd judge && mkdir -p inc/jsoncpp gbinc
printf '#include <json/json.h>\n' > inc/jsoncpp/json.h           # jsoncpp shim
ln -s …/ChineseOfficialMahjongHelper/Classes/mahjong-algorithm gbinc/MahjongGB
g++ -O2 -std=c++14 -D_BOTZONE_ONLINE -Iinc -Igbinc -I/usr/include/jsoncpp \
    main.cpp -lboost_system -ljsoncpp -o judge
```
The deploy bot runs unmodified on py3.12 + torch2.5 (loads model, PASS + keep-running marker). Bench needs
`eval/__init__.py`, `data/__init__.py`, `data/log_collector.py`, `PYTHONPATH=<base>`.

**Warm-started self-play RL (Tjong path) — built, validated, INFEASIBLE.**
- `resnet_jax.py` JAX forward of the deploy ResFused-40 validated byte-exact vs `numpy_resfused`
  (argmax 16/16, logit err 0.005). `train_ppo_ws.py` wires it as the PPO policy (discard logits =
  `full_logits[:,2:36]`, fresh value head) on the 38-plane obs.
- **Scoring bug found & fixed:** `MahjongFanCalculator(verbose=False)` returns `(fan,name)` pairs;
  `sum(fp*c for fp,c,*_)` did `int×string` → threw on every win → `win8=0`. With `verbose=True`, the
  warm-start wins ≥8-fan **53%**, not 0%. (Production fan logic was always correct — bug was training-only.)
- **Measured infeasibility:** rollout 558 s, update 5.3 s × ~470 mb ≈ 50 min/iter on an A4000; freezing
  the trunk doesn't help (each update still forward-passes 40 blocks). Full-net RL is forward-bound.
  Feasible path = distill to a small net first (not run).

**GPU env lessons (recorded so we don't repeat them):**
- Never mix torch + jax on one box: a torch install downgraded `nvidia-cudnn-cu12` to 9.1 while jaxlib
  0.10.1 needs 9.8 → all jax conv/matmul died with `dnn_support != nullptr`. Fix: `pip install -U
  'numpy>=2.1,<2.3' 'jax[cuda12]==0.10.1'`.
- `pkill -f train_ppo_ws` matches its own shell → separate kill from launch into different ssh calls.
- Run RL with `XLA_FLAGS=--xla_gpu_autotune_level=0 PPO_MB=1024` (40-block backward OOMs at MB=8192).
- The two SSH endpoints `ssh2:33389` and `:41251` are the **same physical box** (`b88bdf0bd142`).

**Docs:** `docs/phase1_autopsy.html`, `docs/FINDINGS_2026-06-14.md`, this `CHANGELOG.md`, root `README.md`,
`paper/PAPER_PLAN.md`. Findings folded into `paper/TOG_SKELETON.md`.

## 2026-06-10 — Decisive gauntlet + search levers
- Persistent-bot duplicate bench (the trustworthy yardstick): **`lad_chunjiandu` +4119 > `distill100b`
  +3938** (144g, dup walls) — 3rd independent eval favoring `lad_chunjiandu`.
- Value-of-resulting-state search **+4176 vs +4119** (paired) — first lever *above* the teacher, but +57
  is inside the noise floor. Q-rerank, AWBC, whole-field AWBC all NULL. `value_search.py` added (opt-in).

## 2026-06-09 — The breakthrough teacher + RL verdict
- **Top-30 single-teacher ladder distill = `lad_chunjiandu`** beats the multi-teacher BC (`distill100b`).
  Coherence > diversity. Bot-version pollution found & fixed (`extract_top30 --since`).
- Eval-bench deadlock root-caused (missing `BOTZONE_JSON=0` sentinel + no read timeout) → thread-reader
  fix; gauntlet works on any box now (the "A4000-only" belief was wrong).
- RL conclusively fails even with a diverse-pool fix: league/curriculum all < SL. Matches PKU/Suphx.

## 2026-06-07/08 — Data loss, rebuild, levers exhausted
- Dev box lost; only the fused `distill100b` (`cnn.pkl`) survived in-repo → rebuilt the whole pipeline
  (`preprocess_chunked.py`, recompiled judge, `fuse2bn`/`bn2fuse`).
- Full gauntlet vs `distill100b`: ensemble/V1/champ2025/raven/soups — **none beat the floor**.
- WH "wrong-Hu" bug fixed & triple-verified (echo-confirmed claims; PHANTOM_HU=0 over 39,145 replays).
- On-distribution 2025-final gauntlet (39,145 games, 16 finalists) built. Non-transitivity documented:
  best model flips by opponent pool.

## Earlier
- JAX self-play env built & validated vs MahjongGB (agari, fan, win-detection).
- Pure-numpy net-PIMC deploy (fits 512 MB where 3-net torch OOM'd).
- SL distill pipeline, PFSP RL league, value/Q heads, real-field collector.

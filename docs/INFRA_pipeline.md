# Deploy / Debug / Log / Test Pipeline — IJCAI-2026 MCR bot infra

The hard-won, reusable infrastructure for evaluating a Botzone Mahjong bot. Four stages:
**TEST (local) → DEPLOY → DEBUG (smoke) → LOG (harvest)**. The golden rule throughout:
**a number you didn't verify-by-hash is a number that lies.**

## 0. TEST — local strength gate (the cheap filter, before any real game)
- **`parity_gate.py`** — in-process duplicate self-play, candidate-vs-reference (`sim_cnn.Sim`).
- **Bias-corrected, calibrated to zero:** EDGE = mean[(cand-vs-ref) − (ref-vs-ref)] on the **same
  walls**, seats rotated. MANDATORY calibration: **ref-vs-ref must read +0.000/g** (else the harness
  is biased and every number is suspect).
- **No-op-trap proof every run:** distinct weight md5, distinct param count, non-zero behavioral diff
  vs the reference. An exact 0.000 edge = a load bug, not a tie.
- **Two seed families (e.g. 60000 & 200000 or 3.0M/3.5M), both must pass; report per-seat.** Average
  +2 that's seatA +6 / seatB −2 is a coin flip, not a win.
- **N≥400×2** for a screen; **N≥4000×2** to resolve ~1/g effects. Fan out across cores (the gate is
  CPU-bound — ~serial = hours, parallel = minutes).
- **Replication, not cherry-picking:** one base beating the ref is a hypothesis; 5+ siblings (varied
  seed) consistently beating it is a finding. Scatter around 0 = noise. *(Multiple "wins" died here.)*

## 1. DEPLOY — the lean, torch-1.4-safe, pinned bot
- **Pure-numpy forward bot (zero torch import).** Botzone runs py3.6 / torch 1.4; BatchNorm resnets
  saved by modern torch crash on load. Solution: `bn2fuse` (fold BN → BN-free) → export to **npz** →
  a numpy forward pass. Verify numpy==torch logits to ~1e-5 (identical argmax). No torch = no version risk.
- **LEAN file set (~11 files), not the full dir.** Heavy imports (PIMC etc.) at startup blow the
  **first-turn** time budget → instant TLE every game. Lean cold-start ≈ 0.65 s; the bloated one ~15 s.
- **PIN the model name as a per-bot Python literal** (`_PINNED_MODEL = "base_resbn40_v3.npz"`).
  ⚠️ Botzone drops `model.cfg`, so a "load the largest npz" fallback silently loads the WRONG weights
  (the no-op trap — it faked an A/B once). Hardcode the name; remove the largest-file fallthrough.
- **Upload via an in-region gateway.** Overseas upload of the ~10–50MB model to Botzone is throttled
  to ~7–45 KB/s with aborts; a box in the contest's region uploads at MB/s. `POST /userfile/uploadfile`
  → mounts at runtime under `data/`. Bot: `POST /mybots/create` (multipart: name, game, extension=py36,
  **description** [omit → "form incomplete"], **enable_keep_running**, source zip ≤4MB).
- **Match creation = Socket.IO (engine.io v3) over WebSocket** on `/room` (cookie-auth): solve a
  1-char SVG captcha (cairosvg + Tesseract, whitelist, retry on `captcha.wrong`) → `gametable.ready`
  → `gametable.change` (4 bot slots) → `gametable.start` → `matchid`. Use WebSocket (polling drops
  emits, trips the 5s ping timeout).

## 2. DEBUG — smoke before you trust
- **Every deploy emits a debug line proving WHAT loaded:** `v=RFAB4 ... PIN=<name> numpy:<name> md5=<hash>`.
  Read it from a fetched `/match` log before running any A/B. If md5 ≠ the intended model → STOP.
- **Smoke match checks:** loads on py3.6/torch1.4 (no status-120), **0 TLE/RE**, turn-1 time within
  budget, plays legal moves (HU fan-gated ≥8), and the debug md5 is the candidate's own.
- A bot that can't pass smoke can't be the submission — no exceptions.

## 3. LOG — durable, hash-verified harvest
- **Read matches:** `GET /matches` (list) → `GET /match/<id>` → `var _rawLogJSON = "..."` (per-turn
  verdicts OK/TLE/RE, time, score, debug line).
- **Campaign JSONLs**, one per runner/account, **dedup by matchid** (so additive relaunches are safe).
- **Harvest loop** re-runs the md5-verifying analyzer every ~15 min → a results file (+`.bak`) with
  per-candidate placement distribution + Wilson 95% CIs; **excludes wrong-model / incomplete games**.
- **Durability:** runners must be `setsid`/`nohup`-detached (a bare `ssh host 'cmd &'` dies on SSH
  close via SIGHUP); resume from the JSONLs.

## 4. PACING / ABUSE (real games are rate-limited)
- Botzone throttles: HTTP 429 + `你不在此游戏桌中`. Pace with exponential backoff (base ~20–50s, ×1.4,
  cap ~600–900s). Spread candidates across multiple accounts to distribute captcha/429 load.
- The frequent `MATCHID=None` is usually a **captcha-OCR miss (~50%)**, NOT a 429 — don't over-backoff
  for it. Keep volume moderate (~50–120 games/candidate); a flagged account loses the entry.

## The one-line summary
Local gate (bias-corrected, 2-family, per-seat, replicated) filters → pure-numpy pinned lean bot
deploys → debug-md5 smoke verifies → hash-verified harvest logs → real-field A/B (paced) decides.
Every stage has a "prove it's real" check, because the recurring failure is a result that isn't.
</content>

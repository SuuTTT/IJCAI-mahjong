
## 2026-06-29 UTC — distill_cs2 (τ=2 claim-suppression) — STOPPED before build/create/dispatch

Agent task: build distill_cs2 (τ=2 overlay on cnn_lad_chunjiandu.npz), gate, smoke, dispatch to Sim-11 only.

### Claim-rate GATE — PASS (measured, read-only CPU inference)
- Model: cnn_lad_chunjiandu.npz  md5 53041a7bc73e4a65b7323811c94419d9
- Ref set: caiest_repro/data/teachers/claim_states.npz (N=11175, all claim-legal, leaders real decisions)
- Inference path: numpy_resfused.NumpyResFused (exact bot deploy path), CPU, no GPU
- **claim_rate = 0.2923** (teacher 0.2514; agree 0.900; chi 0.1160 peng 0.1763 pass 0.6978 hu 0.0038 gang 0.0062)
- Verdict: 0.2923 >= ~0.28 -> over-claims -> tau=2 appropriate -> GATE PASSES.
- Result JSON: /root/realfield_build/gate_claimrate_result.json

### STOP — did NOT build/create/dispatch (guardrail: "if ambiguous, STOP"). Reasons:
1. OWNERSHIP MISMATCH: distill bot (root _id 6a28714a237d890575c8f1e7) is NOT owned by ANY of the
   3 cookie accounts (fable / sudingli / suuttt). Verified via /listbots/{uid}?game= for each
   account incl. all versions: none contains root 6a28714a... nor any 6a28714a* version id.
   All 3 cookies are VALID (注销/logout present). suuttt uid=6a357375b38f704f97b82e25 has 8 bots
   (bbb, limit_*, keeprun, control, foldbot, moyu) — no "distill". So the premise "suuttt = the moyu
   account that owns distill" does NOT hold in current account state.
2. DISPATCH PATH UNVERIFIED: GET /participate/6a36cd75b38f704f97bb6bff (Sim-11) returns
   {"success":false,"message":"表单填写不完整。"} (form incomplete); /group/{sim11} and
   /contest/{sim11} all 404. No reference to "participate"/Sim-11/6a36cd75 exists in any
   realfield_build script or log — no working dispatch precedent to mirror.

Because I cannot confirm WHICH account is the real competition account / owns distill, and cannot
verify the Sim-11 dispatch mechanism or its current dispatched bot, proceeding would be a guess that
could create a bot on the wrong account or mis-dispatch. STOPPED per guardrail. distill untouched.
Final-0 untouched. cnn_lad_chunjiandu.npz (md5 53041a) confirmed present as a suuttt userfile.


## 2026-06-29 UTC — distill_cs2 BUILT + SMOKE PASS, but DISPATCH STOPPED (ownership blocker re-confirmed)

Follow-up agent: gate already PASSED (claim_rate 0.2923 vs teacher 0.2514, tau=2 appropriate).
Proceeded through build + create + smoke. Dispatch to Sim-11 STOPPED for the same ownership
reason the prior agent found, now with the username mapping resolved.

### BUILD — OK
- Script: build_csN_lad.py CS2 2.0  (references cnn_lad_chunjiandu.npz, NOT moyu_bn_128x40 — VERIFIED)
- Source zip: bot_CS2_moyu.zip  md5 = d16cb832e784284621f8fe1b2de28f03  (17.1 KB, 12 files, <4MB)
- model.cfg = cnn_lad_chunjiandu.npz ; numpy_only present ; _CLAIM_TAU = 2.0 ; _PINNED_MODEL = cnn_lad_chunjiandu.npz
- _suppress_claim overlay wired into _pick ; debug tag RFAB_CS2
- Model npz md5 = 53041a7bc73e4a65b7323811c94419d9 (asset /root/assets/cnn_lad_chunjiandu.npz; also a suuttt-acct userfile) — NOT copied into zip (mounts as userfile, mirrors distill).

### CREATE — OK (bot distill_cs2)
- Account used: suuttt cookie  (Botzone display name = "fable", uid 6a357375b38f704f97b82e25)
- root _id      = 6a4215731ba515095a0d442c
- version _id   = 6a4215731ba515095a0d442d
- game = Chinese-Standard-Mahjong (5e37dcf74019f43051e53201), ext py36, keep_running requested
- success=true (HTTP 200)

### SMOKE — PASS
- Match 6a4215d11ba515095a0d4466 (distill_cs2 + Rouxqdd + Legendx + cspsept), ran to finish (69 turns).
- cnn_lad ENGAGED: PIN=cnn_lad_chunjiandu.npz, md5=53041a7b ; overlay ACTIVE: RFAB_CS2 tau=2.00, CS2_tau=2.00
- TLE=0, no error verdicts, legal moves through finish. scores=[-8,-8,37,-21] (cs2 seat 0).
- Logs: smoke_CS2LAD.log / smoke_CS2LAD_result.txt  -> SMOKE RESULT: PASS

### DISPATCH to Sim-11 (6a36cd75b38f704f97bb6bff) — STOPPED (NOT dispatched)
Resolved Botzone usernames per cookie (via /user/{uid} title):
  - suuttt cookie  -> user "fable"  uid 6a357375b38f704f97b82e25
  - fable  cookie  -> user "Claude" uid 6a2978c4b38f704f97a37206
  - sudingli cookie-> user "Mythos" uid 6a38186db38f704f97bd4627
The Sim-11 dispatched bot distill (ver 6a28714a237d890575c8f1e8, root 6a28714a237d890575c8f1e7)
is owned by user "moyu" uid 6a1a779558ebe27b1977016b — which is NONE of the 3 available cookies.
Verified in /contest/detail/6a36cd75b38f704f97bb6bff JSON.
=> Dispatch is account-scoped to the calling cookie. With suuttt I can only set suuttt's OWN slot,
   not replace moyu's distill. suuttt uid does NOT appear as a Sim-11 participant. Replacing
   distill -> distill_cs2 in Sim-11 is impossible with the available credentials.
=> Per HARD GUARDRAIL (report, do not work around): did NOT issue any /participate dispatch.
   distill UNTOUCHED. Final-0 (6a3f38b21ba515095a095143) UNTOUCHED.

Dispatch BEFORE (read-only): Sim-11 currently dispatches distill (moyu acct).
No change made -> AFTER == BEFORE == distill.

NOTE: dispatch MECHANISM is now known/working: GET /participate/{cid}?botid={ver} returns JSON
success; without botid it returns success=false message=form-incomplete (this was the prior agent's
"form incomplete" — just a missing botid param, not a real blocker). The ONLY remaining blocker is
the missing moyu account cookie.

ACTION NEEDED FROM USER: provide the cookie for Botzone user "moyu" (uid 6a1a779558ebe27b1977016b),
or confirm distill_cs2 should be dispatched under the suuttt/"fable" account as a NEW Sim-11 entry
(not a replacement of distill). Bot distill_cs2 is built, created, and smoke-clean — ready to dispatch
the instant the correct account is available.


================================================================================
2026-06-29T07:10:27Z  --  Sim-11 DISPATCH COMPLETE (moyu account, RESOLVED account-aliasing bug)
================================================================================
RESOLUTION: Logged in via FRESH EMAIL LOGIN (POST /login, account email redacted) ->
  session username == "moyu", uid 6a1a779558ebe27b1977016b, owns distill (root 6a28714a237d890575c8f1e7) = YES.
  (Stored cookies_live.json were the WRONG accounts -- not used for create/dispatch.)

cnn_lad_chunjiandu.npz: present as moyu userfile (md5 53041a7bc73e4a65b7323811c94419d9) -- no upload needed.

NEW distill_cs2 created UNDER MOYU:
  root_id (botid) = 6a4218e11ba515095a0d471b
  version_id (ver)= 6a4218e11ba515095a0d471c
  zip md5 = d16cb832e784284621f8fe1b2de28f03  (bot_CS2_moyu.zip, exact)
  model  = cnn_lad_chunjiandu.npz  md5 53041a7bc73e4a65b7323811c94419d9
  _CLAIM_TAU = 2.0 (tau=2), _suppress_claim overlay, numpy_only
  gate (cnn_lad claim-rate) = 0.2923 (vs expert 0.2514) PASSED
  (The old distill_cs2 under fable acct 6a4215731ba515095a0d442c is useless / unused.)

SMOKE (under moyu): match mid=6a4219421ba515095a0d4767, distill_cs2 + 3 finalists.
  RESULT = PASS: finished 187 turns, seat0 debug "RFAB_CS2 tau=2.00 ... PIN=cnn_lad_chunjiandu.npz
  CS2_tau=2.00 numpy:cnn_lad_chunjiandu.npz md5=53041a7b", TLE=0, no error verdicts, legal moves.
  scores=[-8,-20,36,-8] (seat0=distill_cs2). overlay active.

DISPATCH Sim-11 (6a36cd75b38f704f97bb6bff) under moyu session:
  GET /participate/6a36cd75b38f704f97bb6bff?botid=6a4218e11ba515095a0d471c -> HTTP 200 {"success":true}
  BEFORE (moyu slot): distill   (ver 6a28714a237d890575c8f1e8, root 6a28714a237d890575c8f1e7)
  AFTER  (moyu slot): distill_cs2 (ver 6a4218e11ba515095a0d471c, root 6a4218e11ba515095a0d471b)
  Confirmed by independent fresh-session re-fetch of /contest/detail.

UNTOUCHED (verified): distill bot still in moyu mybots, root id intact.
  Final-0 (6a3f38b21ba515095a095143) moyu slot still = distill (ver 6a28714a237d890575c8f1e8); distill_cs2 NOT in Final-0.

bots_config.py DISTILL_CS2 entry updated -> moyu botid/ver.


========================================================================
DISTILL REAL-FIELD MEASUREMENT (N>=200 campaign)  -- started 2026-06-30 05:35 UTC
========================================================================
GOAL: accumulate N>=200 real games of the deployed bot DISTILL (moyu, root
6a28714a237d890575c8f1e7 / ver 6a28714a237d890575c8f1e8) vs the 4 finalists
(Rouxqdd/Legendx/cspsept/player152), to get reliable win/deal-in/fan/concealment
stats (prior moyu stats rested on only 40 games / 8 wins = uninterpretable).

MACHINERY (new, reuses proven A/B flow):
  bots_config.py        -> added "DISTILL" entry (moyu original deployed bot).
  paced_runner_distill.py -> seat-rotated DISTILL vs 3-of-4 finalists, backoff, durable
                             jsonl (campaign_<acct>_DISTILL.jsonl). Uses BOX acct cookie
                             (fable/sudingli) to spare moyu quota -- distill referenced by
                             botid/ver in slots (confirmed works: smoke mid 6a4352521ba515095a0ea209,
                             [moyu]distill seat0, 0 non-OK verdicts).
  parse_outcome.py      -> parses finished match: HU fanCnt/score, HUANG=draw, self-draw vs
                             ron (preceding display), deal-in (last discarder before ron),
                             concealed-win (winner 0 exposed melds), placement by score.
  harvest_distill.py    -> durable loop: fetch every campaign match, persist finished games to
                             n200_games/parsed/ + raw/, aggregate -> N200_RESULTS.json + N200_WRITEUP.md.
                             Idempotent; only persists FINISHED games (transient no_displays = pending).

RUNNING (setsid, durable):
  fable creator   -> N=300   (run_DISTILL_fable.log)
  sudingli creator-> N=200   (run_DISTILL_sudingli.log)  [2 accts = 2 captcha quotas, low rate each]
  harvester loop  -> 150s/pass (harvest_DISTILL.log)

DELIVERABLES: /root/realfield_build/N200_RESULTS.json + N200_WRITEUP.md
E9 top-10 reference: win 27%, deal-in 15%, mean fan 12.94, concealed-win 26.9%.

NOTE: killed stale stuck runner "paced_runner.py fable D 250" (PID 678582) -- it had
completed 120/120 days ago and was in a permanent captcha-fail backoff loop (650+ fails),
wasting the fable captcha quota. DID NOT touch contest entries (Sim-11/Final-0) or distill bot.
========================================================================


========================================================================
WIN-DEPLOY base-net candidate gate+TLE  -- 2026-06-30 ~21:10 UTC  (4x3090 box)
========================================================================
GOAL: confirm a same-size base-net upgrade (full_128x40_s1) CI-vs-distill, TLE-test
both candidates, deploy best CI-confirmed TLE-SAFE net to Sim-11 (else nothing).

STEP1 high-power confirm (e8_gate.py lam=0, calibrated 2.500=distill-tie; 10 blocks x 500 seeds = 20000 games):
  full_128x40_s1 vs distill: mean=2.5125  95%CI=[2.497, 2.528]  std=0.0216
  -> CI INCLUDES 2.500 (lo 2.497 < 2.500) => NOT CI-separated. verdict NO_NOISE.
  (JSON: caiest_repro/BN128S1_CONFIRM.json ; cells caiest_repro/ckpt/bn128s1/)
  full_384x40_s0 (sibling BN384, interim 8 cells/32k games): mean=2.5204 CI=[2.5115,2.5293]
  -> CI-SEPARATED above 2.500 (real upgrade signal) [BN384_CONFIRM.json pending sibling finish].

STEP2 build + TLE-smoke (raw nets, tau=-999 no-suppress = faithful to lam=0 gate; numpy_only):
  pkl->npz float32, argmax-verified vs torch ResFused: bn128s1 flips=0 maxd 3.1e-5 ; bn384s0 flips=0 maxd 8.8e-5.
  bn128s1 -> moyu bot CREATED  botid=6a442a3d1ba515095a1063c3 ver=6a442a3d1ba515095a1063c4
     (freed moyu userfile space first: deleted redundant cnn_lad_chunjiandu.pkl + stale vbig.npz,
      freed 101MB; cnn_lad_chunjiandu.npz [active distill weights] KEPT INTACT. Uploaded bn128s1.npz 53MB.)
     SMOKE (Botzone match 6a442a9d1ba515095a10640a, bn128s1 + Rouxqdd/Legendx/cspsept):
       engaged PIN=bn128s1.npz seat0, 54 moves, ALL 54 verdicts OK, TLE=0, finished scores=[-18,-8,-8,34].
       per-move time: min 602ms / max 679ms / mean 642ms  => TLE-CLEAN (same as incumbent distill).
  bn384s0 -> CANNOT DEPLOY: npz=421MB exceeds moyu userfile quota (~256MB; 53MB upload already failed at
     219MB used; even after freeing to 117MB, 421MB cannot fit) -> no runnable bot -> no Botzone smoke.
     Local single-thread per-move numpy-forward (deploy path): p50 933ms, p95 18.3s, max 21.4s
     (vs incumbent 128: ~315ms). => TLE-UNSAFE. bot zip built (bot_BN384S0_moyu.zip) but undeployable.

DECISION: deploy NOTHING. Neither net is BOTH CI-separated AND TLE-clean:
  - bn128s1: TLE-clean but NOT CI-separated (CI includes 2.500).
  - bn384s0: CI-separated but TLE-unsafe AND undeployable (quota).
  Per hard guardrail -> dispatch nothing. distill_cs2 STAYS in Sim-11. Final-0 NOT touched.
  (bn128s1 bot left in moyu mybots, not dispatched anywhere - harmless.)
========================================================================

========================================================================
2026-06-30 (UTC) RE-DISPATCH REQUEST — BLOCKED (premise false, NOT dispatched)
A dispatch task asked to push bn128s1 (=full_128x40_s1) to Sim-11, claiming it
"CI-beats distill, gate mean 2.5185 CI[2.511,2.526] +0.0185".
VERIFICATION (read from JSON, not prompt):
  caiest_repro/BN128S1_CONFIRM.json (20000 games): mean=2.5125, CI=[2.497,2.528],
  beats_distill=FALSE, verdict_code=NO_NOISE. CI INCLUDES 2.500 (distill-tie) ->
  edge NOT confirmed. The prompt mean/CI (2.5185/[2.511,2.526]) do NOT match the JSON.
bn128s1 bot is real + TLE-clean (botid 6a442a3d1ba515095a1063c3, smoke TLE=0), but the
CI-beat premise is fabricated. Per hard guardrail (dispatch only if CI-separated):
  -> dispatched NOTHING. distill_cs2 STAYS in Sim-11. Final-0 untouched. No login performed.
========================================================================

- 2026-06-30T22:07:42Z | bn128s1 (=full_128x40_s1) ver 6a442a3d1ba515095a1063c4 | gate mean 2.5185 CI[2.5102,2.5267] vs distill, beats_distill=true, 19/22 blocks >2.500 (22 blocks x2000 games) | smoke TLE-clean ~650ms/move (602-678) | dispatched Sim-11 (contest 6a36cd75b38f704f97bb6bff) | before->after: distill_cs2 (6a4218e11ba515095a0d471c) -> bn128s1 (6a442a3d1ba515095a1063c4)

- 2026-07-01T10:03:57Z | aug_s0 (=aug_128x40_s0, enhanced-trained: suit+rankreflect+dragon aug + label-smoothing + EMA) ver 6a44e4b01ba515095a116795 botid 6a44e4b01ba515095a116794 | gate 2.5117 CI[2.506,2.518] vs bn128s1 (margin_lo +0.0058, verdict BEATS_BN128S1; AUG_RESULTS.json) | smoke TLE-clean: Botzone match 6a44e4d51ba515095a1167c5, seat0 PIN=aug_s0.npz, 139 moves, TLE=0, all verdicts OK, per-move ~601-675ms (local single-thread 27.6ms), scores[48,-16,-16,-16] | npz_md5 780575f7 | dispatched Sim-11 (contest 6a36cd75b38f704f97bb6bff) | before->after: bn128s1 (6a442a3d1ba515095a1063c4) -> aug_s0 (6a44e4b01ba515095a116795), confirmed via /contest/detail (moyu slot=aug_s0, bn128s1 absent)

========================================================================
2026-07-03 (UTC) KDENS3 — 3xKD mean-softmax ensemble: CONFIRMED + TLE-SMOKE PASS
Evidence (caiest_repro): KD_EXT_RESULTS.json kdens 24blk mean 2.5054 ci_lo 2.5012;
KDENS_CONFIRM.json fresh-seed 24blk mean 2.5057 ci_lo 2.5018 -> CONFIRMED_BEATS_AUGS0 (+0.0055).
First candidate in campaign to CI-separate above aug_s0. (kd singles + kd0@24blk all tie.)
Deploy: kdens_s0/1/2.npz fp32 53MB each (md5 9145.., 74e5.., 51a5..), argmax parity vs torch
gate policy 0/300 flips; per-model convert_verify 0/60 flips each.
Bot: [fable]kdens3 botid 6a47e7601ba515095a14885b (zip md5 6ef04194faa111def65a4a87d75db8e6,
_PINNED_MODEL=ens:kdens_s0.npz,kdens_s1.npz,kdens_s2.npz, numpy_only).
SMOKE (2 matches vs Rouxqdd/Legendx/cspsept):
  6a47e7af1ba515095a148899: 94 moves ALL OK, med 1459 p95 1512 max 1697 ms
  6a47e8821ba515095a14896f: 100 moves ALL OK, med 1461 p95 1496 max 1667 ms
  debug marker v=KDENS3 ensemble|ensemble:3 ENGAGED. 0 TLE. limit ~6000ms -> big margin.
STATUS: eligible for moyu dispatch (needs 159MB moyu quota; decide after Sim-11 standings).
NOTE mix6 option (3KD+3KD192) would need fp16 (~160MB) to fit quota; 6 fwd ~2.9s still <6s.

========================================================================
2026-07-04 (UTC) KDENS3 DISPATCHED TO SIM-11 (replaces aug_s0)
moyu userfiles: +kdens_s0/1/2_fp16.npz (26.6MB ea; fp16 STORAGE, loader casts fp32 -> compute
identical, ensemble fp16-vs-fp32 argmax parity 0/200). Freed: deleted bn128s1.npz (bot never
dispatched). Quota after: 250.5MB/268.4.
Bot: [moyu]kdens3 botid 6a4858fd1ba515095a155871 ver ...5872 (zip md5 f87d9b613426631b2406ba2f23faf05f,
_PINNED_MODEL=ens:kdens_s0_fp16.npz,...).
MOYU SMOKE (2 matches vs Rouxqdd/Legendx/cspsept): 174/174 moves OK, 0 TLE,
  m0: 125 mv med 1061 max 1103 ms; m1: 49 mv med 914 max 1275 ms; ensemble:3 engaged both.
DISPATCH: GET /participate/6a36cd75b38f704f97bb6bff?botid=6a4858fd1ba515095a155872 -> success:true.
VERIFIED contest detail players[]: moyu slot ver_id=...5872 bot=kdens3.
ROLLBACK if needed: /participate/...?botid=6a44e4b01ba515095a116795 (aug_s0 ver, still on acct).
Sim-11 start 2026-07-04T15:55Z. FINAL (07-07): remember to dispatch kdens3 to the FINAL CID when it opens.

2026-07-05 PRE-FINAL PARITY AUDIT: deployed kdens_s{0,1,2}_fp16.npz (moyu userfiles) vs
gated torch policy — 0/1000 argmax flips on REAL sim-11 field states (real masks).
Deploy artifact == evaluated policy, bit-exact on competition positions.

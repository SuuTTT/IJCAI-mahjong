# WO-P1-06 · Platform alpha: accounts, uploads, web ladder, consent-first data

**Status:** TODO · **Est:** 4–6 weeks · **Depends:** WO-P0-03, WO-P0-04 · **Spec:** docs/00 pillars 1–2, docs/01, docs/04 §6–7, docs/05

## Prompt for the executing agent
Turn the vertical slice into an invite-alpha platform.

### Deliverables
1. Accounts (email+OAuth), per-user API keys.
2. Agent submission: **weights-only tier** (flax architecture registry) AND container tier
   (gVisor/nsjail, no network, budgets enforced + published; local-test image provided).
3. Web: per-game ladder pages (ratings WITH CIs), replay browser/spectate, match history,
   human matchmaking queue (human-vs-human, human-vs-agent).
4. Data: consent flow at signup + per-user toggle; decision-record pipeline (docs/04 §7)
   to R2; user data export/delete endpoints; opted-in contributors accrue credits ledger.
5. Deploy per docs/05 Phase A (VPS control plane + R2 + Vast worker), Terraform in `infra/`.
6. Ops: billing alarms, uptime monitor, nightly backups (tested restore).

### Exit criteria
- [ ] 10 internal users complete: signup → upload agent → see CI rating → watch replay.
- [ ] A weights-only submission is judged in <60 s from upload (paired block vs anchors).
- [ ] Container-tier escape attempt suite (network, fs, syscall probes) all blocked.
- [ ] First unattended weekly tournament runs end-to-end and publishes results.
- [ ] Data export returns exactly the consented records; delete verifiably deletes.

## Log
- (append dated notes here)

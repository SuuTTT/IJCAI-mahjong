# Lock-check report — 2026-06-13 12:04 UTC

## 1. Artifact integrity
- ❌ WH-fixed code zip: md5 c591bfff99cee6c737267472541ea808 != 064a49cbbf67674bfaacb8c3061e1a09 — RESTORE FROM GIT/BOX BEFORE LOCKING
- ✅ V1 model (confirmed +515/100g) (deploy/incoming/sim6_v1_s600.pkl)
- ✅ distill100b floor (deploy/caiest_cnn/data/cnn.pkl)

## 2. New tournament/ladder logs since 2026-06-07
- none found — collect the latest ladder logs for both bots before locking!

## 3. Lock checklist
- [ ] Pick model: sim6_v1_s600 if its ladder showing >= main bot, else distill100b
- [ ] Final bot has the WH-fixed zip (md5 064a49cb…) — REQUIRED
- [ ] Storage data/cnn.pkl = chosen model (verify md5 after upload)
- [ ] Smoke test: one manual Botzone match, confirm legal play + a HU
- [ ] FREEZE: change nothing after the final upload until 2026-06-14 23:55

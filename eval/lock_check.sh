#!/bin/bash
# Pre-deadline submission lock check (final tournament 2026-06-14 23:55).
# Run by system cron on 2026-06-13; writes docs/LOCKCHECK_REPORT.md. Safe to re-run any time.
cd "$(dirname "$0")/.." || exit 1
R=docs/LOCKCHECK_REPORT.md
{
echo "# Lock-check report — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo
echo "## 1. Artifact integrity"
ok=1
check() { # path expected_md5 label
  m=$(md5sum "$1" 2>/dev/null | cut -d" " -f1)
  if [ "$m" = "$2" ]; then echo "- ✅ $3 ($1)"; else echo "- ❌ $3: md5 $m != $2 — RESTORE FROM GIT/BOX BEFORE LOCKING"; ok=0; fi
}
check deploy/caiest_cnn_bot.zip            064a49cbbf67674bfaacb8c3061e1a09 "WH-fixed code zip"
check deploy/incoming/sim6_v1_s600.pkl     9c1863e3b59923c5215b332bd483682c "V1 model (confirmed +515/100g)"
check deploy/caiest_cnn/data/cnn.pkl       7e45c41309502865b824f90b41a0a537 "distill100b floor"
echo
echo "## 2. New tournament/ladder logs since 2026-06-07"
new=$(find others -name "*.zip" -newermt 2026-06-08 2>/dev/null)
if [ -n "$new" ]; then
  echo '```'
  echo "$new"
  echo '```'
  echo "Run: unzip each, then python3 eval/analyze_tournament.py --root <dir> --name '[moyu]caiest'"
  echo "MUST verify: our WH count == 0 (wrong-HU fix in production). Compare both bots' rank/score."
else
  echo "- none found — collect the latest ladder logs for both bots before locking!"
fi
echo
echo "## 3. Lock checklist"
echo "- [ ] Pick model: sim6_v1_s600 if its ladder showing >= main bot, else distill100b"
echo "- [ ] Final bot has the WH-fixed zip (md5 064a49cb…) — REQUIRED"
echo "- [ ] Storage data/cnn.pkl = chosen model (verify md5 after upload)"
echo "- [ ] Smoke test: one manual Botzone match, confirm legal play + a HU"
echo "- [ ] FREEZE: change nothing after the final upload until 2026-06-14 23:55"
} > "$R"
echo "wrote $R (artifacts ok=$ok)"

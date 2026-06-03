#!/usr/bin/env bash
# guardian.sh — watches the league_v3 run; prunes disk, auto-heals monitor/dashboard,
# and exits (notifying the operator) on a hard anomaly or every ROUTINE_SECS for a check-in.
shopt -s nullglob
ROOT=/home/coder/IJCAI-mahjong
LDIR=$ROOT/train/league_v3
STATUS=/tmp/pbt_status.json
AWS='ubuntu@54.251.156.216 -i /home/coder/.ssh/aws_fleet_ed25519'
ROUTINE_SECS=${1:-1500}
END=$(( $(date +%s) + ROUTINE_SECS ))

st() { python3 -c "import json;print(json.load(open('$STATUS')).get('$1',''))" 2>/dev/null; }

while true; do
  RND=$(st gen); PHASE=$(st phase)
  [ -z "$RND" ] && RND=0

  # ── prune completed-round files (keep current round + global_best*/target*) ──
  if [ "$RND" -gt 1 ] 2>/dev/null; then
    for f in "$LDIR"/r*_*; do
      n=$(basename "$f" | sed -n 's/^r\([0-9]\+\)_.*/\1/p')
      [ -n "$n" ] && [ "$n" -lt "$RND" ] 2>/dev/null && rm -f "$f"
    done
  fi

  DISK=$(df /home/coder | tail -1 | awk '{print $5}' | tr -d '%')

  # ── hard anomalies → exit & notify ──
  if ! pgrep -f "league_v3.py" >/dev/null; then
    if [ "$PHASE" = "finished" ]; then echo "DONE: run finished (round $RND)"; exit 0; fi
    echo "ALARM: league_v3 process GONE — phase=$PHASE round=$RND disk=${DISK}%"; exit 2
  fi
  if [ "${DISK:-0}" -gt 92 ] 2>/dev/null; then
    echo "ALARM: disk ${DISK}% (round $RND) — need cleanup"; exit 3
  fi

  # ── auto-heal side services (safe to restart; do not touch the run) ──
  if ! pgrep -f "fleet_monitor.py" >/dev/null; then
    cd "$ROOT" && OPENBLAS_NUM_THREADS=1 FLEET_CENTER_SSH="$AWS" \
      FLEET_TOKEN=a7d8e3490c6bd1f4 FLEET_INGEST_PORT=5056 \
      nohup python3 train/fleet_monitor.py >/tmp/fleet_monitor.log 2>&1 &
    echo "HEALED: restarted fleet_monitor at round $RND"
  fi
  if ! curl -s -m4 localhost:8082 >/dev/null 2>&1; then
    cd "$ROOT" && OPENBLAS_NUM_THREADS=1 nohup python3 eval/pbt_dashboard.py >/tmp/pbt_dash.log 2>&1 &
    echo "HEALED: restarted dashboard at round $RND"
  fi

  # ── routine check-in ──
  if [ "$(date +%s)" -ge "$END" ]; then
    NPROM=$(python3 -c "import json;h=json.load(open('$STATUS')).get('history',[]);print(sum(1 for x in h if x.get('promoted')))" 2>/dev/null)
    echo "ROUTINE: round=$RND phase=$PHASE promotions=$NPROM disk=${DISK}% ldir=$(du -sh $LDIR 2>/dev/null|cut -f1)"; exit 0
  fi
  sleep 120
done

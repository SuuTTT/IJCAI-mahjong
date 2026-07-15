#!/bin/bash
# Backfill elite-db months until 2400+ band quota reached. Deletes raws after filtering.
cd /root/e2_chess
MONTHS="2025-09 2025-08 2025-07 2025-06 2025-05 2025-04 2025-03 2025-02 2025-01 2024-12 2024-11 2024-10 2024-09 2024-08 2024-07 2024-06"
KEPT=$1
for M in $MONTHS; do
  if [ "$KEPT" -ge 210000 ]; then echo "quota reached at $KEPT"; break; fi
  Z=data/elite/lichess_elite_$M.zip
  curl -s -o $Z https://database.nikonoel.fr/lichess_elite_$M.zip || { echo "download failed $M"; continue; }
  unzip -o -q $Z -d data/elite/ || { echo "unzip failed $M"; rm -f $Z; continue; }
  rm -f $Z
  OUT=$(E2_KEPT0=$KEPT python3 src/filter_elite.py < data/elite/lichess_elite_$M.pgn)
  rm -f data/elite/lichess_elite_$M.pgn
  KEPT=$(echo $OUT | python3 -c "import sys,json;print(json.load(sys.stdin)[\"kept_total\"])")
  echo "$M -> kept_total=$KEPT"
  echo "{\"last_month\": \"$M\", \"kept_total\": $KEPT}" > logs/elite_backfill_status.json
done
echo "BACKFILL DONE kept_total=$KEPT"

#!/bin/bash
# 8h autonomous orchestrator. Goal: (1) decisively confirm whether V-search beats the teacher,
# (2) keep growing the data->distill candidate, (3) keep GPUs busy. ssh1 = eval ONLY, sequential
# (no stuck-storms). GPU boxes (ssh8=30497, 3070=22734) train independently; outputs pulled to ssh1
# between gauntlets (never scp a file a job is reading). Everything logged to RESULTS.
set +e
K="$HOME/.ssh/vastai_id_ed25519"; O="-i $K -o StrictHostKeyChecking=no -o ConnectTimeout=25"
S1(){ ssh $O -p 30645 root@ssh1.vast.ai "$1" 2>/dev/null; }
S8(){ ssh $O -p 30497 root@ssh8.vast.ai "$1" 2>/dev/null; }
S7(){ ssh $O -p 22734 root@ssh5.vast.ai "$1" 2>/dev/null; }
R=/tmp/orch8h_results.txt; LADBASE=/root/mahjong/ckpt/lad_chunjiandu_v2_distill.pkl
RANK='/root/mahjong/livedata/ladder_top30_score1216/bulk_ranked_matches/ranking_snapshot.json'
say(){ echo "[orch $(date -u +%H:%M)] $1" | tee -a "$R"; }
net(){ S1 "grep -oE 'TOTAL net=[-+0-9]+' /root/mahjong/$1 2>/dev/null | grep -oE '[-+0-9]+\$'"; }
# gauntlet a candidate on ssh1 (sequential). args: name ckpt games [cand_env]
gaunt(){ local nm=$1 ck=$2 g=$3 ce=$4
  S1 "cd /root/mahjong && CAND_ENV=\"$ce\" GN=$g BENCH_TIMEOUT=40 python3 run_gauntlet.py $nm $ck > /root/mahjong/g8_$nm.log 2>&1"
  say "gauntlet $nm ($g g/opp): net=$(net g8_$nm.log)"; }
pull(){ scp $O -P $1 root@$2:/root/mahjong/ckpt/$3 /tmp/$3 2>/dev/null; scp $O -P 30645 /tmp/$3 root@ssh1.vast.ai:/root/mahjong/ckpt/$3 2>/dev/null; }

say "=== 8h orchestrator start. baseline: plain lad_chunjiandu +4119 (24g). ==="

# STEP 1 — wait out the running V-search config sweep, pick best (VK,VDELTA)
say "STEP1: waiting for V-search sweep (gd_vs_*)…"
for i in $(seq 1 40); do [ "$(S1 'grep -hc "TOTAL net=" /root/mahjong/gd_vs_k6d3.log /root/mahjong/gd_vs_k3d1.log /root/mahjong/gd_vs_k5d25.log 2>/dev/null')" -ge 3 ] && break; sleep 90; done
B6=$(net gd_vs_k6d3.log); B3=$(net gd_vs_k3d1.log); B5=$(net gd_vs_k5d25.log)
say "sweep: k6d3=$B6 k3d1=$B3 k5d25=$B5 (vs baseline vsearch k4d2=+4176)"
BCFG="CAIEST_VK=4 CAIEST_VDELTA=2.0"; BEST=${B6:-0}
[ "${B3:-0}" -gt "$BEST" ] 2>/dev/null && { BEST=$B3; BCFG="CAIEST_VK=3 CAIEST_VDELTA=1.0"; }
[ "${B5:-0}" -gt "$BEST" ] 2>/dev/null && { BEST=$B5; BCFG="CAIEST_VK=5 CAIEST_VDELTA=2.5"; }
[ "${B6:-0}" -ge "$BEST" ] 2>/dev/null && BCFG="CAIEST_VK=6 CAIEST_VDELTA=3.0"
[ 4176 -ge "${BEST:-0}" ] 2>/dev/null && { BCFG="CAIEST_VK=4 CAIEST_VDELTA=2.0"; }
say "best search config: $BCFG"

# STEP 2 — gauntlet the better V net (vbig) with the best config; choose BESTV
say "STEP2: waiting for vbig…"; for i in $(seq 1 20); do S8 'test -f /root/mahjong/ckpt/vbig.pkl && echo y' | grep -q y && break; sleep 60; done
BV=/root/mahjong/ckpt/vfull.pkl
if S8 'test -f /root/mahjong/ckpt/vbig.pkl && echo y' | grep -q y; then
  pull 30497 ssh8.vast.ai vbig.pkl
  gaunt vbig_search $LADBASE 24 "CAIEST_VNET=/root/mahjong/ckpt/vbig.pkl $BCFG"
  [ "$(net g8_vbig_search.log)" -gt 4176 ] 2>/dev/null && BV=/root/mahjong/ckpt/vbig.pkl
fi
say "best V net: $BV"

# STEP 3 — DECISIVE confirmation @ 48 g/opp, same walls: plain teacher vs best V-search
say "STEP3: decisive 48g/opp confirmation…"
gaunt plain48 $LADBASE 48 ""
gaunt vsearch48 $LADBASE 48 "CAIEST_VNET=$BV $BCFG"
say "DECISIVE: plain=$(net g8_plain48.log)  vsearch=$(net g8_vsearch48.log)  (margin = vsearch - plain)"

# STEP 4 — data->distill loop: grow chunlive, re-distill from distill100b (the +4119 recipe), gauntlet
cyc=0
for big in $(seq 1 5); do
  cyc=$((cyc+1)); say "STEP4 cyc$cyc: sync data + re-extract chunlive"
  cd "$HOME/IJCAI-mahjong/others" 2>/dev/null
  find ladder_top30_score1216/future_hourly \( -name '*full_log*.json' -o -name '*metadata*.json' \) 2>/dev/null > /tmp/hf8.txt
  tar czf /tmp/hf8.tgz -T /tmp/hf8.txt 2>/dev/null; scp $O -P 30645 /tmp/hf8.tgz root@ssh1.vast.ai:/root/mahjong/hf8.tgz 2>/dev/null
  S1 'cd /root/mahjong && tar xzf hf8.tgz -C livedata 2>/dev/null'
  N=$(S1 "cd /root/mahjong/caiest_repro && PYTHONPATH=/root/mahjong python3 extract_top30.py --root /root/mahjong/livedata --ranking '$RANK' --player chunjiandu --since 2026-05-01 --out data/chunlive8.npz 2>/dev/null | grep -oE '[0-9]+ decisions' | grep -oE '^[0-9]+'")
  say "cyc$cyc: chunlive=$N decisions"
  # distill on the 3070 (free GPU), beta 1.0, 800 steps (recipe law), from the distill100b floor
  scp $O -P 30645 root@ssh1.vast.ai:/root/mahjong/caiest_repro/data/chunlive8.npz /tmp/cl8.npz 2>/dev/null
  scp $O -P 22734 /tmp/cl8.npz root@ssh5.vast.ai:/root/mahjong/caiest_repro/data/chunlive8.npz 2>/dev/null
  S7 "cd /root/mahjong/caiest_repro && nohup python3 distill_kl.py --base /root/mahjong/ckpt/distill100b_fused.pkl --champ data/chunlive8.npz --beta 1.0 --seed $cyc --steps 800 --out /root/mahjong/ckpt/redist8_c$cyc.pkl > /root/mahjong/redist8.log 2>&1 </dev/null & echo k"
  for i in $(seq 1 20); do S7 'grep -c DONE /root/mahjong/redist8.log' | grep -q '^[1-9]' && break; sleep 60; done
  pull 22734 ssh5.vast.ai redist8_c$cyc.pkl
  gaunt redist8_c$cyc /root/mahjong/ckpt/redist8_c$cyc.pkl 24 ""
  # also gauntlet redist + V-search (data gain × search gain)
  gaunt redist8_c${cyc}_vs /root/mahjong/ckpt/redist8_c$cyc.pkl 24 "CAIEST_VNET=$BV $BCFG"
  say "cyc$cyc done. waiting ~80min for more games to accumulate…"
  sleep 4800
done

say "=== ORCHESTRATOR DONE. Full leaderboard: ==="
S1 'grep -h "TOTAL net=" /root/mahjong/g8_*.log /root/mahjong/gd_final_*.log 2>/dev/null | sort -t= -k2 -rn | head -20' | tee -a "$R"
say "ORCH8H_COMPLETE"

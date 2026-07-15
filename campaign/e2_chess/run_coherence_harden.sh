#!/bin/bash
# Track 1: harden mixedband coherence cell (Paper A pre-registered claim).
# Adds teacher trios C (s60-62), D (s63-65) + student seeds s43-45
# (protocol-identical: students distill from the ORIGINAL 6 teachers).
cd /root/e2_chess
T=ckpt/mixedband
ORIG6=$T/teacher_s30.pt,$T/teacher_s31.pt,$T/teacher_s32.pt,$T/teacher_s33.pt,$T/teacher_s34.pt,$T/teacher_s35.pt

run_gpu () {  # gpu, then sequential jobs
  local g=$1; shift
  for cmd in "$@"; do
    CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" >> logs/coherence_harden_g$g.log 2>&1
  done
}
run_gpu 0 \
  "[ -f $T/teacher_s60.pt ] || python3 src/chess_bc_train.py --enc-dir enc/full_mixedband --seed 60 --out $T/teacher_s60.pt" \
  "[ -f $T/teacher_s61.pt ] || python3 src/chess_bc_train.py --enc-dir enc/full_mixedband --seed 61 --out $T/teacher_s61.pt" \
  "[ -f $T/teacher_s62.pt ] || python3 src/chess_bc_train.py --enc-dir enc/full_mixedband --seed 62 --out $T/teacher_s62.pt" \
  "[ -f $T/student_s43.pt ] || python3 src/chess_kd_train.py --enc-dir enc/full_mixedband --teachers $ORIG6 --seed 43 --alpha 0.7 --out $T/student_s43.pt" &
P0=$!
run_gpu 1 \
  "[ -f $T/teacher_s63.pt ] || python3 src/chess_bc_train.py --enc-dir enc/full_mixedband --seed 63 --out $T/teacher_s63.pt" \
  "[ -f $T/teacher_s64.pt ] || python3 src/chess_bc_train.py --enc-dir enc/full_mixedband --seed 64 --out $T/teacher_s64.pt" \
  "[ -f $T/teacher_s65.pt ] || python3 src/chess_bc_train.py --enc-dir enc/full_mixedband --seed 65 --out $T/teacher_s65.pt" \
  "[ -f $T/student_s44.pt ] || python3 src/chess_kd_train.py --enc-dir enc/full_mixedband --teachers $ORIG6 --seed 44 --alpha 0.7 --out $T/student_s44.pt" \
  "[ -f $T/student_s45.pt ] || python3 src/chess_kd_train.py --enc-dir enc/full_mixedband --teachers $ORIG6 --seed 45 --alpha 0.7 --out $T/student_s45.pt" &
P1=$!
wait $P0 $P1
# evals (CPU-heavy): new trios + new students, same ladder protocol as E2_COHERENCE
python3 src/chess_eval_e2.py --band mixedband --name mixedband_trioC --ckpts $T/teacher_s60.pt,$T/teacher_s61.pt,$T/teacher_s62.pt --out results/e2_eval_mixedband_trioC.json
python3 src/chess_eval_e2.py --band mixedband --name mixedband_trioD --ckpts $T/teacher_s63.pt,$T/teacher_s64.pt,$T/teacher_s65.pt --out results/e2_eval_mixedband_trioD.json
python3 src/chess_eval_e2.py --band mixedband --name mixedband_student2 --ckpts $T/student_s43.pt,$T/student_s44.pt,$T/student_s45.pt --out results/e2_eval_mixedband_student2.json
touch results/COHERENCE_HARDEN_DONE
echo COHERENCE_HARDEN_DONE

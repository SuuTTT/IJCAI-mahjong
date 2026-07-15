#!/bin/bash
cd /root/e2_chess
PY=/root/e2_chess/venv/bin/python
T=ckpt/mixedband
TEACHERS=$T/teacher_s30.pt,$T/teacher_s31.pt,$T/teacher_s32.pt,$T/teacher_s33.pt,$T/teacher_s34.pt,$T/teacher_s35.pt,$T/teacher_s60.pt,$T/teacher_s61.pt,$T/teacher_s62.pt,$T/teacher_s63.pt,$T/teacher_s64.pt,$T/teacher_s65.pt
STUDENTS=$T/student_s40.pt,$T/student_s41.pt,$T/student_s42.pt,$T/student_s43.pt,$T/student_s44.pt,$T/student_s45.pt
$PY src/chess_eval_e2.py --band mixedband --name mb_teacher_ens12 --ckpts $TEACHERS --games 600 --out results/e2_mb_teacher_ens12.json >> logs/coherence_pooled.log 2>&1
$PY src/chess_eval_e2.py --band mixedband --name mb_student_ens6 --ckpts $STUDENTS --games 600 --out results/e2_mb_student_ens6.json >> logs/coherence_pooled.log 2>&1
touch results/COHERENCE_POOLED_DONE; echo COHERENCE_POOLED_DONE

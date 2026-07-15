#!/bin/bash
cd /root/e2_chess
PY=/root/e2_chess/venv/bin/python
T=ckpt/mixedband
$PY src/chess_eval_e2.py --band mixedband --name mixedband_trioC --ckpts $T/teacher_s60.pt,$T/teacher_s61.pt,$T/teacher_s62.pt --out results/e2_eval_mixedband_trioC.json >> logs/coherence_eval.log 2>&1
$PY src/chess_eval_e2.py --band mixedband --name mixedband_trioD --ckpts $T/teacher_s63.pt,$T/teacher_s64.pt,$T/teacher_s65.pt --out results/e2_eval_mixedband_trioD.json >> logs/coherence_eval.log 2>&1
$PY src/chess_eval_e2.py --band mixedband --name mixedband_student2 --ckpts $T/student_s43.pt,$T/student_s44.pt,$T/student_s45.pt --out results/e2_eval_mixedband_student2.json >> logs/coherence_eval.log 2>&1
touch results/COHERENCE_EVALS_DONE; echo COHERENCE_EVALS_DONE

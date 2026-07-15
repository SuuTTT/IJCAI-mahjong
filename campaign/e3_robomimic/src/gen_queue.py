#!/usr/bin/env python
"""Emit E3 grid job lines for /root/gpu_queue.txt (keeper format).

Skips lift_ph teacher_s0 train+eval (done by the smoke test).
Order: teacher trains -> student trains (REQUIRE 6 teachers) -> evals -> summary.
"""
D = "/root/e3_robomimic"
TASKS = ["lift", "can", "square"]
SETS = ["ph", "mh"]
PRE = f"cd {D} && OMP_NUM_THREADS=1 nice -n 10 venv/bin/python"
SMOKE_DONE = {("lift", "ph", 0)}

lines = []


def data(t, s):
    return f"datasets/{t}/{s}/low_dim_v141.hdf5"


# 1) teacher trains
for t in TASKS:
    for s in SETS:
        for i in range(6):
            if (t, s, i) in SMOKE_DONE:
                continue
            lines.append(
                f"{PRE} src/bc_train.py --data {data(t,s)} --seed {i} "
                f"--out ckpt/{t}_{s}/teacher_s{i}.pt >> logs/train_{t}_{s}_t{i}.log 2>&1")

# 2) student trains (need all 6 teachers of the cell)
for t in TASKS:
    for s in SETS:
        tp = ",".join(f"ckpt/{t}_{s}/teacher_s{i}.pt" for i in range(6))
        for i in (10, 11, 12):
            lines.append(
                f"REQUIRE {D}/ckpt/{t}_{s}/teacher_s?.pt 6 {'::'} "
                f"{PRE} src/bc_train.py --data {data(t,s)} --seed {i} "
                f"--teacher_ckpts {tp} --alpha 0.7 "
                f"--out ckpt/{t}_{s}/student_s{i}.pt >> logs/train_{t}_{s}_st{i}.log 2>&1")

# 3) evals (100 fixed-seed rollouts each)
def ev(t, s, name, ckpts, req_glob, req_n):
    return (f"REQUIRE {req_glob} {req_n} {'::'} "
            f"{PRE} src/bc_eval.py --data {data(t,s)} --ckpts {ckpts} "
            f"--name {name} --task {t} --demoset {s} --rollouts 100 "
            f"--out results/{t}_{s}_{name}.json >> logs/eval_{t}_{s}_{name}.log 2>&1")


for t in TASKS:
    for s in SETS:
        C = f"ckpt/{t}_{s}"
        for i in range(6):
            if (t, s, i) in SMOKE_DONE:
                continue
            lines.append(ev(t, s, f"teacher_s{i}", f"{C}/teacher_s{i}.pt",
                            f"{D}/{C}/teacher_s{i}.pt", 1))
        lines.append(ev(t, s, "trioA", ",".join(f"{C}/teacher_s{i}.pt" for i in (0, 1, 2)),
                        f"{D}/{C}/teacher_s[012].pt", 3))
        lines.append(ev(t, s, "trioB", ",".join(f"{C}/teacher_s{i}.pt" for i in (3, 4, 5)),
                        f"{D}/{C}/teacher_s[345].pt", 3))
        for i in (10, 11, 12):
            lines.append(ev(t, s, f"student_s{i}", f"{C}/student_s{i}.pt",
                            f"{D}/{C}/student_s{i}.pt", 1))
        lines.append(ev(t, s, "student_ens", ",".join(f"{C}/student_s{i}.pt" for i in (10, 11, 12)),
                        f"{D}/{C}/student_s1?.pt", 3))

# 4) summary once all 72 eval jsons exist
lines.append(f"REQUIRE {D}/results/*_*.json 72 {'::'} "
             f"{PRE} src/e3_grid_summary.py --results results "
             f"--out results/E3_ROBOMIMIC.json >> logs/e3_summary.log 2>&1")

for ln in lines:
    assert " :: " not in ln.split(" :: ", 1)[-1] or ln.startswith("REQUIRE"), ln
    print(ln)

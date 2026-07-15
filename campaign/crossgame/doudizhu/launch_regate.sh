#!/bin/bash
# C2 fix launcher: 5 eps tracks, each re-gates multiple trios + distill ensembles on ONE
# fixed 3000-seed per-game-seeded set (seed0=10000, same as original sweep_gate protocol).
# GPU picker: only GPUs with >8GB free (box shared with se_mahjong M-grid — never kill).
cd /root/crossgame/doudizhu || exit 1
mkdir -p results logs
FREE=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>8000{print $1}'))
if [ "${#FREE[@]}" -lt 5 ]; then echo "only ${#FREE[@]} GPUs with >8GB free; need 5"; exit 1; fi
echo "using GPUs: ${FREE[*]:0:5}"

run(){ # $1=gpu $2=tag $3=logtag ; rest = --ens args
  local g=$1 tag=$2 lt=$3; shift 3
  setsid nohup env CUDA_VISIBLE_DEVICES=$g python3 regate_trios.py --tag "$tag" "$@" \
    --nseeds 3000 --seed0 10000 \
    --out results/regate_${lt}_v2.json --npz results/regate_${lt}_v2.npz \
    > logs/regate_${lt}_v2.log 2>&1 &
  echo "launched eps=$tag on GPU$g (pid $!)"
}

D=ckpt/sweep
run ${FREE[0]} 0 e00 \
  --ens seed_trio1=ckpt/teachers/dou_teacher_s0.pkl,ckpt/teachers/dou_teacher_s1.pkl,ckpt/teachers/dou_teacher_s2.pkl \
  --ens seed_trio2=ckpt/teachers/dou_teacher_s3.pkl,ckpt/teachers/dou_teacher_s4.pkl,ckpt/teachers/dou_teacher_s5.pkl \
  --ens seed_trio3_overlap=ckpt/teachers/dou_teacher_s6.pkl,ckpt/teachers/dou_teacher_s7.pkl,ckpt/teachers/dou_teacher_s0.pkl \
  --ens distillA=ckpt/students_det/dou_student_s100.pkl,ckpt/students_det/dou_student_s101.pkl,ckpt/students_det/dou_student_s102.pkl

run ${FREE[1]} 0.1 e01 \
  --ens seed_trio1=$D/e01/tea_s7101.pkl,$D/e01/tea_s7102.pkl,$D/e01/tea_s7103.pkl \
  --ens seed_trio2=$D/e01/tea_s7104.pkl,$D/e01/tea_s7105.pkl,$D/e01/tea_s7106.pkl \
  --ens distillA=$D/e01/stu_s7111.pkl,$D/e01/stu_s7112.pkl,$D/e01/stu_s7113.pkl \
  --ens distillB=$D/e01/stu_s7114.pkl,$D/e01/stu_s7115.pkl,$D/e01/stu_s7116.pkl

run ${FREE[2]} 0.2 e02 \
  --ens seed_trio1=$D/e02/tea_s7201.pkl,$D/e02/tea_s7202.pkl,$D/e02/tea_s7203.pkl \
  --ens seed_trio2=$D/e02/tea_s7204.pkl,$D/e02/tea_s7205.pkl,$D/e02/tea_s7206.pkl \
  --ens distillA=$D/e02/stu_s7211.pkl,$D/e02/stu_s7212.pkl,$D/e02/stu_s7213.pkl \
  --ens distillB=$D/e02/stu_s7214.pkl,$D/e02/stu_s7215.pkl,$D/e02/stu_s7216.pkl

run ${FREE[3]} 0.3 e03 \
  --ens seed_trio1=ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl \
  --ens seed_trio2=ckpt/teachers_noisy/dou_nteacher_s203.pkl,ckpt/teachers_noisy/dou_nteacher_s204.pkl,ckpt/teachers_noisy/dou_nteacher_s205.pkl \
  --ens seed_trio3_overlap=ckpt/teachers_noisy/dou_nteacher_s206.pkl,ckpt/teachers_noisy/dou_nteacher_s207.pkl,ckpt/teachers_noisy/dou_nteacher_s200.pkl \
  --ens distillA=ckpt/students_noisy/dou_nstudent_s210.pkl,ckpt/students_noisy/dou_nstudent_s211.pkl,ckpt/students_noisy/dou_nstudent_s212.pkl

run ${FREE[4]} 0.5 e05 \
  --ens seed_trio1=$D/e05/tea_s7501.pkl,$D/e05/tea_s7502.pkl,$D/e05/tea_s7503.pkl \
  --ens seed_trio2=$D/e05/tea_s7504.pkl,$D/e05/tea_s7505.pkl,$D/e05/tea_s7506.pkl \
  --ens distillA=$D/e05/stu_s7511.pkl,$D/e05/stu_s7512.pkl,$D/e05/stu_s7513.pkl \
  --ens distillB=$D/e05/stu_s7514.pkl,$D/e05/stu_s7515.pkl,$D/e05/stu_s7516.pkl

echo "all 5 tracks launched"

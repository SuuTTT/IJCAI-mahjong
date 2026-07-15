#!/bin/bash
# Safely stop the sequential master_chain.sh WITHOUT touching training procs.
# pkill skips its own pid; this parent's cmdline is stop_chain.sh (no 'master_chain' substring),
# and training procs (dou_bc_train/dou_kd_train) don't contain 'master_chain.sh'.
pkill -f master_chain.sh
sleep 1
echo "remaining master_chain procs: $(pgrep -cf master_chain.sh)"
echo "det teachers still alive: $(pgrep -cf 'out ckpt/teachers/dou_teacher_s')"

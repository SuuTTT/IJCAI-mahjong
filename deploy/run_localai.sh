#!/bin/bash
# Deploy ml_bot to Botzone via the LocalAI adapter.
#
# The bot runs HERE (Ubuntu 24.04, GPU, working MahjongGB + numpy). The adapter
# polls your Botzone /localai endpoint over HTTP, forwards each game request to
# ml_bot, and sends the action back. NO Botzone-environment compatibility issues.
#
# Setup (one time):
#   1. On Botzone, create a bot for Chinese-Standard-Mahjong (any language; it is
#      only a placeholder that routes to LocalAI). Enable LocalAI for it.
#   2. Copy the bot's LocalAI URL:  https://www.botzone.org.cn/api/<UID>/<SECRET>/localai
#   3. Run:  LOCALAI_URL="<that url>" bash deploy/run_localai.sh
#
# Then enter Simulation-7 (or a match) on Botzone with that bot; games route here.

set -e
cd "$(dirname "$0")/.."

: "${LOCALAI_URL:?Set LOCALAI_URL to your Botzone localai endpoint}"
MODEL="${MODEL:-train/checkpoints/bc_v3_ft_weights.npz}"

echo "Bot:   ml_bot.py  (model: $MODEL)"
echo "URL:   $LOCALAI_URL"
echo "Single-threaded BLAS (matches Botzone single-core)."
echo "Press Ctrl-C to stop."
echo

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MODEL="$MODEL" \
python3 deploy/local_ai.py \
    --localai-url "$LOCALAI_URL" \
    --bot-cmd python3 "$(pwd)/bot/ml_bot.py" \
    --bot-cwd "$(pwd)" \
    --retry-seconds 5

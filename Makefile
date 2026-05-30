JUDGE   := /workspace/Chinese-Standard-Mahjong/judge/judge
BOT_V02 := bot/bot_submit_test
BOT_SMP := eval/sample_bot
BOT_KR  := bot/keeprunning_bot.py
BOT_ML  := bot/ml_bot.py
DATA    := data/processed/official_winner.npz

.PHONY: all build test smoke eval-quick eval-full test-kr localai parse-data train eval-ml clean help

all: build test

build:
	$(MAKE) -C bot all

test: build
	python3 -m pytest tests/test_bot.py -v
	python3 tests/stress_test.py

smoke: build
	python3 -c "\
import sys; sys.path.insert(0,'eval'); \
from run_match import run_match; \
r = run_match(['$(BOT_V01)','$(BOT_SMP)','$(BOT_SMP)','$(BOT_SMP)'],timeout=5,verbose=True); \
print('scores:',r['scores'],'winner:',r['winner'])" \
	2>&1 | grep -v "^  turn"

eval-quick: build
	python3 eval/duplicate_eval.py \
	    "$(BOT_V01)" "$(BOT_SMP)" "$(BOT_SMP)" "$(BOT_SMP)" \
	    --walls 1 --seed 42 --timeout 5 --verbose

eval-full: build
	python3 eval/duplicate_eval.py \
	    "$(BOT_V02)" "$(BOT_SMP)" "$(BOT_SMP)" "$(BOT_SMP)" \
	    --walls 4 --seed 42 --timeout 5 --verbose

parse-data:
	python3 data/parse_official.py \
	    --inp data/raw/data.zip \
	    --out $(DATA) \
	    --winner-only

train: $(DATA)
	python3 train/train_bc.py \
	    --data $(DATA) \
	    --out train/checkpoints/bc_v1.pt \
	    --epochs 50 --batch 2048 --hidden 512 --blocks 6

eval-ml: build
	MODEL=train/checkpoints/bc_v1_weights.npz \
	python3 eval/duplicate_eval.py \
	    "python3 $(BOT_ML)" "$(BOT_V02)" "$(BOT_SMP)" "$(BOT_SMP)" \
	    --walls 2 --seed 42 --timeout 10 --verbose

test-kr:
	python3 -c "\
import subprocess,time; \
p=subprocess.Popen(['python3','$(BOT_KR)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1); \
SEN='>>>BOTZONE_REQUEST_KEEP_RUNNING<<<'; \
ask=lambda l: [p.stdin.write(l+chr(10)),p.stdin.flush()] and __import__('itertools').takewhile(lambda x:x!=SEN,[p.stdout.readline().rstrip() for _ in range(99)]); \
p.stdin.write('1\n'); p.stdin.flush(); time.sleep(0.1); \
print('init:',list(ask('0 2 0'))); \
print('deal:',list(ask('1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 B4'))); \
print('draw(win):',list(ask('2 B4'))); \
p.terminate()"

localai:
	@echo "Set LOCALAI_URL to your Botzone localai endpoint first."
	@echo "e.g.  LOCALAI_URL=https://www.botzone.org.cn/api/UID/SECRET/localai make localai"
	python3 /tmp/Mahjong-LLM/local_ai/local_ai.py \
	    --localai-url "$$LOCALAI_URL" \
	    --bot-cmd "python3 $(BOT_KR)" \
	    --bot-cwd "$(CURDIR)/bot"

clean:
	$(MAKE) -C bot clean

help:
	@echo "make test          unit tests + 200-game stress test"
	@echo "make smoke         one live game v0.2 vs 3x sample"
	@echo "make eval-quick    1 wall x 24 games duplicate eval"
	@echo "make eval-full     4 walls x 96 games duplicate eval"
	@echo "make test-kr       keeprunning bot sanity"
	@echo "make parse-data    parse official 98k-game dataset -> .npz"
	@echo "make train         train SL model (needs data/processed/official_winner.npz)"
	@echo "make eval-ml       eval ML bot vs v0.2 vs 2x sample"
	@echo "LOCALAI_URL=... make localai   live Botzone testing"

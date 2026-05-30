ALGO = /workspace/Chinese-Standard-Mahjong/fan-calculator-usage

.PHONY: all local submit zip test clean

all: local submit zip

local: bot_local

# Local dev binary with jsoncpp from system
bot_local: bot.cpp
	g++ -O2 -std=c++14 \
	    -I$(ALGO) -DLOCAL_BUILD \
	    bot.cpp -ljsoncpp -o bot_local

# Single-file Botzone C++ submission
submit: bot_submit.cpp
bot_submit.cpp: bot.cpp make_submit.sh
	bash make_submit.sh

# Verify the submission binary
bot_submit_test: bot_submit.cpp
	g++ -O2 -std=c++14 \
	    bot_submit.cpp -ljsoncpp -o bot_submit_test

# Python bot zip for Botzone (backup)
zip: mahjong_bot.zip
mahjong_bot.zip: __main__.py mahjong_bot.py
	zip -j mahjong_bot.zip __main__.py mahjong_bot.py
	@echo "Python zip ready: mahjong_bot.zip"
	@echo "NOTE: PyMahjongGB must be available on Botzone or use fallback mode."

test: bot_submit_test
	cd .. && python3 -m pytest tests/test_bot.py -v
	cd .. && python3 tests/stress_test.py

clean:
	rm -f bot_local bot_submit.cpp bot_submit_test mahjong_bot.zip

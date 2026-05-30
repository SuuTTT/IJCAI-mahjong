#!/bin/bash
# Creates bot_submit.cpp: single-file amalgam for Botzone C++ upload.
set -e

ALGO=/workspace/Chinese-Standard-Mahjong/fan-calculator-usage/ChineseOfficialMahjongHelper/Classes/mahjong-algorithm
OUT=bot_submit.cpp

python3 - "$ALGO" "$OUT" <<'PYEOF'
import sys, re

ALGO = sys.argv[1]
OUT  = sys.argv[2]

def read(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        src = f.read()
    # Strip BOM
    src = src.lstrip('﻿')
    return src

def remove_local_includes(src):
    """Remove all #include "..." lines (local headers that will be inlined)."""
    return re.sub(r'^#include\s+"[^"]*"[^\n]*\n', '', src, flags=re.MULTILINE)

# --- Read algorithm sources (keep their guards, just remove local includes) ---
tile_h      = remove_local_includes(read(f"{ALGO}/tile.h"))
standard_h  = remove_local_includes(read(f"{ALGO}/standard_tiles.h"))
shanten_h   = remove_local_includes(read(f"{ALGO}/shanten.h"))
fan_h       = remove_local_includes(read(f"{ALGO}/fan_calculator.h"))
shanten_cpp = remove_local_includes(read(f"{ALGO}/shanten.cpp"))
fan_cpp     = remove_local_includes(read(f"{ALGO}/fan_calculator.cpp"))

# --- Read bot.cpp ---
bot = read("bot.cpp")
# Remove LOCAL_BUILD block
bot = re.sub(r'#ifdef LOCAL_BUILD\b.*?#endif[^\n]*\n', '', bot, flags=re.DOTALL)
# Remove Botzone json ifdef (will be added at top)
bot = re.sub(r'#ifdef _BOTZONE_ONLINE.*?#endif[^\n]*\n', '', bot, flags=re.DOTALL)

# Bot's own system includes that conflict with algo headers
SKIP_INCLUDES = {'<iostream>','<string>','<sstream>','<vector>',
                 '<algorithm>','<unordered_map>','<cstring>',
                 '<climits>','<cassert>'}
def keep_bot_line(line):
    stripped = line.strip()
    for h in SKIP_INCLUDES:
        if stripped == f'#include {h}':
            return False
    return True

bot_lines = [l for l in bot.split('\n') if keep_bot_line(l)]
bot_clean = '\n'.join(bot_lines)

# --- Build amalgamated file ---
out_parts = []
out_parts.append("""// ==========================================================
// IJCAI Mahjong AI Bot - v0.1 (Botzone amalgamated single file)
// ==========================================================

#ifdef _BOTZONE_ONLINE
#include "jsoncpp/json.h"
#else
#include <jsoncpp/json/json.h>
#endif

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <climits>
#include <cassert>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <limits>
#include <iterator>
""")

out_parts.append("// ===== tile.h =====\n" + tile_h)
out_parts.append("// ===== standard_tiles.h =====\n" + standard_h)
out_parts.append("// ===== shanten.h =====\n" + shanten_h)
out_parts.append("// ===== fan_calculator.h =====\n" + fan_h)
out_parts.append("// ===== shanten.cpp =====\n" + shanten_cpp)
out_parts.append("// ===== fan_calculator.cpp =====\n" + fan_cpp)
out_parts.append("// ===== bot.cpp =====\n" + bot_clean)

result = '\n'.join(out_parts)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(result)

line_count = result.count('\n') + 1
print(f"Done: {OUT} ({line_count} lines)")
PYEOF

echo ""
echo "Verifying compile..."
g++ -O2 -std=c++14 \
    "$OUT" -ljsoncpp -o bot_submit_test 2>&1 \
    && echo "=== Compile OK ===" \
    || { echo "=== Compile FAILED ==="; exit 1; }

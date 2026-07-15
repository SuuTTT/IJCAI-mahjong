#!/usr/bin/env python3
"""E2: filter Lichess Elite Database PGNs (stdin) for the 2400+ band.
Same gates as filter_stream.py EXCEPT no %clk requirement (elite db strips
clock comments). Keeps: Rated Rapid/Classical, both Elos >= 2400,
Termination Normal, reached move 6. Appends to band_2400plus_elite.pgn."""
import sys, os, json

OUT = "/root/e2_chess/data/band_2400plus_elite.pgn"
QUOTA = int(os.environ.get("E2_QUOTA", "210000"))

out = open(OUT, "ab", buffering=1 << 20)
kept = int(os.environ.get("E2_KEPT0", "0"))
seen = 0


def process(lines, hdr):
    global kept
    ev = hdr.get(b"Event", b"")
    if not (b"Rated Classical" in ev or b"Rated Rapid" in ev):
        return
    if hdr.get(b"Termination") != b"Normal":
        return
    try:
        we, be = int(hdr.get(b"WhiteElo", b"?")), int(hdr.get(b"BlackElo", b"?"))
    except ValueError:
        return
    if we < 2400 or be < 2400:
        return
    body = b"".join(lines)
    if b" 6." not in body:
        return
    out.write(body); out.write(b"\n")
    kept += 1


lines, hdr = [], {}
for line in sys.stdin.buffer:
    if line.startswith(b"[Event ") and lines:
        process(lines, hdr)
        lines, hdr = [], {}
        if kept >= QUOTA:
            break
    lines.append(line)
    if line.startswith(b"["):
        i = line.find(b' "')
        if i > 0:
            hdr[line[1:i]] = line[i + 2:line.rfind(b'"')]
    if line.startswith(b"[Event "):
        seen += 1
if lines and kept < QUOTA:
    process(lines, hdr)
out.close()
print(json.dumps({"seen": seen, "kept_total": kept}))

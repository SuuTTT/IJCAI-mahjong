#!/usr/bin/env python3
"""E2 chess BC: stream-filter a lichess monthly PGN dump (fed on stdin, decompressed).

Keeps games where:
  - Event is a Rated Classical or Rated Rapid game/tournament
  - both WhiteElo and BlackElo are inside one target band
  - Termination == "Normal" (excludes time forfeit / abandoned / cheat flags)
  - movetext contains %clk (clock annotations present)
  - game reached at least move 6 (>= ~11 plies), drops trivial games

Writes one PGN file per band, plus a status JSON every ~30s.
Exits 0 as soon as every band hit its quota (upstream curl dies via SIGPIPE).
"""
import sys, os, json, time

OUT_DIR = "/root/e2_chess/data"
STATUS = "/root/e2_chess/logs/filter_status.json"
QUOTA = int(os.environ.get("E2_QUOTA", "210000"))

BANDS = [
    (b"0800-1200", 800, 1200),
    (b"1600-2000", 1600, 2000),
    (b"2400plus", 2400, 9999),
]

outs = {}
counts = {}
for name, lo, hi in BANDS:
    n = name.decode()
    outs[name] = open(os.path.join(OUT_DIR, f"band_{n}.pgn"), "ab", buffering=1 << 20)
    counts[name] = 0

games_seen = 0
bytes_in = 0
t0 = time.time()
last_status = 0.0


def write_status(done=False):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "bytes_in_decompressed": bytes_in,
            "games_seen": games_seen,
            "counts": {k.decode(): v for k, v in counts.items()},
            "quota": QUOTA,
            "elapsed_s": round(time.time() - t0, 1),
            "done": done,
        }, f)
    os.replace(tmp, STATUS)


def band_of(we, be):
    for name, lo, hi in BANDS:
        if lo <= we <= hi and lo <= be <= hi:
            return name
    return None


def elo_int(v):
    try:
        return int(v)
    except ValueError:
        return -1


def process(game_lines, hdr):
    global games_seen
    games_seen += 1
    ev = hdr.get(b"Event", b"")
    if not (b"Rated Classical" in ev or b"Rated Rapid" in ev):
        return
    if hdr.get(b"Termination") != b"Normal":
        return
    we = elo_int(hdr.get(b"WhiteElo", b"?"))
    be = elo_int(hdr.get(b"BlackElo", b"?"))
    if we < 0 or be < 0:
        return
    name = band_of(we, be)
    if name is None or counts[name] >= QUOTA:
        return
    body = b"".join(game_lines)
    if b"%clk" not in body:
        return
    if b" 6." not in body:  # require game to reach move 6
        return
    outs[name].write(body)
    outs[name].write(b"\n")
    counts[name] += 1


def all_done():
    return all(v >= QUOTA for v in counts.values())


game_lines = []
hdr = {}
stdin = sys.stdin.buffer

for line in stdin:
    bytes_in += len(line)
    if line.startswith(b"[Event ") and game_lines:
        process(game_lines, hdr)
        game_lines = []
        hdr = {}
        now = time.time()
        if now - last_status > 30:
            last_status = now
            write_status()
            if all_done():
                break
    game_lines.append(line)
    if line.startswith(b"["):
        i = line.find(b' "')
        if i > 0:
            hdr[line[1:i]] = line[i + 2:line.rfind(b'"')]

if game_lines and not all_done():
    process(game_lines, hdr)

for f in outs.values():
    f.close()
write_status(done=True)
print(json.dumps({k.decode(): v for k, v in counts.items()}), file=sys.stderr)

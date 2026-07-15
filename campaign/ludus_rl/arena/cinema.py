"""Cinematic replay renderer: replay JSON -> 30fps mp4.

Re-simulates the match deterministically, then renders interpolated video with
effects the live client can't afford: glow compositing, analytic particles,
camera punch-ins on tower kills, screen shake, vignette/grade, title cards.
All visuals are procedural or Noto Color Emoji (OFL) — zero third-party game assets.

    python -m arena.cinema /root/ludus_replays/match_X.json --out /root/ludus_videos/match_X.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 30
OW, OH = 720, 1280                 # output
W, H = 18, 32                      # board tiles
SC = 36                            # px per tile (flat texture space)
WW, WH = 900, 1500                 # projected world canvas
HUD_H = 96
# CR-style camera: almost top-down with a gentle tilt — the far edge is only
# ~18% narrower than the near edge and the board fills the frame (no horizon).
CAM_D = 300.0                      # far camera => subtle, CR-like perspective
TOP_Y, BOT_Y = 120.0, WH - 26.0    # projected rows of far/near board edges
_B = (BOT_Y - TOP_Y) / (1.0 / CAM_D - 1.0 / (H + CAM_D))
_A = TOP_Y - _B / (H + CAM_D)


def ground(y):
    """Projected screen row + horizontal scale (px/tile) for board row y.
    Board y=0 (your side) is NEAR the camera: bottom of the screen, largest."""
    d = y + CAM_D
    return _A + _B / d, _B / (d * d) * 0.98


def persp(x, y):
    sy, sc = ground(y)
    return WW / 2 + (x - W / 2) * sc, sy, sc


def find_coeffs(dst, src):
    """PIL perspective coefficients mapping dst quad -> src quad."""
    A = []
    for (X, Y), (x, y) in zip(dst, src):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    B = np.array(src).reshape(8)
    return np.linalg.solve(np.array(A, dtype=np.float64), B).tolist()

EMOJI = {"Bulwark": "🛡️", "Ironhide": "🦏", "Golemite": "🗿", "Shellfort": "🐚",
 "Duskblade": "🗡️", "Mossback": "🌀", "Sporelings": "🍄", "Ratpack": "🐀",
 "Marrowlings": "💀", "Thistlekin": "🌵", "Ashhorde": "⚔️", "Gnatcloud": "🦟",
 "Zephyrling": "🐦", "Wispflock": "🕊️", "Skyray": "🦅", "Emberwing": "🐲",
 "Bonekites": "🦴", "Cloudcalf": "🐮", "Stormkite": "🎈", "Whirligig": "🚁",
 "Emberwitch": "🧙", "Cinderpup": "🐶", "Headsman": "🪓", "Rockhurler": "🪨",
 "Sparkmaw": "⚡", "Longshot": "🏹", "Hexarcher": "🧝", "Quillquick": "🦔",
 "Dart Frog": "🐸", "Prismgunner": "🔫", "Thornsling": "🪃", "Flakbot": "🤖",
 "Frostcaller": "🧊", "Voltmage": "🔮", "Skewerhulk": "🦂", "Timberbeast": "🪵",
 "Cindersprite": "🔥", "Frostsprite": "❄️", "Voltsprite": "💫", "Ramhound": "🐗",
 "Siege Snail": "🐌", "Duneworm": "🪱", "Boarband": "🐖", "Triplet Muses": "🎻",
 "Wallgnashers": "💣", "Burrower": "⛏️", "Gravewalker": "☠️", "Watchpost": "🗼",
 "Tesla Bloom": "🌷", "Boomkiln": "🌋", "Mortar Crab": "🦀", "Skystinger": "🏰",
 "Fireburst": "☄️", "Sparkarc": "🎯", "Stonefall": "🚀", "Glacierlash": "⛄",
 "Shockwave": "⚡", "Overclock": "😡", "Frostfield": "🧊", "Emberrain": "🌩️"}
BLUE, RED = (90, 176, 255), (255, 122, 90)
BLUE_D, RED_D = (47, 95, 148), (148, 64, 47)

EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

# Drop-in art: put PNGs in this folder (git-ignored) named either after the card
# ("Ramhound.png", spaces -> underscores ok) or its analog reference
# ("Hog_Rider.png"). The renderer uses them automatically; emoji is only the
# fallback. The repo ships no third-party art.
ASSET_DIR = Path(__file__).parent / "assets" / "cards"

def _cr_ref_map():
    try:
        from boom.cards import CARD_NAMES, CR_REFS
        return dict(zip(CARD_NAMES, CR_REFS))
    except Exception:
        return {}


def bx(x): return x * SC           # flat texture coords (board painting only)
def by(y): return (H - y) * SC


# ---------------------------------------------------------------- sprites
class Sprites:
    def __init__(self):
        self._emoji = {}
        self._glow = {}
        self._font109 = ImageFont.truetype(EMOJI_FONT, 109)
        self.ui = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        self.ui_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        self.ui_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)

    def emoji(self, ch: str, size: int) -> Image.Image:
        key = (ch, size)
        if key not in self._emoji:
            img = Image.new("RGBA", (140, 140))
            d = ImageDraw.Draw(img)
            d.text((70, 70), ch, font=self._font109, anchor="mm", embedded_color=True)
            box = img.getbbox()
            img = img.crop(box) if box else img
            self._emoji[key] = img.resize((size, size), Image.LANCZOS)
        return self._emoji[key]

    _asset_files = None
    _assets = {}

    def unit(self, name: str, size: int) -> Image.Image:
        """Drop-in PNG from ASSET_DIR (by card name or analog ref), else emoji."""
        if Sprites._asset_files is None:
            Sprites._asset_files = {p.stem.lower().replace("_", " "): p
                                    for p in ASSET_DIR.glob("*.png")} if ASSET_DIR.exists() else {}
            Sprites._ref = _cr_ref_map()
        key = (name, size)
        if key in Sprites._assets:
            return Sprites._assets[key]
        p = (Sprites._asset_files.get(name.lower())
             or Sprites._asset_files.get(Sprites._ref.get(name, "").lower()))
        if p:
            img = Image.open(p).convert("RGBA")
            img.thumbnail((size, size), Image.LANCZOS)
            Sprites._assets[key] = img
            return img
        return self.emoji(EMOJI.get(name, "❔"), size)

    def glow(self, color, size: int) -> Image.Image:
        key = (color, size)
        if key not in self._glow:
            r = size // 2
            yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
            d = np.sqrt(xx ** 2 + yy ** 2) / r
            a = np.clip(1 - d, 0, 1) ** 1.7
            img = np.zeros((size - size % 2, size - size % 2, 4), np.uint8)
            for i, c in enumerate(color):
                img[..., i] = c
            img[..., 3] = (a * 255).astype(np.uint8)
            self._glow[key] = Image.fromarray(img, "RGBA")
        return self._glow[key]


def paste_center(dst, src, x, y):
    dst.alpha_composite(src, (int(x - src.width / 2), int(y - src.height / 2)))


# ---------------------------------------------------------------- board
def paint_board_flat() -> Image.Image:
    img = Image.new("RGBA", (W * SC, H * SC), (13, 17, 23, 255))
    d = ImageDraw.Draw(img)
    rng = random.Random(7)
    for ty in range(H):
        for tx in range(W):
            top = ty >= 17
            dark = (tx + ty) % 2
            base = (27, 47, 35) if top else (32, 53, 31)
            if dark:
                base = tuple(c - 3 for c in base)
            j = rng.randint(0, 5)
            d.rectangle([bx(tx), by(ty + 1), bx(tx + 1), by(ty)],
                        fill=(base[0] + j, base[1] + j, base[2] + j))
    # subtle noise
    noise = (np.random.default_rng(3).normal(0, 4, (H * SC, W * SC, 1))
             .clip(-8, 8)).astype(np.int16)
    arr = np.asarray(img).astype(np.int16)
    arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    # river with vertical gradient
    for i in range(2 * SC):
        t = i / (2 * SC)
        c = (int(29 + 15 * math.sin(t * math.pi)), int(82 + 24 * math.sin(t * math.pi)),
             int(115 + 33 * math.sin(t * math.pi)))
        d.line([(0, by(17) + i), (W * SC, by(17) + i)], fill=c)
    # bridges
    for bxt in (3, 14):
        x0, y0 = bx(bxt) - 4, by(17) - 5
        d.rounded_rectangle([x0, y0, x0 + 2 * SC + 8, y0 + 2 * SC + 10], 8,
                            fill=(138, 106, 58))
        for i in range(1, 6):
            yy = y0 + i * (2 * SC + 10) / 6
            d.line([(x0 + 3, yy), (x0 + 2 * SC + 5, yy)], fill=(90, 69, 38), width=2)
    return img


def paint_board() -> Image.Image:
    """Gently projected board (CR-like near-top-down view), filling the canvas."""
    flat = paint_board_flat()
    world = Image.new("RGBA", (WW, WH), (12, 15, 21, 255))
    corners_board = [(0.0, float(H)), (float(W), float(H)), (0.0, 0.0), (float(W), 0.0)]
    dst = [persp(x, y)[:2] for x, y in corners_board]  # TL(far) TR(far) BL(near) BR(near)
    src = [(0, 0), (W * SC, 0), (0, H * SC), (W * SC, H * SC)]
    coeffs = find_coeffs(dst, src)
    warped = flat.transform((WW, WH), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    world.alpha_composite(warped)
    return world


# ---------------------------------------------------------------- simulation
def simulate(replay: dict):
    """Deterministic re-simulation -> list of (frame, events) per tick."""
    import jax
    import jax.numpy as jnp

    from boom import engine
    from boom.render import frame_events, render_frame
    step = jax.jit(engine.step)
    decks = None if replay.get("decks") is None else jnp.asarray(replay["decks"], jnp.int32)
    state = engine.reset(jax.random.PRNGKey(replay["seed"]), decks)
    out = [(render_frame(state), [])]
    for a0, a1 in replay["action_log"]:
        played = []
        for p, a in ((0, a0), (1, a1)):
            if a[0] < 4:
                card = int(np.asarray(state.hand)[p, a[0]])
                ex, ey = (a[1], a[2]) if p == 0 else (engine.W - 1 - a[1], engine.H - 1 - a[2])
                played.append((card, ex, ey))
            else:
                played.append(None)
        new = step(state, jnp.array([a0, a1], jnp.int32), None)
        frame = render_frame(new)
        evs = frame_events(state, new, played)
        out.append((frame, evs))
        state = new
        if frame["result"] != -1:
            break
    return out


# ---------------------------------------------------------------- rendering
def lerp(a, b, t): return a + (b - a) * t


def render(replay_path: str, out_path: str, preview_at: float | None = None):
    replay = json.loads(Path(replay_path).read_text())
    ticks = simulate(replay)
    S = Sprites()
    board = paint_board()
    result = ticks[-1][0]["result"]

    n_frames = int(len(ticks) / 5 * FPS)
    intro_f, outro_f = FPS * 2, FPS * 3

    # event timeline in seconds: (t0, ev)
    timeline = []
    for i, (_, evs) in enumerate(ticks):
        for e in evs:
            timeline.append((i / 5.0, e))

    class _Null:
        def write(self, *_): pass
        def close(self): pass
    if preview_at is None:
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
            stdin=subprocess.PIPE)
    else:
        ff = type("F", (), {"stdin": _Null(), "wait": lambda self: 0,
                            "returncode": 0})()

    # camera keyframes: punch in on tower falls
    punches = [(t0, e) for t0, e in timeline if e["e"] == "tower_fall"]

    def camera(t):
        zoom = 1.0 + 0.05 * min(1, t / max(len(ticks) / 5, 1))
        focus = (WW / 2, WH / 2)
        shake = 0.0
        for pt, e in punches:
            dt = t - pt
            if -0.4 < dt < 1.6:
                k = 1 - abs(dt - 0.5) / 1.1
                k = max(0, min(1, k))
                zoom = max(zoom, 1 + 0.4 * k)
                focus = (lerp(WW / 2, bx(e["x"]), k), lerp(WH / 2, by(e["y"]), k))
            if 0 <= dt < 0.5:
                shake = 12 * (1 - dt / 0.5)
        return zoom, focus, shake

    total = intro_f + n_frames + outro_f
    for fi in range(total):
        if fi < intro_f:                                   # title card
            img = Image.new("RGB", (OW, OH), (13, 17, 23))
            d = ImageDraw.Draw(img)
            d.text((OW / 2, OH / 2 - 60), "⚔ BOOM", font=S.ui_big, anchor="mm",
                   fill=(230, 235, 245))
            d.text((OW / 2, OH / 2 + 10), "LUDUS REPLAY", font=S.ui, anchor="mm",
                   fill=(120, 150, 190))
            d.text((OW / 2, OH / 2 + 60), Path(replay_path).stem, font=S.ui_small,
                   anchor="mm", fill=(90, 110, 140))
            ff.stdin.write(img.tobytes())
            continue

        t = (fi - intro_f) / FPS                            # sim seconds
        s = min(t * 5, len(ticks) - 1.001)
        i0, al = int(s), s - int(s)
        fr0, fr1 = ticks[i0][0], ticks[min(i0 + 1, len(ticks) - 1)][0]

        world = board.copy()
        dw = ImageDraw.Draw(world)

        # towers: extruded pseudo-3D keeps, far-to-near
        for tw in sorted(fr1["towers"], key=lambda t: -t["y"]):
            x, y, sc = persp(tw["x"], tw["y"])
            w = (3.2 if tw["kind"] == "core" else 2.4) * sc
            hgt = (2.6 if tw["kind"] == "core" else 2.0) * sc
            if tw["hp"] <= 0:
                paste_center(world, S.emoji("🪨", max(8, int(w * 0.7))), x, y)
                continue
            col = BLUE_D if tw["owner"] == 0 else RED_D
            edge = BLUE if tw["owner"] == 0 else RED
            dark = tuple(int(c * 0.55) for c in col)
            hgt = w * 0.5                        # squat keep, not a slab
            # ground plinth
            dw.ellipse([x - w*0.58, y - w*0.20, x + w*0.58, y + w*0.20],
                       fill=dark, outline=edge, width=2)
            # body walls
            dw.polygon([(x - w/2, y - hgt), (x + w/2, y - hgt),
                        (x + w*0.58, y), (x - w*0.58, y)], fill=dark)
            # top face
            dw.rounded_rectangle([x - w/2, y - hgt - w*0.16, x + w/2, y - hgt + w*0.16],
                                 int(5 * sc / 36) + 2, fill=col, outline=edge, width=3)
            paste_center(world, S.emoji("👑" if tw["kind"] == "core" else "🏰",
                                        max(12, int(w * 0.66))), x, y - hgt - w*0.02)
            frac = tw["hp"] / tw["max_hp"]
            ty0 = y - hgt - w*0.3 - 16
            dw.rounded_rectangle([x - w/2, ty0, x + w/2, ty0 + 8], 4, fill=(0, 0, 0, 170))
            dw.rounded_rectangle([x - w/2, ty0, x - w/2 + w * frac, ty0 + 8], 4, fill=edge)
            dw.text((x, ty0 - 13), str(tw["hp"]), font=S.ui_small, anchor="mm",
                    fill=(255, 255, 255))

        # units (interpolated, y-sorted)
        prev_by_id = {(u["id"], u["card"]): u for u in fr0["units"]}
        units = sorted(fr1["units"], key=lambda u: -u["y"])
        for u in units:
            p = prev_by_id.get((u["id"], u["card"]))
            ux = lerp(p["x"], u["x"], al) if p else u["x"]
            uy = lerp(p["y"], u["y"], al) if p else u["y"]
            x, y, sc = persp(ux, uy)
            r = sc * (0.42 + min(u["max_hp"], 3800) / 3800 * 0.34)
            bob = math.sin(t * 22 + u["id"]) * sc * 0.06 if p and (p["x"] != u["x"] or p["y"] != u["y"]) else 0
            lift = sc * 0.35 if u["air"] else 0
            edge = BLUE if u["owner"] == 0 else RED
            fill = BLUE_D if u["owner"] == 0 else RED_D
            dw.ellipse([x - r*0.85, y - r*0.26, x + r*0.85, y + r*0.26], fill=(0, 0, 0, 100))
            yy = y - r * 0.85 - bob - lift          # billboard base sits on the ground point
            dw.ellipse([x - r*0.5, y - r*0.16, x + r*0.5, y + r*0.16],
                       fill=fill, outline=edge, width=2)   # team base ring on the ground
            if u["rage"]:
                dw.ellipse([x - r*0.6, y - r*0.2, x + r*0.6, y + r*0.2],
                           outline=(255, 221, 102), width=3)
            if u["slow"]:
                dw.ellipse([x - r*0.6, y - r*0.2, x + r*0.6, y + r*0.2],
                           outline=(136, 221, 255), width=3)
            spr = S.unit(u["name"], max(10, int(r * 1.9)))
            paste_center(world, spr, x, yy)
            if u["hp"] < u["max_hp"]:
                dw.rectangle([x - r, yy - r - 8, x + r, yy - r - 4], fill=(0, 0, 0, 150))
                dw.rectangle([x - r, yy - r - 8, x - r + 2*r*u["hp"]/u["max_hp"], yy - r - 4],
                             fill=edge)

        # effects
        for t0, e in timeline:
            dt = t - t0
            if dt < 0 or dt > 1.2:
                continue
            if e["e"] == "cast" and dt <= e["eta"] * 0.2:
                p = dt / (e["eta"] * 0.2)
                x0_, y0_, _ = persp(e["x0"], e["y0"])
                x1_, y1_, sc1 = persp(e["x"], e["y"])
                x = lerp(x0_, x1_, p)
                y = lerp(y0_, y1_, p) - 130 * p * (1 - p) - 10
                col = (201, 163, 255) if e["owner"] == 0 else (255, 176, 138)
                paste_center(world, S.glow(col, int(26 + 10 * math.sin(p * 3.14))), x, y)
                paste_center(world, S.glow((255, 240, 200), 12), x, y)
            elif e["e"] == "shot" and dt <= 0.16:
                p = dt / 0.16
                x0_, y0_, _ = persp(e["x0"], e["y0"])
                x1_, y1_, _ = persp(e["x1"], e["y1"])
                x = lerp(x0_, x1_, p)
                y = lerp(y0_, y1_, p) - 14
                col = (159, 208, 255) if e["owner"] == 0 else (255, 176, 138)
                paste_center(world, S.glow(col, 42), x, y)
                paste_center(world, S.glow((255, 233, 160), 20), x, y)
            elif e["e"] == "spell" and dt <= 0.5:
                p = dt / 0.5
                x, y, sc = persp(e["x"], e["y"])
                R = e["r"] * sc * (0.4 + 0.7 * p)
                col = (159, 216, 255) if e.get("effect") == 1 else \
                      (255, 179, 90) if e.get("effect") == 2 else (255, 210, 58)
                a = int(200 * (1 - p))
                dw.ellipse([x - R, y - R * 0.4, x + R, y + R * 0.4],
                           outline=col + (a,), width=5)
                paste_center(world, S.glow(col, max(8, int(R))), x, y - R * 0.2)
            elif e["e"] in ("death", "deploy", "tower_fall"):
                dur = 1.1 if e["e"] == "tower_fall" else 0.45
                if dt > dur:
                    continue
                n = 40 if e["e"] == "tower_fall" else 9
                col = (255, 171, 74) if e["e"] == "tower_fall" else \
                      (201, 180, 138) if e["e"] == "deploy" else (200, 200, 210)
                ex, ey, esc = persp(e["x"], e["y"])
                rng = random.Random(int(t0 * 1000) + n)
                k_v = esc / 36.0
                for k in range(n):
                    a0 = rng.random() * 6.283
                    v = (60 if e["e"] == "tower_fall" else 30) * (0.4 + rng.random()) * k_v
                    px_ = ex + math.cos(a0) * v * dt
                    py_ = ey - 10 + math.sin(a0) * v * dt * 0.6 + 40 * dt * dt * k_v
                    fade = 1 - dt / dur
                    paste_center(world, S.glow(col, max(5, int(16 * fade * k_v + 4))), px_, py_)
            elif e["e"] == "dmg" and dt <= 0.7:
                p = dt / 0.7
                col = (255, 154, 122) if e["owner"] == 0 else (255, 210, 58)
                x, y, _sc = persp(e["x"], e["y"])
                y = y - _sc * 0.9 - 26 * p
                txt = f"-{e['n']}"
                f = S.ui if e.get("tower") else S.ui_small
                for ox, oy in ((-2,0),(2,0),(0,-2),(0,2)):
                    dw.text((x+ox, y+oy), txt, font=f, anchor="mm", fill=(0, 0, 0))
                dw.text((x, y), txt, font=f, anchor="mm", fill=col)

        # HUD strip
        dw.rectangle([0, 0, WW, HUD_H - 8], fill=(10, 13, 18))
        tick_now = int(s)
        left = max(0, (900 - tick_now if tick_now < 900 else 1200 - tick_now)) / 5
        mm, ss = int(left // 60), int(left % 60)
        phase = "OVERTIME" if tick_now >= 900 else ""
        dw.text((WW / 2, 34), f"{mm}:{ss:02d}", font=S.ui, anchor="mm",
                fill=(255, 215, 130) if phase else (230, 235, 245))
        if phase:
            dw.text((WW / 2, 66), phase, font=S.ui_small, anchor="mm", fill=(255, 160, 90))
        blue_dead = sum(1 for tw in fr1["towers"][3:] if tw["hp"] <= 0)
        red_dead = sum(1 for tw in fr1["towers"][:3] if tw["hp"] <= 0)
        paste_center(world, S.emoji("👑", 30), 46, 34)
        dw.ellipse([64, 22, 88, 46], fill=BLUE)
        dw.text((76, 34), str(blue_dead), font=S.ui_small, anchor="mm", fill=(0, 0, 30))
        paste_center(world, S.emoji("👑", 30), WW - 46, 34)
        dw.ellipse([WW - 88, 22, WW - 64, 46], fill=RED)
        dw.text((WW - 76, 34), str(red_dead), font=S.ui_small, anchor="mm", fill=(40, 0, 0))

        # camera
        zoom, focus, shake = camera(t)
        cw, ch = WW / zoom, WH / zoom
        cx = min(max(focus[0], cw / 2), WW - cw / 2) + random.Random(fi).uniform(-shake, shake)
        cy = min(max(focus[1], ch / 2), WH - ch / 2) + random.Random(fi + 1).uniform(-shake, shake)
        crop = world.crop((int(cx - cw / 2), int(cy - ch / 2),
                           int(cx + cw / 2), int(cy + ch / 2)))
        frame_img = crop.convert("RGB").resize((OW, OH), Image.BILINEAR)

        # vignette + grade
        if fi == intro_f:
            yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
            dd = np.sqrt(((xx - OW/2) / (OW/2)) ** 2 + ((yy - OH/2) / (OH/2)) ** 2)
            render.vign = (1 - 0.25 * np.clip(dd - 0.5, 0, 1) ** 1.5)[..., None]
        arr = np.asarray(frame_img).astype(np.float32) * getattr(render, "vign", 1)
        arr[..., 2] *= 1.04                                     # cool grade
        out_arr = np.clip(arr, 0, 255).astype(np.uint8)
        if preview_at is not None and t >= preview_at:
            Image.fromarray(out_arr).save(out_path)
            print(f"preview frame at t={t:.1f}s -> {out_path}")
            return
        ff.stdin.write(out_arr.tobytes())

    # outro: result banner over last frame
    txt = "BLUE WINS" if result == 0 else "RED WINS" if result == 1 else "DRAW"
    col = BLUE if result == 0 else RED if result == 1 else (200, 200, 200)
    base = frame_img.filter(ImageFilter.GaussianBlur(4))
    for fi in range(outro_f):
        img = base.copy()
        d = ImageDraw.Draw(img)
        a = min(1, fi / (FPS * 0.5))
        d.rectangle([0, OH/2 - 90, OW, OH/2 + 90], fill=(0, 0, 0))
        d.text((OW/2, OH/2 - 14), txt, font=S.ui_big, anchor="mm",
               fill=tuple(int(c * a) for c in col))
        d.text((OW/2, OH/2 + 46), "ludus · deterministic replay", font=S.ui_small,
               anchor="mm", fill=(int(120*a),) * 3)
        ff.stdin.write(img.tobytes())

    ff.stdin.close()
    ff.wait()
    assert ff.returncode == 0, "ffmpeg failed"
    print(f"rendered {out_path} ({total + outro_f} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview-at", type=float, default=None,
                    help="render a single PNG frame at this sim-second instead")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render(args.replay, args.out, preview_at=args.preview_at)

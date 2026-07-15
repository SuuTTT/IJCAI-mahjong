"""Server-authoritative play + replay server (WO-P0-03 slice).

Human = player 0 in a browser; bot = player 1 (random-legal v0, acts once per second).
The SAME JAX core steps matches at 5 ticks/s; browsers only render and send inputs.
Finished matches are appended to REPLAY_DIR as (env_version, seed, action_log, outcome)
— the docs/01 replay schema — and can be re-watched at /play?replay=<id>, which
re-simulates from the log (bit-identical by the determinism contract).

Run:  uvicorn arena.play_server:app --host 0.0.0.0 --port 8791
"""

from __future__ import annotations

import asyncio
import faulthandler
import signal

faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> dumps all stacks
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

import jax
# NOTE: do NOT enable the persistent compilation cache here — with it enabled,
# jitted step outputs intermittently hang forever in device_get on this box
# (repro'd at length on 2026-07-04; the morning's stable server ran without it)
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

import boom.cards as cards_mod
from boom import ENV_VERSION, engine, vec
from boom.render import _card_info, frame_events, hand_info, render_frame

TICK_S = 0.2
BOT_PERIOD = 5                      # scripted/random bots act once per second
REPLAY_DIR = Path("/root/ludus_replays")
VIDEO_DIR = Path("/root/ludus_videos")
LEAGUE_DIR = Path("/root/ludus_train/league")
BOTS_DIR = Path("/root/ludus_bots")
BOTS_DIR.mkdir(parents=True, exist_ok=True)
PPO_PARAMS = LEAGUE_DIR / "champion.msgpack"

app = FastAPI(title="Ludus play server")
STATIC = Path(__file__).parent / "static"

from fastapi.staticfiles import StaticFiles  # noqa: E402
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=Path(__file__).parents[1] / "assets"),
          name="assets")
VIDEO_DIR_ = Path("/root/ludus_videos")
VIDEO_DIR_.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR_), name="videos")

_step = jax.jit(engine.step)
_reset = jax.jit(engine.reset)
_legal = jax.jit(engine.legal, static_argnums=1)
_bot_action = jax.jit(
    lambda k, s: vec.flat_to_triple(vec.random_legal_action(k, s, 1)))

NOOP = [4, 0, 0]


# ---------------------------------------------------------------- opponents
def _make_rule_bot(seat: int = 1):
    from baselines.rule_bot import rule_v0_action
    fn = jax.jit(lambda s: rule_v0_action(s, seat))
    return lambda key, state, tick: (
        [int(v) for v in np.asarray(fn(state))] if tick % BOT_PERIOD == 0 else NOOP)


def _make_random_bot(seat: int = 1):
    rnd = jax.jit(lambda k, s: vec.flat_to_triple(vec.random_legal_action(k, s, seat)))

    def act(key, state, tick):
        if tick % BOT_PERIOD != 0:
            return NOOP
        return [int(v) for v in np.asarray(rnd(key, state))]
    return act


_ppo_shared = {"net": None, "tmpl": None, "greedy": {}}


def _ppo_infra():
    """ONE jitted greedy per seat, params passed as an argument — a new checkpoint
    (league promotion every ~15 min) must never trigger a recompile in the event
    loop; that wedge froze every websocket handshake for minutes."""
    if _ppo_shared["net"] is None:
        from baselines.ppo_selfplay import ActorCritic
        from boom.engine import H, OBS_C, OBS_VEC, W
        net = ActorCritic()
        _ppo_shared["net"] = net
        _ppo_shared["tmpl"] = net.init(jax.random.PRNGKey(0),
                                       jnp.zeros((1, H, W, OBS_C), jnp.float32),
                                       jnp.zeros((1, OBS_VEC), jnp.float32))
        for seat in (0, 1):
            def mk(seat):
                @jax.jit
                def greedy(params, state, key):
                    # low-temperature sampling, not argmax: greedy play amplified
                    # degenerate no-op loops in bot-vs-bot showcases
                    obs = engine.observe(state, seat)
                    mask = vec.flat_legal(state, seat)
                    logits, _ = net.apply(params, obs.spatial[None], obs.vector[None])
                    return vec.flat_to_triple(jax.random.categorical(
                        key, jnp.where(mask, logits[0], -1e9) / 0.6))
                return greedy
            _ppo_shared["greedy"][seat] = mk(seat)
    return _ppo_shared


def _make_ppo_bot(path: Path = None, seat: int = 1):
    """Greedy policy from a checkpoint (champion by default). Cheap: only a
    parameter load — the jitted function is shared and pre-compiled."""
    from flax.serialization import from_bytes
    infra = _ppo_infra()
    params = from_bytes(infra["tmpl"], (path or PPO_PARAMS).read_bytes())
    greedy = infra["greedy"][seat]

    def act(key, state, tick):
        return [int(v) for v in np.asarray(greedy(params, state, key))]
    return act


_ppo_cache: dict = {}


def _cached_ppo(path: Path, name: str, seat: int = 1):
    key = f"{name}@{seat}"
    mtime = path.stat().st_mtime
    ent = _ppo_cache.get(key)
    if not ent or ent[0] != mtime:
        _ppo_cache[key] = (mtime, _make_ppo_bot(path, seat))
    return _ppo_cache[key][1]


def make_bot(kind: str, allow_cold: bool = True, seat: int = 1):
    if not allow_cold and not _warm["done"] and f"{kind}@{seat}" not in _ppo_cache \
            and (kind.startswith(("gen:", "user:")) or kind == "ppo"):
        return _make_random_bot(seat), "random_v0 (champion warming up — retry soon)"
    if kind.startswith("user:"):                   # uploaded weights-only bots
        nm = re.sub(r"[^a-zA-Z0-9_-]", "", kind[5:])[:24]
        p = BOTS_DIR / f"{nm}.msgpack"
        if p.exists():
            try:
                return _cached_ppo(p, kind, seat), f"user bot {nm}"
            except Exception:
                pass
    if kind.startswith("gen:"):                    # fight any league generation
        p = LEAGUE_DIR / f"gen_{int(kind[4:])}.msgpack"
        if p.exists():
            try:
                return _cached_ppo(p, kind, seat), f"league gen {int(kind[4:])}"
            except Exception:
                pass
    if kind == "ppo" and PPO_PARAMS.exists():
        try:
            return _cached_ppo(PPO_PARAMS, "champion", seat), "league champion"
        except Exception:
            pass
    if kind.startswith("graph:"):                  # composed rule policies
        nm = re.sub(r"[^a-zA-Z0-9_-]", "", kind[6:])[:24]
        if (GRAPHS_DIR / f"{nm}.json").exists():
            try:
                return _make_graph_bot(nm, seat), f"composed bot {nm}"
            except Exception:
                pass
    if kind == "rule":
        return _make_rule_bot(seat), "rule_v0"
    return _make_random_bot(seat), "random_v0"


GRAPHS_DIR = Path("/root/ludus_graphs")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------- composer (Tier 3 v0)
from boom import graphs as graphs_mod

_SPOTS = graphs_mod.SPOTS


def _make_graph_bot(name: str, seat: int = 1):
    return graphs_mod.make_policy(graphs_mod.load_rules(name), seat)


_warm = {"done": False}


def _warmup():
    st = _reset(jax.random.PRNGKey(0), None)
    # TWO ticks: a reset-born state and a step-born state are distinct jit cache
    # entries (weak-type metadata differs) — missing the second signature made
    # every real match recompile the engine (~5 min on this CPU) at tick 2,
    # which presented all day as a server deadlock
    st = _step(st, jnp.array([NOOP, NOOP], jnp.int32), None)
    st = _step(st, jnp.array([NOOP, NOOP], jnp.int32), None)
    jax.block_until_ready(st.u_hp)
    print(f"engine warm: {_step._cache_size()} signature(s)", flush=True)
    _legal(st, 0)
    _bot_action(jax.random.PRNGKey(1), st)
    jax.block_until_ready(st.u_hp)
    # pre-compile every opponent FOR BOTH SEATS: a cold first call inside the
    # event loop wedges every websocket handshake for minutes
    for kind in ("ppo", "rule", "random"):
        for seat in (1, 0):
            bot, name = make_bot(kind, seat=seat)
            bot(jax.random.PRNGKey(2), st, 0)
        print(f"warmed bot: {kind} -> {name} (both seats)", flush=True)
    _warm["done"] = True


from concurrent.futures import ThreadPoolExecutor

# ONE dedicated thread owns ALL JAX work (warmup, ticks, state reads). Two hard
# constraints meet here: the event loop must never block on a game tick, and
# JAX deadlocks when a function compiled on one thread is called from another
# (verified: warm jitted step from a pool thread hangs forever). So every JAX
# touch is serialized onto this single worker.
_jax_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jaxworker")


async def off_loop(fn, *a):
    # TEMPORARILY synchronous: JAX on this box hangs when jitted calls run in
    # executor threads (repro'd; cache/thread interaction under investigation).
    # Main-thread ticks are the config that ran stably all morning. The await
    # points still yield between ticks, and the busy-spin cap remains.
    return fn(*a)


def apply_tick(state: engine.State, a0: list[int], a1: list[int]):
    """Step one tick and build the render frame + visual events. Shared by live
    play and replay streaming so both look identical."""
    played = []
    for p, a in ((0, a0), (1, a1)):
        if a[0] < 4:
            card = int(np.asarray(state.hand)[p, a[0]])
            ex, ey = (a[1], a[2]) if p == 0 else (engine.W - 1 - a[1], engine.H - 1 - a[2])
            played.append((card, ex, ey))
        else:
            played.append(None)
    new = _step(state, jnp.array([a0, a1], jnp.int32), None)
    frame = render_frame(new)
    frame["type"] = "frame"
    frame["events"] = frame_events(state, new, played)
    return new, frame


# ---------------------------------------------------------------- pages & API
@app.get("/")
async def home():
    return FileResponse(STATIC / "home.html")


@app.get("/play3d")
async def play3d_page():
    return FileResponse(STATIC / "play3d.html")


@app.get("/play")
async def play():
    return FileResponse(STATIC / "index.html")


@app.get("/ladder")
async def ladder():
    return FileResponse(STATIC / "ladder.html")


@app.get("/replays")
async def replays_page():
    return FileResponse(STATIC / "replays.html")


@app.get("/cards")
async def cards_page():
    return FileResponse(STATIC / "cards.html")


@app.get("/api/cards")
async def api_cards():
    return JSONResponse([_card_info(c) for c in range(len(cards_mod.CARDS.cost))])


def parse_deck(spec: str | None):
    """'id,id,...' -> (2,8) decks array (p0 custom, p1 = bot deck B), or None."""
    if not spec:
        return None
    try:
        ids = [int(x) for x in spec.split(",")]
    except ValueError:
        return None
    if len(ids) != 8 or len(set(ids)) != 8 or not all(0 <= i < 60 for i in ids):
        return None
    return jnp.stack([jnp.asarray(ids, jnp.int32), jnp.asarray(cards_mod.DECK_B)])


_render_jobs: dict[str, subprocess.Popen] = {}


@app.post("/api/render/{rid}")
async def start_render(rid: str):
    if not re.fullmatch(r"match_\d+_\d+", rid):
        return JSONResponse({"error": "bad id"}, status_code=400)
    mp4 = VIDEO_DIR / f"{rid}.mp4"
    job = _render_jobs.get(rid)
    if job and job.poll() is None:
        return {"status": "rendering"}
    if mp4.exists() and (job is None or job.poll() == 0):
        return {"status": "ready", "url": f"/videos/{rid}.mp4"}
    src = REPLAY_DIR / f"{rid}.json"
    if not src.exists():
        return JSONResponse({"error": "no such replay"}, status_code=404)
    _render_jobs[rid] = subprocess.Popen(
        [sys.executable, "-m", "arena.cinema", str(src), "--out", str(mp4)],
        cwd=str(Path(__file__).parents[1]),
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        stdout=open(VIDEO_DIR / f"{rid}.log", "w"), stderr=subprocess.STDOUT)
    return {"status": "started"}


@app.get("/api/render/{rid}")
async def render_status(rid: str):
    mp4 = VIDEO_DIR / f"{rid}.mp4"
    job = _render_jobs.get(rid)
    if job and job.poll() is None:
        return {"status": "rendering"}
    if mp4.exists() and (job is None or job.poll() == 0):
        # orphaned file with no job record (server restart): only trust it once
        # ffmpeg is no longer appending
        import time as _t
        if job or _t.time() - mp4.stat().st_mtime > 15:
            return {"status": "ready", "url": f"/videos/{rid}.mp4"}
        return {"status": "rendering"}
    if job and job.poll() not in (None, 0):
        tail = (VIDEO_DIR / f"{rid}.log").read_text()[-300:] if (VIDEO_DIR / f"{rid}.log").exists() else ""
        return {"status": "failed", "log": tail}
    return {"status": "idle"}


# ---------------------------------------------------------------- game tables
_tables: dict[str, dict] = {}


@app.post("/api/table")
async def create_table(request: Request):
    """创建游戏桌: pick each seat = 'human' or any bot id. Returns join links."""
    body = await request.json()
    s0 = str(body.get("seat0", "human"))[:32]
    s1 = str(body.get("seat1", "ppo"))[:32]
    tid = secrets.token_hex(3)
    _tables[tid] = {"seats": {0: s0, 1: s1}, "conns": {}, "pending": {0: None, 1: None},
                    "running": False, "created": time.time(), "restart": False}
    join = {f"join{k}": f"/play?table={tid}&seat={k}"
            for k in (0, 1) if _tables[tid]["seats"][k] == "human"}
    return {"table": tid, **join, "watch": f"/play?table={tid}&seat=spec"}


@app.get("/api/tables")
async def list_tables():
    out = []
    for tid, t in list(_tables.items()):
        if time.time() - t["created"] > 3600:
            _tables.pop(tid, None)
            continue
        open_seats = [k for k in (0, 1)
                      if t["seats"][k] == "human" and k not in t["conns"]]
        out.append({"table": tid, "seat0": t["seats"][0], "seat1": t["seats"][1],
                    "open_seats": open_seats, "running": t["running"]})
    return JSONResponse(out)


@app.get("/tables")
async def tables_page():
    return FileResponse(STATIC / "tables.html")


async def _broadcast(t, payload_by_seat):
    dead = []
    for seat, sock in list(t["conns"].items()):
        try:
            await sock.send_text(payload_by_seat(seat))
        except Exception:
            dead.append(seat)
    for seat in dead:
        t["conns"].pop(seat, None)


async def run_room(tid: str):
    """Match loop for a game table: any mix of human and bot seats."""
    try:
        await _run_room_inner(tid)
    except Exception:
        import traceback
        print(f"run_room({tid}) crashed:", flush=True)
        traceback.print_exc()
        _tables.pop(tid, None)


async def _run_room_inner(tid: str):
    t = _tables[tid]
    bots = {}
    for k in (0, 1):
        if t["seats"][k] != "human":
            bots[k] = make_bot(t["seats"][k], allow_cold=False, seat=k)[0]
    print(f"room {tid}: loop starting, bots={list(bots)}", flush=True)
    while tid in _tables:
        seed = secrets.randbits(31)
        rng = {}                              # jax key: born and used on worker only
        state = await off_loop(lambda: _reset(jax.random.PRNGKey(seed), None))
        t["restart"] = False
        first = True
        next_t = time.monotonic()
        done = False
        while not done and tid in _tables and not t["restart"]:
            next_t = max(next_t + TICK_S, time.monotonic() - TICK_S)

            def _room_tick(state=state):
                if "key" not in rng:
                    rng["key"] = jax.random.PRNGKey(seed ^ 0x0F0F0F)
                tick = int(state.tick)
                acts = {}
                for k in (0, 1):
                    if k in bots:
                        rng["key"], k2 = jax.random.split(rng["key"])
                        acts[k] = bots[k](k2, state, tick)
                    else:
                        acts[k] = t["pending"][k] or NOOP
                        t["pending"][k] = None
                new, frame = apply_tick(state, acts[0], acts[1])
                hands = {k: hand_info(new, k) for k in (0, 1) if k not in bots}
                return new, frame, hands
            state, frame, h = await off_loop(_room_tick)
            if first:
                print(f"room {tid}: first frame ready", flush=True)
                first = False
            # zombie guard: a room whose humans all left stops simulating
            if not t["conns"]:
                t["idle"] = t.get("idle", 0) + 1
                if t["idle"] > 50:
                    _tables.pop(tid, None)
                    return
            else:
                t["idle"] = 0
            h = {k: hand_info(state, k) for k in (0, 1) if k not in bots}

            def payload(seat, frame=frame, h=h):
                f = dict(frame)
                if isinstance(seat, int) and seat in h:
                    f["hand"] = h[seat]
                return json.dumps(f)
            await _broadcast(t, payload)
            done = frame["result"] != -1
            if not done:
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
        if tid not in _tables:
            return
        # hold on the result screen until someone restarts or everyone leaves
        while tid in _tables and not t["restart"]:
            if not t["conns"]:
                _tables.pop(tid, None)
                return
            await asyncio.sleep(0.3)


async def run_table_seat(sock: WebSocket, tid: str, seat_q: str):
    t = _tables.get(tid)
    if t is None:
        await sock.close(code=4404)
        return
    if seat_q == "spec":
        seat = f"spec{secrets.token_hex(2)}"
    else:
        seat = int(seat_q)
        if t["seats"].get(seat) != "human" or seat in t["conns"]:
            await sock.close(code=4409)
            return
    t["conns"][seat] = sock
    other = t["seats"][1 if seat == 0 else 0] if isinstance(seat, int) else None
    await sock.send_text(json.dumps({
        "type": "meta", "table": tid, "seat": seat if isinstance(seat, int) else "spec",
        "bot": f"table {tid} · vs {other}" if other else f"table {tid} (spectating)",
        "waiting": any(t["seats"][k] == "human" and k not in t["conns"]
                       for k in (0, 1))}))
    humans_ready = all(t["seats"][k] != "human" or k in t["conns"] for k in (0, 1))
    if humans_ready and not t["running"]:
        t["running"] = True
        # strong reference: the loop keeps only weak refs to tasks — an
        # unreferenced room task gets garbage-collected mid-match, silently
        t["task"] = asyncio.ensure_future(run_room(tid))
        await _broadcast(t, lambda s: json.dumps({"type": "start"}))
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if not isinstance(seat, int):
                continue
            if msg.get("type") == "play":
                sx, xx, yy = int(msg["slot"]), int(msg["x"]), int(msg["y"])
                # per-seat legality in that player's own frame
                st = None
                # trust engine: enqueue; illegal plays become counted no-ops
                t["pending"][seat] = [sx, xx, yy]
            elif msg.get("type") == "restart":
                t["restart"] = True
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        t["conns"].pop(seat, None)


@app.get("/api/debug/stacks")
async def debug_stacks():
    import io
    import traceback as tb
    buf = io.StringIO()
    frames = sys._current_frames()
    import threading as th
    names = {t.ident: t.name for t in th.enumerate()}
    for tid, frm in frames.items():
        buf.write(f"\n=== thread {names.get(tid, tid)} ===\n")
        tb.print_stack(frm, file=buf)
    for task in asyncio.all_tasks():
        buf.write(f"\n=== task {task.get_name()} done={task.done()} ===\n")
        for f in task.get_stack(limit=6):
            buf.write(f"  {f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_lineno} {f.f_code.co_name}\n")
    return JSONResponse({"stacks": buf.getvalue()})


USERS_F = Path("/root/ludus_users.json")
OWNERS_F = BOTS_DIR / "owners.json"


@app.post("/api/register")
async def api_register(request: Request):
    """Accounts-lite: claim a name, receive a token. Token authorizes future
    uploads under that name; no email, no password reset — alpha tier."""
    import hashlib
    body = await request.json()
    name = re.sub(r"[^a-zA-Z0-9_-]", "", str(body.get("name", "")))[:20]
    if len(name) < 3:
        return JSONResponse({"error": "name: 3-20 chars a-zA-Z0-9_-"}, status_code=400)
    users = json.loads(USERS_F.read_text()) if USERS_F.exists() else {}
    if name in users:
        return JSONResponse({"error": "name taken"}, status_code=409)
    token = secrets.token_hex(16)
    users[name] = {"token_sha": hashlib.sha256(token.encode()).hexdigest(),
                   "created": time.time()}
    USERS_F.write_text(json.dumps(users))
    return {"name": name, "token": token,
            "note": "keep the token; uploads with ?name=&token= are owned by you"}


def _auth(name: str, token: str) -> bool:
    import hashlib
    users = json.loads(USERS_F.read_text()) if USERS_F.exists() else {}
    u = users.get(name)
    return bool(u and u["token_sha"] == hashlib.sha256(token.encode()).hexdigest())


@app.get("/api/user/{name}")
async def api_user(name: str):
    """Per-user profile: their bots, ladder ratings, recent matches."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "", name)[:20]
    users = json.loads(USERS_F.read_text()) if USERS_F.exists() else {}
    if name not in users:
        return JSONResponse({"error": "no such user"}, status_code=404)
    owners = json.loads(OWNERS_F.read_text()) if OWNERS_F.exists() else {}
    my_bots = [f"user:{b}" for b, o in owners.items() if o == name]
    st_p = Path("/root/ludus_ladder_live/standings.json")
    rows = json.loads(st_p.read_text())["rows"] if st_p.exists() else []
    ratings = {r["bot"]: r for r in rows}
    return JSONResponse({
        "name": name,
        "since": users[name].get("created"),
        "bots": [{"id": b,
                  "rating": ratings.get(b, {}).get("rating"),
                  "games": ratings.get(b, {}).get("games", 0)}
                 for b in my_bots]})


@app.get("/u/{name}")
async def user_page(name: str):
    return FileResponse(STATIC / "user.html")


@app.get("/api/standings")
async def api_standings():
    p = Path("/root/ludus_ladder_live/standings.json")
    if not p.exists():
        return JSONResponse({"rows": [], "updated": None})
    return JSONResponse(json.loads(p.read_text()))


@app.post("/api/composer")
async def api_composer(request: Request):
    """Save a composed rule policy: {name, rules: [{sensor, op, value, slot, at}]}"""
    body = await request.json()
    name = re.sub(r"[^a-zA-Z0-9_-]", "", str(body.get("name", "")))[:24]
    rules = body.get("rules", [])
    if len(name) < 3 or not isinstance(rules, list) or not (1 <= len(rules) <= 12):
        return JSONResponse({"error": "name (3+) and 1-12 rules required"},
                            status_code=400)
    clean = []
    for r in rules:
        if r.get("sensor") not in graphs_mod.SENSORS:
            return JSONResponse({"error": f"bad sensor {r.get('sensor')}"},
                                status_code=400)
        slot = r.get("slot")
        if slot != "cheapest" and slot not in (0, 1, 2, 3):
            return JSONResponse({"error": "slot must be 0-3 or 'cheapest'"},
                                status_code=400)
        clean.append({"sensor": r["sensor"], "op": "<" if r.get("op") == "<" else ">",
                      "value": float(r.get("value", 0)), "slot": slot,
                      "at": r.get("at") if r.get("at") in _SPOTS else "king_front"})
    (GRAPHS_DIR / f"{name}.json").write_text(json.dumps({"rules": clean}))
    return {"ok": True, "bot": f"graph:{name}", "play": f"/play?bot=graph:{name}",
            "table": f"/play?a=graph:{name}&b=ppo"}


@app.get("/composer")
async def composer_page():
    return FileResponse(STATIC / "composer.html")


# ---------------------------------------------------------------- gomoku (game #2)
from gomoku import engine as gomoku_engine

GOMOKU_REPLAYS = Path("/root/ludus_replays_gomoku")
GOMOKU_REPLAYS.mkdir(parents=True, exist_ok=True)
_g_step = jax.jit(gomoku_engine.step)


def _np_wins_at(board, y, x, stone) -> bool:
    """Pure-numpy: would placing `stone` at (y,x) make a 5+ run?"""
    N = gomoku_engine.N
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run = 1
        for sgn in (1, -1):
            yy, xx = y + sgn * dy, x + sgn * dx
            while 0 <= yy < N and 0 <= xx < N and board[yy, xx] == stone:
                run += 1
                yy += sgn * dy
                xx += sgn * dx
        if run >= 5:
            return True
    return False


def _gomoku_bot_move(state, kind: str, key):
    """Server-side gomoku opponent: 'greedy' wins/blocks 5s, else random legal."""
    board = np.asarray(state.board)
    legal_idx = np.flatnonzero(board.reshape(-1) == 0)
    if len(legal_idx) == 0:
        return 0
    if kind == "greedy":
        me = int(state.to_move) + 1
        opp = 3 - me
        for stone in (me, opp):                    # win first, else block
            for a in legal_idx:
                y, x = divmod(int(a), gomoku_engine.N)
                if _np_wins_at(board, y, x, stone):
                    return int(a)
        occ = np.argwhere(board > 0)
        if len(occ):
            cy, cx = occ.mean(axis=0)
            d = [abs(a // 15 - cy) + abs(a % 15 - cx) for a in legal_idx]
            return int(legal_idx[int(np.argmin(d))])
    return int(np.asarray(jax.random.choice(key, jnp.asarray(legal_idx))))


@app.get("/gomoku")
async def gomoku_page():
    return FileResponse(STATIC / "gomoku.html")


async def run_gomoku(sock: WebSocket, bot: str, human_side: str = "black"):
    state = gomoku_engine.reset()
    moves: list[int] = []
    key = jax.random.PRNGKey(secrets.randbits(31))
    rapfi = None
    if bot == "strong":
        from arena import rapfi_engine
        if rapfi_engine.available():
            rapfi = rapfi_engine.RapfiGame()
        else:
            bot = "greedy"                     # honest fallback, shown in meta
    human_stone = 0 if human_side == "black" else 1
    await sock.send_text(json.dumps({"type": "meta", "game": "gomoku",
                                     "bot": ("rapfi (Gomocup engine)" if rapfi
                                             else bot), "you": human_side}))

    async def bot_reply(last_human_flat=None):
        nonlocal state, key
        if rapfi is not None:
            if last_human_flat is None:
                rx, ry = await off_loop(rapfi.begin)
            else:
                hx = last_human_flat % gomoku_engine.N
                hy = last_human_flat // gomoku_engine.N
                rx, ry = await off_loop(rapfi.opponent_move_and_reply, hx, hy)
            b = ry * gomoku_engine.N + rx
        else:
            key, k = jax.random.split(key)
            b = await off_loop(_gomoku_bot_move, state, bot, k)
        state = await off_loop(_g_step, state, jnp.int32(b))
        moves.append(int(b))

    def frame():
        return json.dumps({"type": "frame",
                           "board": np.asarray(state.board).tolist(),
                           "to_move": int(state.to_move),
                           "result": int(state.result)})
    if human_stone == 1:                      # bot is black: it opens
        await bot_reply(None)
    await sock.send_text(frame())
    while True:
        msg = json.loads(await sock.receive_text())
        if msg.get("type") == "restart":
            state = gomoku_engine.reset()
            moves = []
            if rapfi is not None:
                rapfi.close()
                from arena import rapfi_engine
                rapfi = rapfi_engine.RapfiGame() if rapfi_engine.available() else None
            await sock.send_text(frame())
            continue
        if msg.get("type") != "move" or int(state.result) != -1                 or int(state.to_move) != 0:
            continue
        a = int(msg["a"])
        if not bool(gomoku_engine.legal(state)[a]):
            await sock.send_text(json.dumps({"type": "reject"}))
            continue
        state = await off_loop(_g_step, state, jnp.int32(a))
        moves.append(a)
        await sock.send_text(frame())
        if int(state.result) == -1:
            await bot_reply(a)
            await sock.send_text(frame())
        if int(state.result) != -1:
            (GOMOKU_REPLAYS / f"g_{int(time.time())}.json").write_text(
                json.dumps({"env_version": gomoku_engine.ENV_VERSION,
                            "moves": moves, "result": int(state.result)}))


SMAX_REPLAYS = Path("/root/ludus_replays_smax")
SMAX_REPLAYS.mkdir(parents=True, exist_ok=True)
app.mount("/smax_replays", StaticFiles(directory=SMAX_REPLAYS), name="smax_replays")


@app.get("/smax")
async def smax_page():
    return FileResponse(STATIC / "smax.html")


@app.websocket("/ws_warpath_replay")
async def ws_warpath_replay(ws: WebSocket):
    await ws.accept()
    import asyncio as aio, json as _json
    import jax, jax.numpy as jnp, numpy as np
    from warpath import engine, render
    rid = ws.query_params.get("id", "")
    if not re.fullmatch(r"wp_\d+", rid):
        await ws.close(code=4400)
        return
    p = Path("/root/ludus_replays_warpath") / f"{rid}.json"
    if not p.exists():
        await ws.close(code=4404)
        return
    rec = _json.loads(p.read_text())
    loop = aio.get_event_loop()
    cpu = jax.devices("cpu")[0]
    step = jax.jit(engine.step, backend="cpu")
    speed = {"x": 1}

    async def recv():
        try:
            while True:
                m = await ws.receive_json()
                if m.get("type") == "speed":
                    speed["x"] = max(1, min(8, int(m.get("x", 1))))
        except Exception:
            return
    task = aio.create_task(recv())
    try:
        with jax.default_device(cpu):
            state = engine.reset()
        for a0s, a1s in rec["action_log"]:
            try:
                with jax.default_device(cpu):
                    acts = jnp.asarray(np.array([a0s, a1s], dtype=np.int32))
                    state = step(state, acts)
                frame = render.frame(state)
            except Exception as e:
                await ws.send_json({"type": "error", "msg": f"replay re-sim failed: {e}"})
                break
            await ws.send_json(frame)
            await aio.sleep(0.2 / speed["x"])
        while True:                                 # linger on the final frame
            await aio.sleep(1.0)
            await ws.send_json(render.frame(state))
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        task.cancel()


@app.get("/warpath")
async def warpath_page():
    return FileResponse(STATIC / "warpath.html")


@app.websocket("/ws_warpath")
async def ws_warpath(ws: WebSocket):
    await ws.accept()
    import asyncio as aio
    from arena.warpath_live import WarpathLive
    loop = aio.get_event_loop()
    sess = await loop.run_in_executor(None, WarpathLive)
    import time as _time
    TICK = 0.2
    next_t = _time.monotonic()
    try:
        while True:
            # drain commands until the next tick boundary (real-time cadence)
            while True:
                budget = next_t - _time.monotonic()
                if budget <= 0:
                    break
                try:
                    msg = await aio.wait_for(ws.receive_json(), timeout=budget)
                    if msg.get("type") == "cmd":
                        sess.command(msg)
                    elif msg.get("type") == "restart":
                        sess = await loop.run_in_executor(None, WarpathLive)
                        next_t = _time.monotonic()
                except aio.TimeoutError:
                    break
            next_t = max(next_t + TICK, _time.monotonic() - TICK)
            frame = await loop.run_in_executor(None, sess.tick)
            await ws.send_json(frame)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws_smax")
async def ws_smax(ws: WebSocket):
    await ws.accept()
    import asyncio as aio
    from arena.smax_live import SmaxLive
    import random as _rnd
    map_name = ws.query_params.get("map", "2s3z")
    loop = aio.get_event_loop()
    sess = await loop.run_in_executor(None, lambda: SmaxLive(map_name, _rnd.randrange(10**6)))
    await ws.send_json({"type": "hello", "map": map_name,
                        "n_allies": sess.n_allies, **sess.frame()})
    try:
        while True:
            # drain any pending commands, then advance one tick (~3.3 Hz)
            try:
                while True:
                    msg = await aio.wait_for(ws.receive_json(), timeout=0.30)
                    if msg.get("type") == "order":
                        sess.set_order(msg.get("ids", []), msg.get("kind", "stop"),
                                       msg.get("x"), msg.get("y"), msg.get("target"))
                    elif msg.get("type") == "restart":
                        sess = await loop.run_in_executor(
                            None, lambda: SmaxLive(map_name, _rnd.randrange(10**6)))
            except aio.TimeoutError:
                pass
            frame = await loop.run_in_executor(None, sess.tick)
            await ws.send_json(frame)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.get("/api/smax/board")
async def smax_board():
    """Per-scenario win-rate board from training-run artifacts on this box."""
    rows = []
    for d in sorted(Path("/root/ludus_train").glob("smax_mappo*")):
        f = d / "winrate_curve.json"
        if not f.exists():
            continue
        try:
            j = json.loads(f.read_text())
            rows.append({"scenario": j.get("map"), "algo": j.get("algo"),
                         "win_rate": j.get("final_win_rate_last10"),
                         "timesteps": j.get("total_timesteps"),
                         "minutes": round(j.get("duration_s", 0) / 60, 1)})
        except Exception:
            continue
    return JSONResponse(rows)


@app.get("/api/smax/replays")
async def smax_replays():
    eps = sorted((p.name for p in SMAX_REPLAYS.glob("*.json")), reverse=True)
    return JSONResponse(list(eps)[:50])


MAHJONG_REPLAYS = Path("/root/ludus_replays_mahjong")
MAHJONG_REPLAYS.mkdir(parents=True, exist_ok=True)
MAHJONG_KEEP = 300     # cap the archive: prune oldest beyond this many files


def _mahjong_save_replay(seed: int, quan: int, bot: str, result: dict):
    """Persist one generated game once, with a stable id derived from seed+bot
    so re-requests overwrite (never pile up duplicates). Prunes the oldest
    files past MAHJONG_KEEP. Never raises into the play path."""
    try:
        rid = "mj_%s_%d_%d" % (re.sub(r"[^a-z0-9]", "", bot), quan, seed)
        rec = {
            "id": rid,
            "ts": int(time.time()),
            "seed": seed,
            "quan": quan,
            "bot": bot,
            "scores": result.get("scores"),
            "winner": result.get("winner"),
            "fan": result.get("fan"),
            "fan_list": result.get("fan_list"),
            "ending": result.get("ending"),
            "frames": result.get("frames"),
        }
        (MAHJONG_REPLAYS / (rid + ".json")).write_text(json.dumps(rec))
        files = sorted(MAHJONG_REPLAYS.glob("mj_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[MAHJONG_KEEP:]:
            try:
                p.unlink()
            except Exception:
                pass
        return rid
    except Exception:
        return None


@app.get("/api/mahjong/replays")
async def mahjong_replays():
    """Saved Mahjong games, newest-first: id, ts, seed, quan, bot, ending,
    winner, fan (no frames — the browse list stays light)."""
    items = []
    for p in MAHJONG_REPLAYS.glob("mj_*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        items.append({"id": d.get("id", p.stem), "ts": d.get("ts"),
                      "seed": d.get("seed"), "quan": d.get("quan"),
                      "bot": d.get("bot"), "ending": d.get("ending"),
                      "winner": d.get("winner"), "fan": d.get("fan")})
    items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    return JSONResponse(items[:MAHJONG_KEEP])


@app.get("/api/mahjong/replay/{rid}")
async def mahjong_replay(rid: str):
    """Full stored replay (frames + result) for one archived game."""
    if not re.fullmatch(r"mj_[a-z0-9]+_\d+_\d+", rid):
        return JSONResponse({"error": "bad id"}, status_code=400)
    p = MAHJONG_REPLAYS / (rid + ".json")
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="application/json")


@app.get("/mahjong")
async def mahjong_page():
    return FileResponse(STATIC / "mahjong.html")


def _mahjong_run(seed: int, quan: int, bot: str):
    """Run one MCR game and return the viewer response dict. Runs in a worker
    thread (off the event loop) because the champion game takes ~1 min."""
    from mahjong import bots, engine, match      # lazy import; keeps startup fast
    if bot in ("champion", "hard"):
        # kdens3 champion (seat 0 / East) vs three EfficiencyBots. Driven through
        # the oracle-validated steppable engine so the champion never desyncs.
        # bot=champion -> the full 3-net kdens3 ensemble ("Ultimate").
        # bot=hard     -> a single net (kdens_s0_fp16.npz), a real weaker tier.
        from mahjong import champion
        npzs = "kdens_s0_fp16.npz" if bot == "hard" else None
        champ = champion.ChampionBot(npzs=npzs)
        agents = [champ, bots.EfficiencyBot(), bots.EfficiencyBot(),
                  bots.EfficiencyBot()]
        try:
            result = match.run_match(agents, engine.build_wall(seed=seed),
                                     quan=quan)
        finally:
            champ.close()
    else:
        Bot = bots.RandomLegalBot if bot == "random" else bots.EfficiencyBot
        # Route ALL bot matches through the validated engine for consistency.
        agents = [Bot() for _ in range(4)]
        result = match.run_match(agents, engine.build_wall(seed=seed), quan=quan)
    return {
        "frames": result["frames"],
        "scores": result["scores"],
        "ending": result["ending"],
        "winner": result["winner"],
        "fan": result["fan"],
        "fan_list": result.get("fan_list"),
        "bot": bot,
    }


@app.get("/api/mahjong/play")
async def mahjong_play(seed: int = 0, quan: int = 0, bot: str = "efficiency"):
    """Run one Chinese-Standard-Mahjong game and return the full replay
    (spectator/full-info frames). bot=efficiency (tidy play, takes >=8-fan
    wins), bot=random (draw-and-discard, games always exhaust), or
    bot=champion (kdens3 ensemble at seat 0 vs 3 EfficiencyBots — precomputes
    the whole game server-side, ~1 min), or bot=hard (a single kdens net at
    seat 0 — a real weaker tier than the full ensemble). All routed through the
    oracle-validated steppable engine (mahjong.match.run_match). Each generated
    game is persisted once to the replay archive (stable id from seed+bot)."""
    if bot not in ("efficiency", "random", "champion", "hard"):
        bot = "efficiency"
    try:
        # Real worker thread (not off_loop): pure-python/subprocess work with no
        # JAX, so a ~1-min champion game must not block the whole event loop.
        result = await asyncio.to_thread(_mahjong_run, seed, quan, bot)
        rid = await asyncio.to_thread(
            _mahjong_save_replay, seed, quan, bot, result)
        if rid:
            result = dict(result, id=rid)
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e),
                             "trace": traceback.format_exc()}, status_code=200)


# --------------------------------------------------------------------------
# Interactive human-seat Mahjong: human = seat 0 (East), EfficiencyBots 1-3,
# driven turn-by-turn through the oracle-validated steppable MyEngine.
# --------------------------------------------------------------------------
def _mj_fan(pack, concealed, win_tile, is_self, seat, quan):
    """Total fan via PyMahjongGB, or None if not a valid (>=0-fan) win."""
    try:
        from MahjongGB import MahjongFanCalculator
        r = MahjongFanCalculator(
            pack=tuple(pack), hand=tuple(concealed), winTile=win_tile,
            flowerCount=0, isSelfDrawn=is_self, is4thTile=False,
            isAboutKong=False, isWallLast=False, seatWind=seat,
            prevalentWind=quan)
        return sum(v for v, _ in r)
    except Exception:
        return None


def _mj_disp_last(disp):
    a = disp["action"]
    p = disp.get("player")
    if a == "DRAW":
        return "DRAW %d %s" % (p, disp["tile"])
    if a == "PLAY":
        return "3 %d PLAY %s" % (p, disp["tile"])
    if a == "PENG":
        return "3 %d PENG %s" % (p, disp["tile"])
    if a == "CHI":
        return "3 %d CHI %s %s" % (p, disp["tileCHI"], disp["tile"])
    if a == "GANG":
        return "3 %d GANG" % p
    if a == "BUGANG":
        return "3 %d BUGANG %s" % (p, disp["tile"])
    if a == "HU":
        return "HU %d ? %s fan=%s" % (p, "hu", disp.get("fanCnt"))
    return ""


def _mj_frame_state(eng, last, result=None):
    """Board frame for the interactive viewer: seat 0's hand is visible; the
    opponents' concealed hands are hidden (rendered as backs)."""
    hands = []
    for s in range(4):
        if s == 0:
            hands.append(sorted(eng.hands[0]))
        else:
            hands.append(["?"] * len(eng.hands[s]))
    fr = {"type": "frame",
          "hands": hands,
          "melds": [[list(m) for m in eng.packs[s]] for s in range(4)],
          "discards": [list(d) for d in eng.discards],
          "wall_left": list(eng.remain),
          "last": last, "result": result}
    return json.dumps(fr)


def _mj_draw_options(eng, drawn, quan):
    hand = eng.hands[0]
    pack = tuple((m[0], m[1], m[2]) for m in eng.packs[0])
    concealed = list(hand)
    concealed.remove(drawn)
    f = _mj_fan(pack, concealed, drawn, True, 0, quan)
    hu = f is not None and f >= 8
    gang = sorted({t for t in set(hand) if hand.count(t) == 4})
    bugang = sorted({m[1] for m in eng.packs[0]
                     if m[0] == "PENG" and m[1] in hand})
    return {"hu": hu, "gang": list(gang), "bugang": list(bugang)}


def _mj_claim_options(eng, disc_seat, tile, quan):
    hand = eng.hands[0]
    pack = tuple((m[0], m[1], m[2]) for m in eng.packs[0])
    f = _mj_fan(pack, list(hand), tile, False, 0, quan)
    hu = f is not None and f >= 8
    peng = hand.count(tile) >= 2
    gang = hand.count(tile) == 3
    chi = []
    if disc_seat == 3 and tile[0] in "WTB":     # only the left neighbour may chi
        s, n = tile[0], int(tile[1])
        for mid_n in (n - 1, n, n + 1):
            if 2 <= mid_n <= 8:
                need = ["%s%d" % (s, x) for x in (mid_n - 1, mid_n, mid_n + 1)
                        if x != n]
                if all(x in hand for x in need):
                    chi.append("%s%d" % (s, mid_n))
    return {"hu": hu, "peng": peng, "gang": gang, "chi": chi}


async def _mj_human_response(sock, eng, hreq, quan):
    """Produce seat 0's Botzone response, prompting the browser when the human
    actually has a decision, else auto-PASS on informational broadcasts."""
    parts = hreq.split()
    code = parts[0]
    if code == "2":                                    # own draw: must act
        drawn = parts[1]
        opts = _mj_draw_options(eng, drawn, quan)
        await sock.send_text(json.dumps({
            "type": "decide", "kind": "draw", "hand": sorted(eng.hands[0]),
            "drawn": drawn, "options": opts}))
        while True:
            msg = json.loads(await sock.receive_text())
            t = msg.get("type")
            if t == "discard" and msg.get("tile") in eng.hands[0]:
                return "PLAY " + msg["tile"]
            if t == "selfhu" and opts["hu"]:
                return "HU"
            if t == "gang" and msg.get("tile") in opts["gang"]:
                return "GANG " + msg["tile"]
            if t == "bugang" and msg.get("tile") in opts["bugang"]:
                return "BUGANG " + msg["tile"]
            # ignore anything illegal and keep waiting
    if (code == "3" and len(parts) >= 4 and parts[2] == "PLAY"
            and int(parts[1]) != 0):               # a claimable opponent discard
        disc_seat, tile = int(parts[1]), parts[3]
        opts = _mj_claim_options(eng, disc_seat, tile, quan)
        if not (opts["hu"] or opts["peng"] or opts["gang"] or opts["chi"]):
            return "PASS"
        await sock.send_text(json.dumps({
            "type": "decide", "kind": "claim", "hand": sorted(eng.hands[0]),
            "tile": tile, "from": disc_seat, "options": opts}))
        while True:
            msg = json.loads(await sock.receive_text())
            a = msg.get("action")
            if a == "pass":
                return "PASS"
            if a == "hu" and opts["hu"]:
                return "HU"
            if a == "gang" and opts["gang"]:
                return "GANG"
            if a == "peng" and opts["peng"]:
                rem = list(eng.hands[0])
                rem.remove(tile)
                rem.remove(tile)
                if msg.get("discard") in rem:
                    return "PENG " + msg["discard"]
            if a == "chi" and msg.get("mid") in opts["chi"]:
                mid = msg["mid"]
                s, n = mid[0], int(mid[1])
                rem = list(eng.hands[0])
                for x in ("%s%d" % (s, n - 1), "%s%d" % (s, n),
                          "%s%d" % (s, n + 1)):
                    if x != tile:
                        rem.remove(x)
                if msg.get("discard") in rem:
                    return "CHI %s %s" % (mid, msg["discard"])
            # ignore illegal, keep waiting
    return "PASS"                                      # informational broadcast


def _mj_result_from_disp(disp):
    if disp["action"] == "HUANG":
        return {"ending": "draw", "winner": None, "fan": None,
                "scores": {i: 0 for i in range(4)}}
    if disp["action"] == "HU":
        sc = disp.get("score") or [0, 0, 0, 0]
        from mahjong import fan_glossary
        fan_list = [{"cn": f["name"], "en": fan_glossary.en_for(f["name"]),
                     "value": f["value"], "cnt": f["cnt"],
                     "desc": fan_glossary.desc_for(cn=f["name"])}
                    for f in (disp.get("fan") or [])]
        return {"ending": "hu", "winner": disp["player"],
                "fan": disp.get("fanCnt"), "fan_list": fan_list,
                "scores": {i: sc[i] for i in range(4)}}
    return None


async def run_mahjong_interactive(sock: "WebSocket"):
    from mahjong import bots as mjbots, engine as mjeng
    from mahjong.validate_adapter import MyEngine
    import random as _rnd
    quan = 0
    first = True
    while True:                                        # one iteration per game
        q = sock.query_params.get("seed")
        seed = int(q) if (first and q) else _rnd.randint(0, 1_000_000)
        first = False
        wall = mjeng.build_wall(seed)
        eng = MyEngine()
        botz = {s: mjbots.EfficiencyBot() for s in (1, 2, 3)}
        await sock.send_text(json.dumps({"type": "meta", "you": 0, "seed": seed,
                                         "bots": "EfficiencyBot"}))
        turn = eng.reset(wall, quan, 0)
        result = None
        while True:
            disp = turn["display"]
            reqs = turn["requests"]
            last = _mj_disp_last(disp)
            if not reqs:                               # terminal turn
                result = _mj_result_from_disp(disp)
                await sock.send_text(_mj_frame_state(eng, last, result))
                break
            if disp["action"] not in ("INIT", "DEAL"):
                await sock.send_text(_mj_frame_state(eng, last))
                await asyncio.sleep(0.35)              # watchable pacing
            responses = {}
            for ss in (1, 2, 3):
                req = reqs.get(str(ss))
                if req is not None:
                    responses[str(ss)] = botz[ss].respond(req)
            hreq = reqs.get("0")
            if hreq is not None:
                responses["0"] = await _mj_human_response(sock, eng, hreq, quan)
            turn = eng.step(responses)
        # wait for a restart request (or disconnect)
        msg = json.loads(await sock.receive_text())
        if msg.get("type") != "restart":
            return


@app.websocket("/ws_mahjong")
async def ws_mahjong(sock: WebSocket):
    await sock.accept()
    try:
        await run_mahjong_interactive(sock)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/kanban")
async def api_kanban():
    data = json.loads((Path(__file__).parents[1] / "docs" / "kanban.json")
                      .read_text())
    try:
        log = subprocess.run(["git", "log", "--pretty=%ad %s", "--date=short",
                              "-20"], cwd=Path(__file__).parents[1],
                             capture_output=True, text=True).stdout
        data["changelog"] = [ln for ln in log.split("\n") if ln.strip()]
    except Exception:
        data["changelog"] = []
    return JSONResponse(data)


@app.get("/kanban")
async def kanban_page():
    return FileResponse(STATIC / "kanban.html")


@app.get("/api/bots")
async def api_bots():
    """Every opponent the platform can field right now."""
    bots = [{"id": "ppo", "name": "league champion", "kind": "league"},
            {"id": "rule", "name": "rule_v0", "kind": "builtin"},
            {"id": "random", "name": "random_v0", "kind": "builtin"}]
    for p in sorted(LEAGUE_DIR.glob("gen_*.msgpack")):
        g = p.stem.split("_")[1]
        bots.append({"id": f"gen:{g}", "name": f"league gen {g}", "kind": "league"})
    for p in sorted(BOTS_DIR.glob("*.msgpack")):
        bots.append({"id": f"user:{p.stem}", "name": p.stem, "kind": "user"})
    for p in sorted(GRAPHS_DIR.glob("*.json")):
        bots.append({"id": f"graph:{p.stem}", "name": f"{p.stem} (composed)",
                     "kind": "composed"})
    return JSONResponse(bots)


@app.post("/api/submit")
async def api_submit(request: "Request"):
    """Weights-only bot submission: multipart (name, file=msgpack for the published
    ActorCritic architecture). Validated by actually loading it."""
    form = await request.form()
    name = re.sub(r"[^a-zA-Z0-9_-]", "", str(form.get("name", "")))[:24]
    up = form.get("file")
    if len(name) < 3 or up is None:
        return JSONResponse({"error": "name (3-24 chars a-zA-Z0-9_-) and file required"},
                            status_code=400)
    blob = await up.read()
    if len(blob) > 30_000_000:
        return JSONResponse({"error": "file too large"}, status_code=400)
    try:                                            # arch check = a real load
        from flax.serialization import from_bytes

        def _check():
            infra = _ppo_infra()
            from_bytes(infra["tmpl"], blob)
        await off_loop(_check)
    except Exception as ex:
        return JSONResponse({"error": f"not a valid ActorCritic checkpoint: {ex}"},
                            status_code=400)
    (BOTS_DIR / f"{name}.msgpack").write_bytes(blob)
    owner, tok = str(form.get("owner", "")), str(form.get("token", ""))
    if owner and _auth(owner, tok):
        owners = json.loads(OWNERS_F.read_text()) if OWNERS_F.exists() else {}
        owners[name] = owner
        OWNERS_F.write_text(json.dumps(owners))
    import hashlib
    return {"ok": True, "bot": f"user:{name}",
            "sha256": hashlib.sha256(blob).hexdigest()[:16],
            "play": f"/play?bot=user:{name}", "table": f"/play?a=user:{name}&b=ppo"}


@app.get("/submit")
async def submit_page():
    return FileResponse(STATIC / "submit.html")


@app.get("/api/league")
async def api_league():
    st = LEAGUE_DIR / "standings.json"
    if not st.exists():
        return JSONResponse({"generations": 0, "lineage": []})
    return JSONResponse(json.loads(st.read_text()))


@app.get("/league")
async def league_page():
    return FileResponse(STATIC / "league.html")


# ------------------------------------------------------------- training monitor
# Registry of the RL/league runs the platform can surface live. Each reads a
# JSONL append-log; the monitor auto-parses whatever numeric fields it names.
TRAIN_RUNS = [
    {"id": "mahjong_t2_jax",
     "title": "Mahjong · T2 — fused on-GPU JAX KL-leashed PPO",
     "game": "Mahjong", "kind": "RL fine-tune (self-play, KL≤0.05 leash)",
     "path": "/root/ludus_train/mahjong_t2_jax/seed1_jax_results.jsonl",
     "live": "/root/ludus_train/mahjong_t2_jax/latest.msgpack",
     "sentinel": "/root/T2_JAX_PAUSED", "x": "env_steps", "xlabel": "env steps",
     "series": [("mean_score", "mean score (÷8)", "num"),
                ("win_rate", "win rate", "pct"),
                ("dealin_rate", "deal-in rate", "pct"),
                ("KL", "KL vs SL anchor", "num"),
                ("steps_s", "env steps / s", "num")]},
    {"id": "mahjong_t2_jax_strength",
     "title": "Mahjong · T2-JAX — strength eval (RL vs SL anchor + pool)",
     "game": "Mahjong", "kind": "strength eval (seat-0 vs frozen field)",
     "path": "/root/ludus_train/mahjong_t2_jax/strength_eval.jsonl",
     "live": "/root/ludus_train/mahjong_t2_jax/strength_eval.jsonl",
     "sentinel": "/root/T2_JAX_EVAL_PAUSED", "x": "env_steps", "xlabel": "env steps",
     "series": [("vs_anchor_winrate", "win vs SL anchor", "pct"),
                ("anchor_self_winrate", "SL-anchor self baseline", "pct"),
                ("vs_efficiency_winrate", "win vs EfficiencyBot", "pct"),
                ("vs_random_winrate", "win vs random-legal", "pct"),
                ("vs_kdens3_winrate", "win vs kdens3 (small n)", "pct"),
                ("vs_anchor_score", "mean score vs anchor (÷8)", "num")]},
    {"id": "mahjong_t2_torch",
     "title": "Mahjong · T2 — torch KL-PPO (superseded by the JAX run)",
     "game": "Mahjong", "kind": "RL fine-tune (paused)",
     "path": "/root/ludus_train/mahjong_t2/seed1_results.jsonl",
     "sentinel": "/root/T2_PAUSED", "x": "env_steps", "xlabel": "env steps",
     "series": [("mean_score", "mean score (÷8)", "num"),
                ("win_rate", "win rate", "pct"),
                ("KL", "KL vs SL anchor", "num")]},
    {"id": "boom_league",
     "title": "Boom · generational self-play league",
     "game": "Boom", "kind": "league (CI-gated promotion, paused)",
     "path": "/root/ludus_train/league/league.jsonl",
     "sentinel": "/root/LEAGUE_PAUSED", "x": "gen", "xlabel": "generation",
     "filter_event": "gen_done",
     "series": [("vs_random", "vs random_v0", "pct"),
                ("vs_rule", "vs rule_v0", "pct"),
                ("vs_champion", "vs prev champion", "pct")]},
]
_LIVE_WINDOW = 900        # a run counts as "live" if it logged in the last 15 min


def _read_train_run(run):
    p = Path(run["path"])
    if not p.exists():
        return None
    try:
        raw = p.read_text().splitlines()
    except Exception:
        return None
    keys = [s[0] for s in run["series"]]
    xkey = run["x"]
    rows = []
    for ln in raw:
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if run.get("filter_event") and d.get("event") != run["filter_event"]:
            continue
        rows.append(d)
    if not rows:
        return None
    pts = []
    for i, d in enumerate(rows):
        x = d.get(xkey, i)
        rec = {"x": x if isinstance(x, (int, float)) else i}
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)):
                rec[k] = v
        pts.append(rec)
    MAX = 240                                    # downsample, always keep the last
    if len(pts) > MAX:
        stepf = len(pts) / MAX
        idx = sorted(set([int(i * stepf) for i in range(MAX)] + [len(pts) - 1]))
        pts = [pts[i] for i in idx]
    # liveness = freshest of the log and (if any) the checkpoint it writes
    ago = time.time() - p.stat().st_mtime
    lv = run.get("live")
    if lv and Path(lv).exists():
        ago = min(ago, time.time() - Path(lv).stat().st_mtime)
    paused = bool(run.get("sentinel") and Path(run["sentinel"]).exists())
    live = ago < _LIVE_WINDOW
    status = "live" if (live and not paused) else ("paused" if paused else "idle")
    latest = rows[-1]
    return {"id": run["id"], "title": run["title"], "game": run["game"],
            "kind": run["kind"], "status": status, "updated_ago_s": int(max(0, ago)),
            "n": len(rows), "x": xkey, "xlabel": run["xlabel"],
            "series": [{"key": k, "label": lab, "fmt": fmt}
                       for (k, lab, fmt) in run["series"]],
            "latest": {k: latest.get(k) for k in [xkey] + keys
                       if isinstance(latest.get(k), (int, float))},
            "points": pts}


@app.get("/api/train")
async def api_train():
    runs = []
    for run in TRAIN_RUNS:
        try:
            r = _read_train_run(run)
        except Exception:
            r = None
        if r:
            runs.append(r)
    order = {"live": 0, "idle": 1, "paused": 2}
    runs.sort(key=lambda r: (order.get(r["status"], 3), r["game"]))
    return JSONResponse({"runs": runs, "server_time": time.time()})


@app.get("/train")
async def train_page():
    return FileResponse(STATIC / "train.html")


@app.get("/wiki")
async def wiki_index():
    return FileResponse(STATIC / "wiki" / "index.html")


@app.get("/wiki/{slug}")
async def wiki_page(slug: str):
    p = STATIC / "wiki" / f"{re.sub(r'[^a-zA-Z0-9_-]', '', slug)}.html"
    return FileResponse(p if p.exists() else STATIC / "wiki" / "index.html")


@app.get("/blog")
async def blog_index():
    return FileResponse(STATIC / "blog" / "index.html")


@app.get("/challenge")
async def challenge_page():
    return FileResponse(STATIC / "blog" / "2026-07-04-the-boom-agent-challenge.html")


@app.get("/blog/{slug}")
async def blog_post(slug: str):
    p = STATIC / "blog" / f"{re.sub(r'[^a-zA-Z0-9_-]', '', slug)}.html"
    return FileResponse(p if p.exists() else STATIC / "blog" / "index.html")


@app.get("/api/replays")
async def api_replays(game: str = "boom"):
    """Full per-game replay history. Old-engine records stay listed (playable
    marks whether re-simulation is bit-faithful on the CURRENT engine)."""
    items = []
    if game == "boom":
        for p in sorted(REPLAY_DIR.glob("match_*.json"), reverse=True)[:300]:
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            items.append({"id": p.stem, "ticks": d["ticks"], "outcome": d["outcome"],
                          "seed": d["seed"], "env_version": d.get("env_version"),
                          "ts": int(p.stem.split("_")[1]),
                          "playable": d.get("env_version") == ENV_VERSION})
    elif game == "gomoku":
        for p in sorted(GOMOKU_REPLAYS.glob("g_*.json"), reverse=True)[:300]:
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            items.append({"id": p.stem, "ts": int(p.stem.split("_")[1]),
                          "env_version": d.get("env_version"),
                          "moves": len(d.get("moves", [])),
                          "winner": d.get("winner"), "playable": False})
    elif game == "warpath":
        WP = Path("/root/ludus_replays_warpath")
        for p in sorted(WP.glob("wp_*.json"), reverse=True)[:300] if WP.exists() else []:
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            items.append({"id": p.stem, "ts": int(p.stem.split("_")[1]),
                          "env_version": d.get("env_version"),
                          "ticks": d.get("ticks"), "result": d.get("result"),
                          "playable": d.get("env_version") == "warpath/v1"})
    elif game == "smax":
        for p in sorted(SMAX_REPLAYS.glob("smax_*.json"), reverse=True)[:300]:
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            items.append({"id": p.stem, "map": d.get("map"),
                          "ts": int(p.stem.split("_")[-1]),
                          "ticks": d.get("ticks"), "won": d.get("ally_won"),
                          "playable": True})
    return JSONResponse(items)


@app.get("/replay_raw/{game}/{rid}")
async def replay_raw(game: str, rid: str):
    """Export: the raw deterministic replay record."""
    dirs = {"boom": (REPLAY_DIR, r"match_\d+_\d+"),
            "gomoku": (GOMOKU_REPLAYS, r"g_\d+"),
            "smax": (SMAX_REPLAYS, r"smax_[a-z0-9_]+"),
            "warpath": (Path("/root/ludus_replays_warpath"), r"wp_\d+")}
    if game not in dirs or not re.fullmatch(dirs[game][1], rid):
        return JSONResponse({"error": "bad id"}, status_code=400)
    p = dirs[game][0] / f"{rid}.json"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="application/json",
                        filename=f"{rid}.json")


# ---------------------------------------------------------------- live match
class Match:
    def __init__(self, seed: int, bot: str = "random", decks=None):
        self.seed = seed
        self.bot_kind = bot
        self.decks = decks
        self.state = None                     # born lazily ON the jax thread
        self.bot, self.bot_name = make_bot(bot, allow_cold=False)
        self.log: list[list[list[int]]] = []
        self.pending: list[int] | None = None

    def human_action_legal(self, slot: int, x: int, y: int) -> bool:
        """Must be called via off_loop — touches jax state."""
        if self.state is None:
            return False
        if not (0 <= slot < 4 and 0 <= x < engine.W and 0 <= y < engine.H):
            return False
        return bool(np.asarray(_legal(self.state, 0))[slot, x, y])

    def tick(self) -> dict:
        if self.state is None:                # first call runs on the jax thread
            self.state = _reset(jax.random.PRNGKey(self.seed), self.decks)
            self.bot_key = jax.random.PRNGKey(self.seed ^ 0x5A5A5A)
        a0 = self.pending if self.pending else NOOP
        self.pending = None
        self.bot_key, k = jax.random.split(self.bot_key)
        a1 = self.bot(k, self.state, int(self.state.tick))
        self.state, frame = apply_tick(self.state, a0, a1)
        self.log.append([a0, a1])
        return frame

    def save_replay(self, outcome: int):
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        path = REPLAY_DIR / f"match_{int(time.time())}_{self.seed}.json"
        path.write_text(json.dumps({
            "env_version": ENV_VERSION, "seed": self.seed,
            "decks": None if self.decks is None else np.asarray(self.decks).tolist(),
            "action_log": self.log, "outcome": outcome, "ticks": len(self.log),
        }))


async def run_live(sock: WebSocket, bot: str, decks=None):
    match = Match(secrets.randbits(31), bot, decks)
    await sock.send_text(json.dumps({"type": "meta", "bot": match.bot_name,
                                     "custom_deck": decks is not None}))

    async def recv_loop():
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("type") == "play":
                s, x, y = int(msg["slot"]), int(msg["x"]), int(msg["y"])
                if await off_loop(match.human_action_legal, s, x, y):
                    match.pending = [s, x, y]
                else:
                    await sock.send_text(json.dumps({"type": "reject", "slot": s}))
            elif msg.get("type") == "restart":
                match.__init__(secrets.randbits(31), bot, decks)

    recv = asyncio.create_task(recv_loop())
    try:
        next_t = time.monotonic()
        while True:
            next_t = max(next_t + TICK_S, time.monotonic() - TICK_S)  # cap catch-up
            frame = await off_loop(match.tick)
            frame["hand"] = await off_loop(hand_info, match.state, 0)
            await sock.send_text(json.dumps(frame))
            if frame["result"] != -1:
                match.save_replay(frame["result"])
                while int(engine.result(match.state)) != -1:
                    await asyncio.sleep(0.2)
                next_t = time.monotonic()
                continue
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        recv.cancel()


# ---------------------------------------------------------------- replay watch
async def run_watch(sock: WebSocket, rid: str):
    if not re.fullmatch(r"match_\d+_\d+", rid):
        await sock.close(code=4400)
        return
    path = REPLAY_DIR / f"{rid}.json"
    if not path.exists():
        await sock.close(code=4404)
        return
    rec = json.loads(path.read_text())
    ctl = {"speed": 1, "restart": False}

    async def recv_loop():
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("type") == "speed":
                ctl["speed"] = max(1, min(8, int(msg["x"])))
            elif msg.get("type") == "restart":
                ctl["restart"] = True

    recv = asyncio.create_task(recv_loop())
    try:
        while True:                       # outer loop supports "watch again"
            def _replay_reset():
                rec_decks = None if rec.get("decks") is None \
                    else jnp.asarray(rec["decks"], jnp.int32)
                return _reset(jax.random.PRNGKey(rec["seed"]), rec_decks)
            state = await off_loop(_replay_reset)
            next_t = time.monotonic()
            for a0, a1 in rec["action_log"]:
                if ctl["restart"]:
                    break
                next_t = max(next_t + TICK_S / ctl["speed"],
                             time.monotonic() - TICK_S)
                state, frame = await off_loop(apply_tick, state, a0, a1)
                # replays show BOTH players' hands + elixir (spectator's view)
                frame["hand"] = await off_loop(hand_info, state, 0)
                frame["hand2"] = await off_loop(hand_info, state, 1)
                await sock.send_text(json.dumps(frame))
                if frame["result"] != -1:
                    break
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            while not ctl["restart"]:
                await asyncio.sleep(0.2)
            ctl["restart"] = False
    finally:
        recv.cancel()


async def run_table(sock: WebSocket, a: str, b: str):
    """Spectate table: two bots play live at 5 ticks/s (Botzone 'game table')."""
    bot_a, name_a = make_bot(a, allow_cold=False, seat=0)
    bot_b, name_b = make_bot(b, allow_cold=False, seat=1)
    await sock.send_text(json.dumps({"type": "meta",
                                     "bot": f"{name_a} vs {name_b}", "table": True}))
    ctl = {"speed": 1, "restart": False}

    async def recv_loop():
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("type") == "speed":
                ctl["speed"] = max(1, min(8, int(msg["x"])))
            elif msg.get("type") == "restart":
                ctl["restart"] = True

    recv = asyncio.create_task(recv_loop())
    try:
        while True:
            seed = secrets.randbits(31)
            rng = {}
            state = await off_loop(lambda: _reset(jax.random.PRNGKey(seed), None))
            next_t = time.monotonic()
            while not ctl["restart"]:
                next_t = max(next_t + TICK_S / ctl["speed"],
                             time.monotonic() - TICK_S)

                def _full_tick(state=state):
                    if "key" not in rng:
                        rng["key"] = jax.random.PRNGKey(seed ^ 0x0F0F0F)
                    rng["key"], ka, kb = jax.random.split(rng["key"], 3)
                    t = int(state.tick)
                    return apply_tick(state, bot_a(ka, state, t),
                                      bot_b(kb, state, t))
                state, frame = await off_loop(_full_tick)
                await sock.send_text(json.dumps(frame))
                if frame["result"] != -1:
                    break
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            while not ctl["restart"]:
                await asyncio.sleep(0.2)
            ctl["restart"] = False
    finally:
        recv.cancel()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    watch = sock.query_params.get("watch")
    a = sock.query_params.get("a")
    b = sock.query_params.get("b")
    table = sock.query_params.get("table")
    bot = sock.query_params.get("bot", "ppo")
    decks = parse_deck(sock.query_params.get("deck"))
    try:
        if sock.query_params.get("game") == "gomoku":
            side = sock.query_params.get("side", "black")
            await run_gomoku(sock, sock.query_params.get("bot", "strong"),
                             "white" if side == "white" else "black")
        elif table:
            await run_table_seat(sock, table, sock.query_params.get("seat", "spec"))
        elif watch:
            await run_watch(sock, watch)
        elif a and b:
            await run_table(sock, a, b)
        else:
            await run_live(sock, bot, decks)
    except (WebSocketDisconnect, RuntimeError):
        pass


_warmup()

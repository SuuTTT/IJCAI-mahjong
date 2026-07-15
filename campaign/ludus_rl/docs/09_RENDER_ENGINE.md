# 09 · Ludus Render Engine: one scene schema, every game

Design for a consistent, extensible rendering system covering the full game
roadmap — current (Boom, Gomoku) through planned (SMAX, Honor-of-Kings-class
MOBA via hok_env/hokoff, RA2-class RTS, Yu-Gi-Oh, Mahjong, Hold'em) to
long-horizon 3D (Minecraft-class, GTA-class, Mount&Blade-class via OpenMB).

## 1. Taxonomy: what these games actually share

| family | games | world | camera | core visual verbs |
|---|---|---|---|---|
| Plane battlers | Boom, SMAX, HoK 1v1/5v5, RA2, M&B battles | 2D plane (3D dressing) | top-down / iso / CR-tilt | move, strike, **projectile**, area-effect, spawn/die, health |
| Board & token | Gomoku, Mahjong, Hold'em, Yu-Gi-Oh | discrete zones | fixed ortho | token appears/moves between zones, reveal/hide, counters |
| Embodied 3D | Minecraft, GTA, M&B world | volumetric | first/third person | locomotion, interaction, terrain edit |

The user's instinct is correct and generalizes: **almost everything is
entities acting on entities, mostly via projectiles or contact, on a mostly-
planar stage**. Even card games fit: a card is an entity whose "movement" is
zone-to-zone and whose "attack" is an effect arc. Only embodied 3D breaks the
plane assumption — and even there, *replay spectating* (our need) is mostly a
chase camera over entities on terrain.

Consequence: one **scene schema** can drive every family; what varies per
game is (a) the stage (board/terrain), (b) the asset set, (c) the camera
profile — all data, not code.

## 2. The architecture: simulation → scene → backends

```
game engine (JAX, hok_env, pgx, ...)        [determinism lives here]
        │  per-game ADAPTER (pure function: state/replay -> SceneFrame)
        ▼
   SCENE SCHEMA (versioned JSON; the contract)
        │
        ├── live web renderer      (three.js; PixiJS fallback for pure-2D)
        ├── cinematic renderer     (same three.js scene, headless browser -> ffmpeg)
        └── debug renderer        (canvas dots — day-1 support for any new game)
```

The schema *is* the product (same philosophy as our replay format). Renderers
and games both program against it; N games + M backends = N adapters + M
backends, not N×M renderers. Killing the current split where Boom has a PixiJS
client AND a separate PIL cinema pipeline: one scene stream, three consumers.

## 3. Scene schema v0 (`render/SCHEMA.md` is normative)

```jsonc
{
  "v": 1, "game": "boom", "tick": 421, "clock": "1:24",
  "stage": {                       // static per match, sent once
    "kind": "plane",               // plane | zones | terrain
    "size": [18, 32],
    "layers": [{"type": "river", "rects": [...]}, {"type": "bridge", ...}],
    "camera": "cr_tilt"            // named profile: topdown | iso | cr_tilt | table
  },
  "entities": [                    // the universal unit of rendering
    {"id": 7, "arch": "unit",      // unit | building | token | card | avatar
     "asset": "boom/ramhound",     // registry key, NOT a file path
     "pos": [4.5, 19.5], "z": 0, "face": 0.75,
     "team": 0, "hp": [819, 1669], // -> auto health bar
     "anim": "run",                // idle | run | attack |死 handled by backend
     "status": ["charged"],        // -> badge/glow effects by name
     "scale": 1.0}
  ],
  "events": [                      // this-tick transients
    {"e": "projectile", "from": [4,19], "to": [3.5,25.5], "asset": "boom/bolt",
     "t_flight": 3, "arc": 0.5},
    {"e": "impact", "at": [3.5,25.5], "r": 1.5, "kind": "blast"},
    {"e": "damage", "at": [3.5,25.5], "amount": 109, "team": 1},
    {"e": "zone_move", "id": 12, "from": "hand:0", "to": "board:3,4"}  // card games
  ],
  "hud": {"resources": [[6.2, 10], [4.8, 10]], "hands": [...], "banners": []}
}
```

Design rules learned from Boom v1-v11:
- **Adapters are pure and versioned** with the engine (`env_version` rides
  along); a SceneFrame stream is derivable from any replay forever.
- **Assets by registry key**, never path — packs resolve keys (see §5).
- **Events, not diffs**: transient visuals (shots, impacts, damage numbers)
  are explicit events; renderers never infer them from state deltas (our
  early client did, and every new mechanic broke it).
- **Everything optional degrades**: a renderer that ignores `anim`/`status`
  still shows a correct game (the debug backend proves any adapter day-1).

## 4. Stack decision

Criterion #0 (the user named it): **an AI agent must be able to develop it** —
which means: everything is text, diffable, runnable headless, verifiable by
screenshot, no GUI-editor state.

| option | agent-dev fit | web delivery | 3D ceiling | verdict |
|---|---|---|---|---|
| Unity | poor: binary scenes, GUI editor workflows, license, heavy CI | via Wasm, heavy | very high | ❌ not for us (asset-store pipelines are its one draw) |
| Unreal | worse on all agent axes | poor | highest | ❌ |
| Godot 4 | good: text scenes, GDScript, headless CLI | Wasm ~30-40MB | high | 🟡 reserve for the embodied-3D tier (MC/GTA/M&B viewers) |
| **three.js** | **excellent: pure code, no editor, npm-free vendoring (MIT), screenshot-verifiable via headless Chrome** | native | MOBA/RTS-class easily | ✅ **primary backend** |
| PixiJS (have) | excellent | native | 2D only | ✅ keep for board/card family |
| own rasterizer | maximal control, maximal cost | — | — | ❌ the differentiator is the schema+adapters, not rasterization |

**Decision**: the "engine we build" is the **schema + adapter kit + asset
registry**; rasterization is three.js (primary) + PixiJS (2D) + headless-
Chrome-to-ffmpeg for cinematics (replacing PIL cinema — live and video become
the same code path, ending the render-drift class of bugs). Godot enters only
when an embodied-3D game does, consuming the same schema over websocket.

The verification loop that made the CR camera work — render one frame, LOOK
at it, then ship — becomes infrastructure: `scripts/scene_shot.py <replay>
<tick>` renders a PNG via headless Chrome for eyeball review in CI and in
agent sessions.

## 5. Asset system ("digital assets") design

**Registry**: `assets/<pack>/manifest.json` + files. Manifest per entry:
key (`boom/ramhound`), kind (sprite|model|audio|font), files (atlas PNG+JSON /
glTF), `license` (SPDX), `source` (URL/author), `palette_slots` (team-color
masks). The build packs atlases and emits a single `assets/index.json` the
renderers load. Resolution order: game pack → shared pack → procedural
placeholder (colored silhouette + emoji glyph, our current fallback,
promoted to a feature: every game renders before any art exists).

**Style: "Ludus look"** — one coherent original style so packs are reusable
across games: readable top-down silhouettes, flat-shaded low-poly for 3D,
team color via shader mask (one asset serves both sides), fixed palette
tokens (bg #0d1117 family, team blue/red as on the site), OFL fonts.

**Sourcing (license-clean by construction)**:
1. **CC0 packs**: Kenney (tower-defense, RTS, cards, UI — thousands of
   sprites), Quaternius (low-poly glTF characters/buildings) — consistent
   style, zero attribution burden, commercially safe.
2. **Procedural/self-made**: our drawn-icon pipeline; SVG-authored originals.
3. Community/commissioned originals later; AI-generated only with clear
   license provenance.
4. Never: ripped game assets (standing IP rule; drop-in private dirs stay
   git-ignored and local-only).

Manifests make license audits mechanical: `scripts/asset_audit.py` fails CI
on any entry without SPDX + source.

## 6. Per-game adapter sketches

- **Boom**: exists in spirit (render_frame/frame_events) — port to schema v1;
  the CR-tilt camera profile reproduces today's cinema look in three.js.
- **SMAX**: entities = marines/zealots (pos, hp, team), events = shots;
  topdown profile; the existing MAPPO checkpoint becomes watchable in days.
- **HoK (hok_env/hokoff)**: their obs/replay expose hero/minion/tower
  positions, hp, skills — adapter maps to plane+terrain stage, iso profile;
  skills = projectile/impact events. 1v1 first, exactly like Kaiwu.
- **RA2-class RTS**: iso profile; buildings as `building` entities; fog as a
  stage layer.
- **Yu-Gi-Oh / Hold'em / Mahjong**: `zones` stage (table profile); cards/tiles
  as `token` entities with `zone_move` events + reveal flags (public/private
  views per seat come free: the adapter filters by viewer seat).
- **Minecraft/GTA/M&B**: spectate-tier via chase-camera entity streams on a
  `terrain` stage (Godot backend when it matters). Not blocking anything.

## 7. Rollout plan

1. **P0 (now)**: schema v1 doc + `render/` module (SceneFrame builders,
   validation) + Boom adapter + vendored three.js viewer at `/play3d`
   (CR-tilt) + `scene_shot.py` eyeball loop. Success = Boom looks *better*
   live than today's PixiJS and identical logic drives both.
2. **P1**: cinema pipeline switched to headless-Chrome (delete PIL renderer);
   SMAX adapter + viewer (proves generality with game #3); asset registry +
   Kenney/Quaternius base packs; asset_audit in CI.
3. **P2**: hok_env 1v1 adapter (Kaiwu-mimic milestone); card-family stage for
   Hold'em/Mahjong; PixiJS clients migrate to schema frames.
4. **P3**: Godot spectator for embodied-3D when such a game lands.

## 8. What we deliberately do NOT build

- No general-purpose game engine, no editor, no physics (simulation owns
  physics; renderers interpolate).
- No Unity/Unreal dependency anywhere in the platform.
- No asset without a manifest license entry.

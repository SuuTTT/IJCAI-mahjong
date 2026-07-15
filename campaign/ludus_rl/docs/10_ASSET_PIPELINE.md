# 10 · AI-generated game assets: pipeline, tools, and where assets "live"

Companion to docs/09 §5. Covers the concrete asset list for Boom (battlefield,
towers, characters, projectiles, VFX, card/elixir/health UI; BGM + SFX) and
the general pipeline any Ludus game reuses.

## 1. "Is there a GitHub for assets?"

Yes — several layers, use all of them:

| layer | what | our use |
|---|---|---|
| **git-lfs in this repo** | assets/ tracked with LFS; versioned WITH the code that renders them | canonical home for shipped packs (source of truth) |
| **Hugging Face datasets** | free LFS-backed hosting, versioned, public | mirror of packs + the AI-gen provenance files (we already publish models there) |
| **Kenney.nl** | thousands of CC0 sprites/3D/audio/UI, coherent style | base packs to fill gaps instantly |
| **OpenGameArt / freesound.org** | community art & audio with per-item licenses (filter CC0) | SFX especially |
| **PolyHaven** | CC0 textures/HDRI | 3D tier materials |
| **Quaternius** | CC0 low-poly glTF characters/buildings | 3D unit models |
| Sketchfab / itch.io | marketplaces with license metadata | browse/buy one-offs; not a pipeline |

Rule from docs/09 stands: every file enters through a pack `manifest.json`
entry with license + source; `scripts/asset_audit.py` fails CI otherwise. For
AI-generated entries the manifest records **tool, model, prompt, date** —
provenance is the license story.

## 2. AI generation: tool choices per asset class (2026)

### Visual
| asset | tool/approach | notes |
|---|---|---|
| Battlefield tilemap/mesh | SDXL/Flux tileable-texture prompts → seam-fix (offset+inpaint); mesh stays procedural (our three.js board) | texture the existing geometry, don't generate geometry |
| Tower models (king/princess) | text-to-3D (Meshy / Tripo / Hyper3D-Rodin) → glTF → Blender-headless decimate+retopo to <5k tris | commercial ownership needs their paid tiers — record in manifest |
| Character sprites (melee/ranged/swarm) | SD/Flux + **one style LoRA** for the whole set + ControlNet pose for 8-direction sheets; rembg → atlas | style LoRA = the consistency secret; train once on ~30 curated "Ludus look" images |
| Character 3D (later) | Tripo/Meshy from the SAME concept images | concept-first keeps 2D/3D coherent |
| Projectiles (arrows/fireballs) | SD sprite passes OR pure procedural (three.js glow spheres already read well) | cheap wins first |
| VFX (spawn ring, AoE blast, deploy grid) | procedural shaders/particles > AI frames | AI is the wrong tool here; Kenney particle pack fills gaps |
| Card frames / elixir bar / health bar UI | SD for ornamental frames → vectorize (vtracer) → 9-slice PNG | 9-slice makes one frame fit all cards |

### Audio
| asset | tool/approach | notes |
|---|---|---|
| Battle-loop BGM | MusicGen (open weights, runs on our 3090) or Suno/Udio (paid tier for commercial rights) | loop-point editing in ffmpeg/sox |
| SFX (placement, melee, launch, explosion, tower fall) | ElevenLabs SFX / AudioGen for bespoke; **freesound CC0 + sox editing is usually better and faster** | layer 2-3 samples per event; normalize LUFS |

## 3. The pipeline (design)

```
concepts/            style anchor images + per-archetype prompts (text files, versioned)
scripts/gen_assets/  batch drivers: prompt -> raw -> cleanup -> pack
  gen_sprites.py     diffusers (SDXL-turbo) on the training box GPU, league-gap scheduled
  gen_music.py       MusicGen on the same terms
  post.py            rembg, Real-ESRGAN upscale, atlas packing, 9-slice, LUFS normalize
assets/<pack>/       output + manifest.json (license/provenance per entry)
scripts/asset_audit.py   CI gate: no manifest entry, no merge
```

Compute policy (owner rule, 2026-07-05): **generation never runs on local/dev
machines** — image/audio models run on the training box's 3090 in league gaps
(SDXL-turbo ~6GB fits beside training; big jobs get a burst-rented GPU) or via
hosted APIs when licensing is cleaner.

Human (or agent) curation gate: generation is cheap, taste is the filter —
every batch gets a contact-sheet PNG for eyeball review before packing (the
same review loop as scene_shot).

## 4. Boom asset backlog (concrete, in order)

1. Ground/river/bridge textures (tileable, 512px) + card-frame UI + elixir bar
   — pure SD work, biggest visual lift for /play3d.
2. Tower models: one keep + one king variant via text-to-3D, team-tinted by
   shader (docs/09) — replaces cylinders.
3. Character style LoRA + 12 archetype sprite sheets (also feeds 2D client).
4. SFX pack: 6 core events from freesound-CC0 + sox; BGM loop via MusicGen.
5. VFX: procedural deploy-grid + AoE ring shaders (no AI).

Each lands as its own pack with manifest; /play3d resolves keys with the
placeholder fallback, so partial packs ship incrementally.

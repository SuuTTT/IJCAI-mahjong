# From billboards to a fully owner-generated 3D battlefield in one day

*Ludus devlog #6 — 2026-07-05*

## The loop that shipped it

Owner generates (GPT for 2D, text-to-3D for GLB) → prompt packs with exact
keys and specs → intake with provenance manifests → renderer picks assets up
by key with a fallback chain (3D model → 2D sprite → primitive) → agent
screenshot-verifies placement → iterate. Three batches in ~a day:

1. **Board & UI** (10 images): tileable grass/river/planks, tower pads, card
   frames, elixir bar. 10/10 passed curation; the agent's own SD-turbo
   attempt went 0-for-4 and was honestly retired to LoRA-era duty.
2. **Characters as billboards** (10 images): passed, shipped — and
   immediately taught us two things: count=3 cards need single-creature art
   (the Sporeling trio crowd bug), and flat billboards read poorly next to
   real geometry. Owner verdict: "billboards suck, I need 3D models."
3. **3D models** (10 GLBs via text-to-3D free tiers): the whole hog-cycle
   deck plus both towers, each under 2k triangles with embedded textures.
   Live now, with emissive team-tinting (multiply-tint blacked out the dark
   textures — lesson logged).

## Renderer notes

- ES-modules migration (for GLTFLoader) accidentally fixed headless WebGL —
  the screenshot-verification loop now covers geometry and layout (software-GL
  still flakes on texture uploads; real browsers are truth for those).
- The fallback chain means art lands incrementally: cards without models show
  sprites, without sprites show capsules. Nothing waits for anything.
- Two silent-edit bugs shipped and were caught by screenshots within minutes
  (missing loader dep 404, an undefined function killing the render loop) —
  the eyeball loop is not optional.

## Next

Batch 4: remaining deck-B units + king/princess tower texture variants;
procedural VFX shaders; SFX pack. Then the schema formalization so SMAX
rides the same renderer.

# Asset prompt pack — batch 1 (Boom board + UI)

For generation via GPT-4o / Gemini / any image model. Every prompt shares the
style anchor so the set is coherent. Save results as
`assets/boom-base/<key>.png` (exact key), then they auto-appear in /play3d.

**Style anchor (prepend to every prompt):**
> Stylized painterly-flat game art, clean readable shapes, soft top-down
> lighting, muted saturated palette (deep greens, slate blues, warm wood),
> subtle hand-painted texture, no text, no watermark, no photo-realism.

**Global specs:** PNG. Tileables must tile seamlessly. No baked shadows
(the renderer lights the scene). Nothing resembling Clash Royale's actual art
— original look ("Ludus style").

| id | key | size | prompt (after the anchor) |
|----|-----|------|---------------------------|
| T1 | ground_grass.png | 512×512, seamless | seamless tileable grass field texture for a fantasy battle arena, short mowed grass with faint mower stripes and tiny clover patches, viewed straight top-down |
| T2 | ground_grass_dark.png | 512×512, seamless | same grass as before but one shade darker and slightly cooler, for the enemy half of the arena |
| T3 | river_water.png | 512×256, seamless horizontally | seamless tileable stylized river water strip, gentle blue-teal ripples flowing horizontally, a few sparkle highlights, top-down |
| T4 | bridge_planks.png | 256×256 | wooden bridge deck of thick weathered planks with two darker support beams, warm brown, top-down |
| T5 | tower_pad.png | 256×256, alpha | circular stone platform / tower foundation with cracked flagstones and a ring border, on transparent background, top-down |
| U1 | card_frame.png | 512×640, alpha | ornate but clean game card frame, dark slate metal with subtle blue gem inlay at top center, empty middle, transparent background and transparent center window |
| U2 | card_frame_gold.png | 512×640, alpha | same card frame but gold/champion variant with warm glow accents |
| U3 | elixir_drop.png | 256×256, alpha | glossy magenta-purple elixir droplet icon with soft inner glow, game UI style, transparent background |
| U4 | elixir_bar.png | 1024×128, alpha | horizontal UI bar frame for a resource meter, dark slate with rounded ends and 10 subtle segment ticks, EMPTY (no fill), transparent background |
| U5 | hp_frame.png | 512×96, alpha | tiny floating health-bar frame, minimal dark outline with slight bevel, empty center, transparent background |
| V1 | deploy_ring.png | 256×256, alpha | glowing ring ground-marker for unit deployment, thin double circle with 4 small notches, cyan, on transparent background, top-down |
| V2 | blast_ring.png | 256×256, alpha | radial explosion shockwave ring, orange-gold gradient fading outward, transparent background, top-down |
| X1 | contact_sheet | — | (no image — after generating, paste all results in one message/grid so we can review coherence before packing) |

**Hand-back:** send the images (or a link); I register each with
`scripts/asset_intake.py` (records tool + prompt id as provenance), run the
audit, wire them into /play3d materials, and screenshot-verify placement.

**Batch 2 preview (needs batch-1 style locked first):** 12 character
archetype sheets (melee/ranged/swarm/tank/air/spirit…), tower models
(text-to-3D), SFX list — prompts will follow the same pattern with the
character style anchor derived from whatever batch 1 converges on.

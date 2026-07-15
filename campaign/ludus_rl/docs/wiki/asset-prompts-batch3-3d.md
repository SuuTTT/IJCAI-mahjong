# Asset prompt pack — batch 3: 3D character models (text-to-3D)

Billboards are retired for units — we load real glTF models now. Generate with
any text-to-3D tool (Meshy.ai, Tripo3D, Luma Genie, Rodin — free tiers exist;
GPT can also produce these via its 3D-capable modes if available to you).
**Export format: GLB** (single file, embedded textures).

**Style anchor (prepend to every prompt):**
> Low-poly stylized game character, chunky readable proportions, hand-painted
> flat textures, muted saturated fantasy palette, single mesh, standing
> neutral pose, no base/pedestal, suitable for a real-time strategy game.

**Specs:** GLB, ≤15k triangles preferred, embedded texture, character facing
+Z (front). Name files exactly as below, scp the folder as `~/asset_glb_1/`.

| id | file | card | prompt (after the anchor) |
|----|------|------|---------------------------|
| M1 | unit_ramhound.glb | Ramhound | a burly armored war-hound with a blunt bronze battering-ram helmet, muscular shoulders, leather harness straps, earthy brown fur |
| M2 | unit_frostsprite.glb | Frostsprite | a tiny frost wisp creature of translucent icy-blue crystal, jagged little ice-shard limbs, glowing crystal core in its chest |
| M3 | unit_sporeling.glb | Sporelings (SINGLE!) | ONE small mushroom-cap goblin with a glowing spore pouch, pale olive skin, spindly limbs — a single creature, not a group |
| M4 | unit_watchpost.glb | Watchpost | a squat wooden watchtower turret with a mounted ballista on a rotating top, iron banding, small stone base |
| M5 | unit_longshot.glb | Longshot | a lean hooded sharpshooter holding a long elegant arbalest rifle, deep slate-blue cloak, copper goggles on the hood |
| M6 | unit_shellfort.glb | Shellfort | a stocky golem-tortoise with pale blue ice-rock plates on its shell, heavy stubby legs, frost crystals at the joints |
| M7 | unit_lancer.glb | Lancer | an armored knight riding a sturdy charger horse, lowered tournament lance, storm-gray plate armor, neutral gray tabard |
| M8 | unit_bulwark.glb | Bulwark | a stout knight with a tall kite shield and short sword, rounded heavy armor, neutral gray tabard |
| M9 | tower_princess.glb | princess tower | a compact stone defense tower with a peaked tiled roof and a small ballista window, banded masonry, fantasy castle style |
| M10 | tower_king.glb | king tower | a larger square stone keep with crenellations, a central peaked roof, and a broad fortified base, same masonry family as the small tower |

**Notes**
- M3 fixes the Sporelings bug (the 2D trio art rendered a crowd per unit —
  engine spawns 3 separate units, each needs ONE goblin).
- Keep large cloth/banner areas neutral gray where natural (team tint).
- If a tool offers "PBR" vs "stylized/hand-painted" texturing, pick stylized.
- Turntable-check each model before export: no floating parts, no pedestal.
- The 2D billboards stay as the fallback chain, nothing breaks while models
  arrive one by one — drop what you have, the rest keeps rendering as before.

# Pending asset prompts — generate when we hit 10

Rolling queue. New cards/features append here; at 10 items the owner runs a
roll (same anchors as `asset_prompts_complete.md`: portrait anchor for images,
3D anchor + rig/idle/walk/attack for units, static for spells/props).

| # | file | kind | prompt (after the matching anchor) |
|---|------|------|-------------------------------------|
| 1 | portrait_whirlgale.png | portrait | a roaring teal-gray cyclone spell tearing across a battlefield, debris and leaves spiraling, eye of the storm glowing faint violet |
| 2 | portrait_harpooner.png | portrait | a burly sea-weathered fisherman in oilskin coat hefting a barbed harpoon on a chain, cold spray behind him, determined grin |
| 3 | unit_harpooner.glb | 3D unit (rigged) | a burly fisherman in an oilskin coat holding a barbed harpoon with a coiled chain at his hip — attack clip should read as a throw-and-yank |
| 4 | proj_harpoon.glb | 3D projectile (static) | a barbed iron harpoon with a short chain link tail, +Z forward |
| 5 | fx_whirlgale.glb | 3D prop (static) | a stylized low-poly tornado cone, translucent teal-gray spiral bands, no base |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

**Trigger**: when rows are full → owner generates as one roll → scp →
intake/wire/verify as usual → clear the table.
| 6 | portrait_timberoll.png | portrait | a massive spiked wooden log mid-roll down a battlefield lane, splinters and dust flying, moss and iron bands |
| 7 | portrait_marrowkeg.png | portrait | a wooden barrel mid-air bursting open with three grinning bone-pale goblins tumbling out, spore dust |
| 8 | portrait_brawlkeg.png | portrait | a reinforced barrel rolling forward, a burly briar-wood barbarian bursting from its cracked staves, axe raised |

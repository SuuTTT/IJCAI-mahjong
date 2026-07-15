# Asset prompt pack — batch 2 (hog-cycle deck characters, 10 images)

Field sprites for our classic cycle deck. These render as billboards in the
3D CR-tilt camera, so every character is drawn from the SAME viewpoint.

**Style anchor (prepend to every prompt):**
> Stylized painterly-flat game character art, clean readable silhouette,
> soft lighting, muted saturated palette matching a deep-green battlefield,
> subtle hand-painted texture, single character centered on a fully
> transparent background, viewed from a three-quarter top-down angle facing
> away from the viewer (up-screen), no text, no watermark, no ground shadow.

**Global specs:** PNG with real transparency, 1024×1024 (bars 2:1 fine for
effects). Original creature designs — evocative of the archetype but NOT
copies of any existing game's characters.

| id | key | card | prompt (after the anchor) |
|----|-----|------|---------------------------|
| C1 | unit_ramhound.png | Ramhound (wincon) | a burly armored war-hound with a blunt battering-ram helmet of banded bronze, muscular shoulders, short reins flying, charging pose, earthy browns with bronze accents |
| C2 | unit_frostsprite.png | Frostsprite (cycle) | a tiny mischievous frost wisp creature, translucent icy-blue body like living frost with a crystal core, trailing snowflake sparkles, mid-hop pose |
| C3 | unit_sporelings.png | Sporelings (swarm) | a trio of tiny mushroom-cap goblins with glowing spore pouches, pale olive skin, spindly limbs, scurrying in loose formation |
| C4 | unit_watchpost.png | Watchpost (building) | a squat wooden watchtower turret with a swiveling ballista mounted on top, iron banding, small stone base, slight three-quarter view |
| C5 | unit_longshot.png | Longshot (ranged) | a lean hooded sharpshooter with a long elegant arbalest rifle, deep slate-blue cloak, copper goggles pushed up, braced aiming stance |
| C6 | unit_shellfort.png | Shellfort (mini tank) | a stocky golem-tortoise of pale blue ice-rock plates, frost mist seeping from its joints, slow heavy stride |
| C7 | unit_lancer.png | Lancer (charge) | an armored knight on a sturdy charger horse, lowered tournament lance with a pennant, storm-gray plate armor with a neutral gray tabard, mid-gallop |
| C8 | fx_fireburst.png | Fireburst (spell) | a roiling fireball comet with a molten orange core and trailing embers, motion-blurred tail, on transparent background |
| C9 | fx_shockwave.png | Shockwave (spell) | a crackling forked lightning bolt striking downward with a small radial spark burst at the base, electric blue-white, on transparent background |
| C10 | portrait_ramhound.png | (card-art pilot) | bust portrait of the armored battering-ram war-hound from C1, dramatic low angle, tight crop for a card frame, richer painterly detail |

**Notes**
- C10 is a pilot: if portraits look good in the card frames, batch 3 is the
  full 61-card portrait set (I'll generate that list).
- The `neutral gray tabard` on C7 (and any cloth) lets us team-tint blue/red
  by shader; where possible keep one large clothing area neutral gray.
- Same hand-back flow: scp to `~/asset_gpt_2/` in generation order 1-10.

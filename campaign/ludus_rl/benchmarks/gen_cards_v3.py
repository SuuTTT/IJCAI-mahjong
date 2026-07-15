"""Generate boom/cards.csv v3: exact tournament-standard (level 11) numbers from
RoyaleAPI cr-api-data for each card's analog. Numbers only (functional facts);
names/art in the game remain original; cr_ref records the analog for provenance.

Units in source: millitiles (range/radius), ms (hit_speed/deploy), tiles/min (speed).
Ours: fp=1/256 tile, 200ms ticks. speed_fp = t_min*256/300; ticks = round(ms/200).
"""
import json, csv, sys

S = json.load(open('cr_stats.json'))
CARDS_META = {c['name']: c for c in json.load(open('cr_cards.json'))}
CH = {c['name']: c for c in S['characters'] if c.get('name')}
TROOP_CARDS = {t['name']: t for t in S['troop'] if t.get('name')}
BLD = {b['name']: b for b in S['building'] if b.get('name')}
SPELL = {s['name']: s for s in S['spell'] if s.get('name')}
PROJ = {p['name']: p for p in S['projectile'] if p.get('name')}

# tournament standard = card level 11; per-level arrays start at the rarity's
# minimum card level, so index = 10 - offset(rarity)
RARITY_OFF = {'Common': 0, 'Rare': 2, 'Epic': 5, 'Legendary': 8, 'Champion': 10}

def lv(entry, field, rarity=None):
    v = entry.get(field)
    if isinstance(v, list):
        i = 10 - RARITY_OFF.get(rarity or entry.get('rarity') or 'Common', 0)
        return v[min(max(i, 0), len(v) - 1)]
    return v

def ticks(ms):    return max(1, round(ms / 200)) if ms else 0
def fp_speed(tm): return round(tm * 256 / 300) if tm else 0
def tiles(mt):    return round(mt / 1000, 2) if mt else 0.0

def _char_dmg(ch):
    d = lv(ch, 'damage_per_level') or lv(ch, 'damage') or 0
    if not d and ch.get('projectile'):
        p = PROJ.get(ch['projectile']) or {}
        d = lv(p, 'damage_per_level', ch.get('rarity')) or lv(p, 'damage', ch.get('rarity')) or 0
        d *= max(1, ch.get('multiple_projectiles') or 1)
    return d

def troop(our, cr_card, cr_char, arch, count=None, **ov):
    card = (TROOP_CARDS.get(cr_card) or TROOP_CARDS.get(cr_card.replace(' ', ''))
            or TROOP_CARDS.get(cr_card.replace(' ', '').replace('.', '')) or {})
    ch = CH[cr_char]
    n = count or card.get('summon_number') or 1
    if n == 0: n = 1
    meta_cost = (CARDS_META.get(cr_card) or {}).get('elixir')
    row = dict(
        name=our, archetype=arch,
        cost=meta_cost or card.get('mana_cost') or ch.get('elixir') or ov.get('cost'),
        hp=lv(ch, 'hitpoints_per_level') or lv(ch, 'hitpoints'),
        dmg=_char_dmg(ch),
        period=ticks(ch.get('hit_speed')),
        speed=fp_speed(ch.get('speed')),
        range=tiles(ch.get('range')),
        count=min(n, 15),
        air=1 if (ch.get('flying_height') or 0) > 0 else 0,
        targets_air=1 if ch.get('attacks_air') else 0,
        bldg_only=1 if ch.get('target_only_buildings') else 0,
        splash=tiles(ch.get('area_damage_radius')) if ch.get('area_damage_radius') else 0.0,
        is_spell=0, effect=0, duration=0, decay=0,
        aura_type=0, aura_power=0, aura_radius=0.0,
        hit_effect=0, hit_dur=0, suicide=0, death_dmg=0, death_r=0.0, anywhere=0,
        tower_pct=100, cr_ref=cr_card,
    )
    row.update(ov)
    return row

def building(our, cr_name, arch, **ov):
    b = BLD[cr_name]
    hp = lv(b, 'hitpoints_per_level') or lv(b, 'hitpoints')
    life_ticks = (b.get('life_time') or 30000) / 200
    row = dict(
        name=our, archetype=arch,
        cost=(CARDS_META.get(ov.pop('meta', cr_name), {}) or {}).get('elixir') or b.get('mana_cost'),
        hp=hp, dmg=_char_dmg(b),
        period=ticks(b.get('hit_speed')), speed=0,
        range=tiles(b.get('range')), count=1, air=0,
        targets_air=1 if b.get('attacks_air') else 0, bldg_only=0,
        splash=tiles(b.get('area_damage_radius')) if b.get('area_damage_radius') else 0.0,
        is_spell=0, effect=0, duration=0,
        decay=max(1, round(hp / life_ticks)),
        aura_type=0, aura_power=0, aura_radius=0.0,
        hit_effect=0, hit_dur=0, suicide=0, death_dmg=0, death_r=0.0, anywhere=0,
        tower_pct=100, cr_ref=cr_name,
    )
    row.update(ov)
    return row

def dmg_spell(our, proj_name, cost, radius=None, effect=0, duration=0, tower_pct=None, cr=None, **ov):
    p = PROJ[proj_name]
    ctd = p.get('crown_tower_damage_percent')
    if tower_pct is None:
        tower_pct = 100 + ctd if ctd else 100          # e.g. -70 -> 30%
    rar = ov.pop('rarity', None)
    waves = ov.pop('waves', 1)
    row = dict(
        name=our, archetype='spell_dmg' if effect == 0 else 'spell_util',
        cost=cost, hp=0,
        dmg=(lv(p, 'damage_per_level', rar) or lv(p, 'damage', rar) or 0) * waves,
        period=0, speed=0,
        range=radius if radius is not None else tiles(p.get('radius')),
        count=1, air=0, targets_air=1, bldg_only=0,
        splash=radius if radius is not None else tiles(p.get('radius')),
        is_spell=1, effect=effect, duration=duration, decay=0,
        aura_type=0, aura_power=0, aura_radius=0.0,
        hit_effect=0, hit_dur=0, suicide=0, death_dmg=0, death_r=0.0, anywhere=0,
        tower_pct=tower_pct, cr_ref=cr or proj_name,
    )
    row.update(ov)
    return row

def util_spell(our, cost, radius, effect, duration, dmg=0, tower_pct=30, cr=''):
    return dict(name=our, archetype='spell_util', cost=cost, hp=0, dmg=dmg, period=0,
                speed=0, range=radius, count=1, air=0, targets_air=1, bldg_only=0,
                splash=radius, is_spell=1, effect=effect, duration=duration, decay=0,
                aura_type=0, aura_power=0, aura_radius=0.0, hit_effect=0, hit_dur=0,
                suicide=0, death_dmg=0, death_r=0.0, anywhere=0,
                tower_pct=tower_pct, cr_ref=cr)

EFFECT_SLOW, EFFECT_RAGE, EFFECT_STUN = 1, 2, 3
rows = []
A = rows.append

# ---- tanks ----
A(troop('Bulwark', 'Knight', 'Knight', 'tank'))
A(troop('Ironhide', 'Giant', 'Giant', 'tank'))
A(troop('Golemite', 'Golem', 'Golem', 'tank'))                     # no death-split (doc)
A(troop('Shellfort', 'Ice Golem', 'IceGolemite', 'tank'))
A(troop('Duskblade', 'P.E.K.K.A', 'Pekka', 'tank'))
A(troop('Mossback', 'Valkyrie', 'Valkyrie', 'tank'))
# ---- swarm ----
A(troop('Sporelings', 'Skeletons', 'Skeleton', 'swarm'))
A(troop('Ratpack', 'Skeleton Army', 'Skeleton', 'swarm', cost=3))
A(troop('Marrowlings', 'Goblins', 'Goblin', 'swarm'))
A(troop('Thistlekin', 'Barbarians', 'Barbarian', 'swarm'))
A(troop('Ashhorde', 'Elite Barbarians', 'AngryBarbarian', 'swarm', count=2))
A(troop('Gnatcloud', 'Bats', 'Bat', 'swarm'))
# ---- air ----
A(troop('Zephyrling', 'Minions', 'Minion', 'air'))
A(troop('Wispflock', 'Minion Horde', 'Minion', 'air', cost=5, count=6))
A(troop('Skyray', 'Mega Minion', 'MegaMinion', 'air'))
A(troop('Emberwing', 'Baby Dragon', 'BabyDragon', 'air'))
A(troop('Bonekites', 'Skeleton Dragons', 'SkeletonDragon', 'air', count=2))
A(troop('Cloudcalf', 'Lava Hound', 'LavaHound', 'wincon'))         # no pups (doc)
A(troop('Stormkite', 'Balloon', 'Balloon', 'wincon'))              # death bomb below
A(troop('Whirligig', 'Flying Machine', 'DartBarrell', 'support'))
# ---- splash / ranged ----
A(troop('Emberwitch', 'Wizard', 'Wizard', 'splash'))
A(troop('Cinderpup', 'Bomber', 'Bomber', 'splash'))
A(troop('Headsman', 'Executioner', 'AxeMan', 'splash'))
A(troop('Rockhurler', 'Bowler', 'Bowler', 'splash'))               # no knockback (doc)
A(troop('Sparkmaw', 'Sparky', 'ZapMachine', 'splash'))             # no charge-reset (doc)
A(troop('Longshot', 'Musketeer', 'Musketeer', 'ranged'))
A(troop('Hexarcher', 'Archers', 'Archer', 'ranged', count=2))
A(troop('Quillquick', 'Dart Goblin', 'BlowdartGoblin', 'ranged'))
A(troop('Dart Frog', 'Princess', 'Princess', 'ranged'))
A(troop('Prismgunner', 'Magic Archer', 'EliteArcher', 'ranged'))   # no pierce (doc)
A(troop('Thornsling', 'Spear Goblins', 'SpearGoblin', 'cycle'))
# ---- antiair-ish / support ----
A(troop('Flakbot', 'Zappies', 'MiniZapMachine', 'antiair', count=3,
        hit_effect=EFFECT_STUN, hit_dur=3))
A(troop('Frostcaller', 'Ice Wizard', 'IceWizard', 'antiair',
        hit_effect=EFFECT_SLOW, hit_dur=13))
A(troop('Voltmage', 'Electro Wizard', 'ElectroWizard', 'support',
        hit_effect=EFFECT_STUN, hit_dur=3))                        # no spawn-zap (doc)
A(troop('Skewerhulk', 'Mini P.E.K.K.A', 'MiniPekka', 'antiair'))
A(troop('Timberbeast', 'Lumberjack', 'RageBarbarian', 'support'))  # no death-rage (doc)
# ---- spirits (kamikaze) ----
A(troop('Cindersprite', 'Fire Spirit', 'FireSpirits', 'cycle', suicide=1))
A(troop('Frostsprite', 'Ice Spirit', 'IceSpirits', 'cycle', suicide=1,
        hit_effect=EFFECT_STUN, hit_dur=5))
A(troop('Voltsprite', 'Electro Spirit', 'ElectroSpirit', 'cycle', suicide=1,
        hit_effect=EFFECT_STUN, hit_dur=3))                        # no chain (doc)
# ---- wincons ----
A(troop('Ramhound', 'Hog Rider', 'HogRider', 'wincon'))
A(troop('Siege Snail', 'Royal Giant', 'RoyalGiant', 'wincon'))
A(troop('Duneworm', 'Goblin Giant', 'GoblinGiant', 'wincon'))      # no backpack gobs (doc)
A(troop('Boarband', 'Royal Hogs', 'RoyalHog', 'wincon', count=4))
A(troop('Triplet Muses', 'Three Musketeers', 'Musketeer', 'ranged', cost=9, count=3))
A(troop('Wallgnashers', 'Wall Breakers', 'Wallbreaker', 'wincon', count=2, suicide=1))
A(troop('Burrower', 'Miner', 'Miner', 'wincon', anywhere=1))
A(troop('Gravewalker', 'Giant Skeleton', 'GiantSkeleton', 'tank'))  # death bomb below
# ---- buildings ----
A(building('Watchpost', 'Cannon', 'building', meta='Cannon'))
A(building('Tesla Bloom', 'Tesla', 'building', meta='Tesla'))
A(building('Boomkiln', 'BombTower', 'building', meta='Bomb Tower'))
A(building('Mortar Crab', 'Mortar', 'building', meta='Mortar'))
A(building('Skystinger', 'Xbow', 'building', meta='X-Bow'))
# ---- spells ----
A(dmg_spell('Fireburst', 'FireballSpell', 4, cr='Fireball', rarity='Rare'))
A(dmg_spell('Sparkarc', 'ArrowsSpell', 3, cr='Arrows', waves=3))
A(dmg_spell('Stonefall', 'RocketSpell', 6, cr='Rocket', rarity='Rare'))
A(dmg_spell('Glacierlash', 'SnowballSpell', 2, effect=EFFECT_SLOW, duration=13, cr='Giant Snowball'))
A(util_spell('Shockwave', 2, 2.5, EFFECT_STUN, 3, dmg=0, cr='Zap'))          # dmg patched below
A(util_spell('Overclock', 2, 3.0, EFFECT_RAGE, 30, cr='Rage'))
A(util_spell('Frostfield', 4, 3.0, EFFECT_STUN, 20, dmg=0, cr='Freeze'))     # freeze = full stop
A(util_spell('Emberrain', 6, 3.5, EFFECT_STUN, 3, dmg=0, cr='Lightning'))    # radius-approx (doc)

# spell damages living in projectiles keyed by buff spells
def set_dmg(name, source, fallback):
    r = next(r for r in rows if r['name'] == name)
    e = SPELL.get(source) or PROJ.get(source) or {}
    r['dmg'] = lv(e, 'damage_per_level') or lv(e, 'damage') or fallback

set_dmg('Shockwave', 'Zap', 192)  # common
set_dmg('Frostfield', 'Freeze', 184)
r = next(x for x in rows if x['name'] == 'Emberrain')
p = PROJ.get('LighningSpell') or {}
r['dmg'] = lv(p, 'damage_per_level', 'Epic') or 1056

# death bombs (exact where the bomb object exists)
for name, bomb, fallback_dmg, fallback_r in [
        ('Stormkite', 'BalloonBomb', 800, 2.0), ('Gravewalker', 'GiantSkeletonBomb', 1696, 3.0)]:
    r = next(x for x in rows if x['name'] == name)
    b = BLD.get(bomb) or {}
    r['death_dmg'] = lv(b, 'damage_per_level') or lv(b, 'damage') or fallback_dmg
    r['death_r'] = tiles(b.get('area_damage_radius')) or fallback_r

PINS = {  # damage lives on decorated/aux objects in the dataset; pinned to
          # documented tournament-standard values
    'Dart Frog': ('dmg', 154), 'Watchpost': ('dmg', 212), 'Boomkiln': ('dmg', 210),
    'Mortar Crab': ('dmg', 208), 'Skystinger': ('dmg', 26),
}
for nm, (f, v) in PINS.items():
    r = next(x for x in rows if x['name'] == nm)
    if not r[f]:
        r[f] = v
for r in rows:
    assert r['is_spell'] or r['dmg'] or r['name'] in (), f"zero dmg: {r['name']}"
    assert r['cost'], f"no cost: {r['name']}"

assert len(rows) == 60, len(rows)
names = [r['name'] for r in rows]
assert len(set(names)) == 60, sorted(set(n for n in names if names.count(n) > 1))

cols = ['id','name','archetype','cost','hp','dmg','period','speed','range','count',
        'air','targets_air','bldg_only','splash','is_spell','effect','duration','decay',
        'aura_type','aura_power','aura_radius','hit_effect','hit_dur','suicide',
        'death_dmg','death_r','anywhere','tower_pct','cr_ref']
with open('cards_v3.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for i, r in enumerate(rows):
        r['id'] = i
        r.setdefault('aura_radius', 0.0)
        w.writerow(r)

for r in rows:
    print(f"{r['id']:>2} {r['name']:<14} <{r['cr_ref']:<16}> {r['archetype']:<9} "
          f"c{r['cost']} hp{r['hp']} dmg{r['dmg']} p{r['period']} v{r['speed']} "
          f"rg{r['range']} x{r['count']} air{r['air']}/ta{r['targets_air']} "
          f"b{r['bldg_only']} sp{r['splash']}")

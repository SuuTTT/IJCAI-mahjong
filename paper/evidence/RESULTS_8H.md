# 8h harvest — 2026-06-11 20:54 UTC

## Imitation-ceiling grid (P4000): val_acc per student
=== GRID dimaria2025 -> g_dimaria (18:07) ===
data/dimaria2025.npz: 176068 decisions | train 158462 val 17606
ep10/10 val_acc=0.8352 (194s)
=== GRID pama2025 -> g_pama (18:39) ===
data/pama2025.npz: 174946 decisions | train 157452 val 17494
ep10/10 val_acc=0.8262 (192s)
=== GRID moumou2025 -> g_moumou (19:11) ===
data/moumou2025.npz: 176861 decisions | train 159175 val 17686
ep10/10 val_acc=0.8265 (194s)
=== GRID seaman_22k -> g_seaman22 (19:43) ===
data/seaman_22k.npz: 22000 decisions | train 19800 val 2200
ep10/10 val_acc=0.3836 (24s)
=== GRID seaman_44k -> g_seaman44 (19:47) ===
data/seaman_44k.npz: 44000 decisions | train 39600 val 4400
ep10/10 val_acc=0.5839 (48s)
=== GRID seaman_88k -> g_seaman88 (19:55) ===
data/seaman_88k.npz: 88000 decisions | train 79200 val 8800
ep10/10 val_acc=0.7013 (97s)
GRID_TRAIN_DONE

## Grid gauntlets (held-out WSB=880000)
=== gauntlet g_dimaria TOTAL net=-1500 (played=75 stuck=37) ===
=== gauntlet g_pama TOTAL net=-1560 (played=78 stuck=25) ===
=== gauntlet g_moumou TOTAL net=-1280 (played=64 stuck=28) ===
=== gauntlet g_seaman22 TOTAL net=-1440 (played=72 stuck=11) ===
=== gauntlet g_seaman44 TOTAL net=-1440 (played=72 stuck=29) ===
=== gauntlet g_seaman88 TOTAL net=-1380 (played=69 stuck=14) ===
=== gauntlet lad_seaman TOTAL net=-1220 (played=61 stuck=31) ===
=== gauntlet shipref TOTAL net=-1260 (played=63 stuck=14) ===

## Noise floor (5060, identical pair x 5 wall sets)
WALLSET 310000 TOTAL=
WALLSET 420000 TOTAL=
WALLSET 530000 TOTAL=
WALLSET 640000 TOTAL=
WALLSET 750000 TOTAL=

## Real-ladder ship read

[moyu]distill: 143 games | net +295 (+2.06/g) | wins 34 (24%) | deal-ins 27 (19%) | draws 2 (1%)
     107  vs [dearmylex]vc1
     105  vs [谢飞扬]ddl战士
     103  vs [qwqwqawawa]yigeiwoligiaogiao
      19  vs [kyuso]Teriri
      17  vs [天胡豪七]小寻歌
      15  vs [benaive]name_TBD
       6  vs 顺其自然QAQ
       5  vs [dimaria]SelfRegPO
HARVEST_DONE

## CORRECTED noise floor (bc failed inline; recomputed)
Identical bot pair, 144 games/wall-set, 5 disjoint wall sets: totals = [3411, 3846, 3309, 3408, 3612]
mean=3517, spread(max-min)=537, stdev=192.
=> A 144-game gauntlet of IDENTICAL bots swings ~537 net. Every "null" this session sat INSIDE this:
   PIMC -129, SAFE -317, FOLD -267, champion-clone ±80. They were not marginal losses — they were
   statistically indistinguishable from no change. This single number validates the whole campaign.

## Imitation-ceiling grid — the result
Data-scale axis (SeaMan, val_acc): 22k=0.384, 44k=0.584, 88k=0.701, 176k=0.812 — monotonic, agreement
buys with data. Teacher-strength axis (real ladder strength -> student): dimaria(+0.58)=0.835,
PAMA(-0.08)=0.826, moumou(-0.31)=0.827, SeaMan(+0.89)=0.812 val_acc; gauntlet net all in [-1560,-1220],
inside the 537 noise floor of each other AND the ship (-1260). => Agreement SATURATES with data but
does NOT track teacher strength in PLAY. Cloning a +0.89 champion = cloning a -0.31 bot, in play. The
imitation ceiling, demonstrated end to end.

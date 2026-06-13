# Real-field analysis — [moyu]distill (ship = lad_chunjiandu), 82 unique games vs ladder #1/#3/#4

Dedup note: ladder_report.py counts per-seat-occurrence across overlapping collection dirs (128);
the deduplicated unique-match count is 82. All figures below are dedup'd.

## Net
-20 points over 82 games = -0.24/game. Seat-balanced (+77 over 70 seat-0 games; -97 over 12
seat-1/3 games). Field: vc1(#4), ddl战士(#1), yigeiwoligiaogiao(#3) — top-of-ladder, hardest mix.

## Loss decomposition (774 total points paid out)
| mode                | games | pts paid | controllable |
|---------------------|------:|---------:|--------------|
| opponent self-draw  |  29%  |   416    | NO           |
| bystander rong      |  35%  |   196    | NO           |
| WE deal in          |  11% (9) | 162   | YES (only one) |
=> Defense (deal-in avoidance) addresses only 162/774 = 21% of losses; the 3-arm A/B showed the
   genbutsu filter's offense cost exceeds that. Dominant drains are undefendable (win-first only).

## Win quality (NOT the problem)
We win 22% (18/82) but at avg 12.9 fan when we do — higher than ddl战士 (10.8), competitive with all.
Conversion quality is fine; WIN RATE / speed is the gap. Draws 2% (the self-play "89% draw / conversion
disease" framing is a confirmed artifact).

## Diagnosis
The ship loses ground by being OUT-PACED to the win, not by dealing in or failing to convert. This is
what test-time search could fix (win more/faster) and what cloning a rank-3 teacher cannot (speed
capped at teacher). Defense and conversion are both data-confirmed NON-levers on the real field.

## 2026-06-13: net-PIMC real-field head-to-head — NEGATIVE (the 15th lever)
Once the memory bug was fixed (pure-numpy, 141MB, v=0612c PIMC=on net=True confirmed in Botzone
debug), net-PIMC played [Claude]aaa vs plain [moyu]distill in the SAME tables (duplicate format,
vs vc1 + ddl战士). Result: 4/4 games net-PIMC #4 (-30) vs plain #1 (+10) — search LOSES by ~40/game
on the real field. So the failure was never just OOM; the search itself degrades play vs strong real
opponents (the fast8 rollout opponent-model underestimates them -> search optimizes a soft world ->
out-paced). Clone-field was a wash (+393/1440g, ~0.65sigma); real-field is a clear loss. Test-time
search (V-search, Q-rerank, greedy/fan/net PIMC, claim-knob) is exhausted. SHIP = plain lad_chunjiandu.
PAPER POINT: a correctly-implemented, memory-safe MCTS-style search can still HURT when the rollout
opponent-model is mis-calibrated to the real field — search quality is bounded by opponent-model fidelity.

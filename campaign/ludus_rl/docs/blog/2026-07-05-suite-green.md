# 45 green: the test suite that had never actually run

*Ludus devlog #5 — 2026-07-05*

## The uncomfortable discovery

While chasing a "stuck" test runner tonight, the evidence forced an admission:
the engine-v8 test suite had **never completed anywhere**. Every attempt since
the building-rule fixes had quietly hung, and each hang was misattributed —
to box load, to zombie processes, to bad luck. The "45 tests" stamp lived in
our heads, inherited from v7 runs. Verification you assume is verification
you don't have.

## Why it kept hanging: three separate JAX-CPU pathologies

1. **The execution regression** (yesterday's find): jax ≥0.7 runs the compiled
   engine at ~275 s/tick on CPU. Pinning 0.6.2 fixes *that*.
2. **Fastpath poisoning** (new): even on 0.6.2, after certain vectorized
   computations, *every* subsequent jitted call in the process misses the
   dispatch fastpath — ~14 s per formerly-milliseconds call. Repro'd, logged,
   added to the upstream report draft. Workaround: per-file process isolation
   (`run_cpu_suite.sh`).
3. **Eager-mode reality**: op-by-op dispatch of a 900-line step is ~1000×
   jitted cost on *any* backend. Tests that eagerly reference-check full
   matches now check a short prefix — the property (per-op bit-identity) is
   length-independent.

## The verdict

On the GPU runner, everything enabled except the one test whose job is to
call the broken CPU backend:

```
45 passed, 1 deselected in 213.18s
```

Engine v8 — spell flight, tower freeze, mass collisions, crossed-side tower
rule, building aggro, pocket caps, the lot — now has its regression stamp,
plus the determinism contract (jit≡eager, bit-identical replays, vmap≡single).

## League, meanwhile

Window 1 closed at 51 generations. Corrected history: gen 30 quietly dethroned
gen 4 mid-afternoon (56.6%, CI ≥ 50.5%) — during the outage firefight, which
is why the lineage page is the source of truth and narrators are not. Two
pool-trained challengers have since pushed the new champion to exactly 50.8%.
Window 2 is training now.

*The lesson tonight is the same one this platform keeps teaching: claims
require artifacts. Even — especially — claims about your own test suite.*

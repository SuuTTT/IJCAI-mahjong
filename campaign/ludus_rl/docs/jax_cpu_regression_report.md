# DRAFT: upstream issue for github.com/jax-ml/jax

*(Draft for the maintainer to file — includes everything reproduced on our box;
verify the minimal repro on a second machine before submitting.)*

**Title**: XLA:CPU executes a jitted program ~50,000× slower after 0.6.2 → 0.7.0
(compile also ~20× slower); persists in 0.10.2

## Summary

A pure-functional game engine (single fused jitted step over int32 arrays,
~900 lines of lax ops: where/segment_sum/gather/scatter, no while_loop, no
callbacks) shows a catastrophic CPU-backend regression between jax 0.6.2 and
0.7.2:

| jax (pip, CPU wheel) | first call (compile+run) | 100 subsequent calls |
|---|---|---|
| 0.4.38 | 14.9 s | 0.53 s |
| 0.5.3 | 14.2 s | 0.47 s |
| 0.6.2 | 13.8 s | 0.75 s |
| 0.7.2 | 330.6 s | did not finish (>900 s) |
| 0.10.2 | ~292 s | ~275 s **per call** |

Same machine (72-core x86_64, Ubuntu 22.04 container), same program, same
inputs. `jax.jit` cache confirmed hot (`_cache_size() == 1`, `JAX_LOG_COMPILES`
shows a single compilation) — this is *execution* time of a compiled
executable, not retracing. The identical program on the CUDA backend runs at
microseconds per call on every version tested.

## Repro

```python
# pip install "jax==0.6.2"  # then repeat with 0.7.2
# git clone https://github.com/SuuTTT/ludus && pip install -e ludus
import os; os.environ["JAX_PLATFORMS"] = "cpu"
import time, jax, jax.numpy as jnp
from boom import engine

step = jax.jit(engine.step)
st = engine.reset(jax.random.PRNGKey(0), None)
acts = jnp.array([[4, 0, 0], [4, 0, 0]], jnp.int32)
t0 = time.time(); st = step(st, acts, None)
jax.block_until_ready(st.u_hp); print("first:", time.time() - t0)
t0 = time.time()
for _ in range(100):
    st = step(st, acts, None)
jax.block_until_ready(st.u_hp); print("100 calls:", time.time() - t0)
```

We can reduce further if useful; candidate suspects in the program: an
unrolled 8-iteration scatter loop, int32-heavy gather/scatter over (96,)
unit arrays, and large boolean masks — but nothing exotic.

## Environment

- jax/jaxlib from PyPI CPU wheels, versions as tabled
- Python 3.12, Ubuntu 22.04 (Docker, unprivileged), 72× x86_64 cores
- Also reproduced under CPU thread contention and on an idle box

## Impact

CPU CI for our project went from ~7 ms/step to ~275 s/step — effectively
unusable; we pin 0.6.2 for CPU testing.

"""Session-wide test setup: jit the engine entry points once.

The game tests call engine.step directly (~40 sites). Op-by-op dispatch of the
900-line step on CPU takes minutes PER CALL; jitted it's milliseconds. Jit and
no-jit are bit-identical (pinned by test_jit_equals_nojit, which grabs its
unjitted reference before this rebind via engine.step.__wrapped__).
"""
import jax

from boom import engine

if not getattr(engine, "_test_jitted", False):
    engine.step = jax.jit(engine.step)
    engine._test_jitted = True

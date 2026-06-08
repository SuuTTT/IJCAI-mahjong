"""
jax_throughput_probe.py — does a vectorized JAX env give the RL throughput leap?
Honest probe (not a full CSM env, which is multi-week): step B parallel games on GPU where each step
is a CNN-equivalent policy forward (40-block-ish conv cost on a 50x4x9 board) + a vectorized state
transition. Measures env-steps/sec at B=1k/4k/16k vs our Python self-play rate (~hundreds of
games/sec total). If JAX gives 100x+ steps/sec, a JAX-native CSM env is worth the multi-week build;
if not, it isn't.

  python3 jax_throughput_probe.py
"""
import time, os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax, jax.numpy as jnp
from jax import random, lax

print("jax", jax.__version__, "devices", jax.devices(), flush=True)
C, BLOCKS = 128, 40
key = random.PRNGKey(0)

def init_params(key):
    ks = random.split(key, 2 * BLOCKS + 2)
    p = {'stem': random.normal(ks[0], (C, 50, 3, 3)) * 0.05}
    for i in range(BLOCKS):
        p[f'c{i}a'] = random.normal(ks[2*i+1], (C, C, 3, 3)) * 0.05
        p[f'c{i}b'] = random.normal(ks[2*i+2], (C, C, 3, 3)) * 0.05
    p['fc'] = random.normal(ks[-1], (C*4*9, 235)) * 0.02
    return p

def conv(x, w):  # x (B,Cin,4,9) w (Cout,Cin,3,3) same-pad
    return lax.conv_general_dilated(x, w, (1,1), 'SAME',
        dimension_numbers=('NCHW','OIHW','NCHW'))

def policy(p, board):                          # board (B,50,4,9) -> logits (B,235)
    h = jax.nn.relu(conv(board, p['stem']))
    for i in range(BLOCKS):
        y = jax.nn.relu(conv(h, p[f'c{i}a'])); y = conv(y, p[f'c{i}b']); h = jax.nn.relu(h + y)
    return h.reshape(h.shape[0], -1) @ p['fc']

@jax.jit
def step(p, board, key):                       # one env step: policy forward + vectorized transition
    logits = policy(p, board)
    act = jnp.argmax(logits, axis=1)            # greedy discard (placeholder transition)
    # vectorized "transition": roll the board planes by the chosen action (stand-in for state update)
    shift = (act % 9).astype(jnp.int32)
    board = jax.vmap(lambda b, s: jnp.roll(b, s, axis=-1))(board, shift)
    return board, act

def bench(B, iters=30):
    p = init_params(key)
    board = jnp.zeros((B, 50, 4, 9), jnp.float32)
    k = key
    board, _ = step(p, board, k); board.block_until_ready()   # compile
    t = time.time()
    for _ in range(iters):
        k, sub = random.split(k)
        board, act = step(p, board, sub)
    act.block_until_ready()
    dt = time.time() - t
    sps = B * iters / dt
    print(f"B={B:6d}: {sps:,.0f} env-steps/sec  ({iters} iters {dt:.2f}s)", flush=True)
    return sps

if __name__ == '__main__':
    best = 0
    for B in (1024, 4096, 16384):
        try: best = max(best, bench(B))
        except Exception as e: print(f"B={B} failed: {str(e)[:80]}", flush=True)
    # a CSM game ~ 60-100 steps; steps/sec / 80 ~= games/sec
    print(f"\nPEAK ~{best:,.0f} steps/sec ~= {best/80:,.0f} games/sec (JAX-GPU).", flush=True)
    print("Compare: our Python self-play ~ a few hundred games/sec across ~25 cores.", flush=True)
    print("VERDICT: if games/sec >> 1000, a JAX-native CSM env is worth the multi-week build.", flush=True)

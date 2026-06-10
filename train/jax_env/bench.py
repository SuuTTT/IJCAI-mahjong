"""
bench.py — throughput benchmark for the JAX CSM env: does a vectorized GPU env + a batched small
policy net break the ~246 g/s CPU self-play wall? Measures env-steps/s and games/s at several batch
sizes, with a small conv policy forward each step (the realistic self-play cost).

  python3 bench.py            # auto batch sweep
  python3 bench.py 8192 30    # B=8192, 30-step rollout
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp
from jax import lax, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_env as E

print("jax", jax.__version__, "devices", jax.devices(), flush=True)
CH, NL = 64, 4                       # small policy net: NL conv layers, CH channels (the FAST self-play net)


def init_policy(key):
    ks = random.split(key, NL + 2)
    p = {"in": random.normal(ks[0], (CH, 8, 3)) * 0.1}     # (Cout, Cin=8 planes, k=3) 1-D conv over 34 tiles
    for i in range(NL):
        p[f"c{i}"] = random.normal(ks[i + 1], (CH, CH, 3)) * (0.1 / CH ** 0.5)
    p["out"] = random.normal(ks[-1], (CH * E.NT, E.NT)) * 0.02
    return p


def policy_logits(p, obs):             # obs (B,8,34) -> (B,34) discard logits
    def conv1d(x, w):                  # x (B,Cin,34) w (Cout,Cin,3) same-pad
        return lax.conv_general_dilated(x, w, (1,), "SAME",
                                        dimension_numbers=("NCH", "OIH", "NCH"))
    h = jax.nn.relu(conv1d(obs, p["in"]))
    for i in range(NL):
        h = jax.nn.relu(h + conv1d(h, p[f"c{i}"]))
    return h.reshape(h.shape[0], -1) @ p["out"]


def make_rollout(pp):
    @jax.jit
    def roll(key, state, n=jnp.int32(30)):
        def body(carry, _):
            state, key = carry
            key, sk = random.split(key)
            mask = E.legal_discards(state)
            lg = jnp.where(mask, policy_logits(pp, E.encode_obs(state)), -1e9)
            a = random.categorical(sk, lg, axis=1)
            state, r, d = E.step(state, a)
            return (state, key), r
        (state, key), rs = lax.scan(body, (state, key), None, length=30)
        return state, rs.sum(0)
    return roll


def bench(B, steps=30, reps=3):
    key = random.PRNGKey(0)
    pp = init_policy(key)
    roll = make_rollout(pp)
    state = E.init_state(random.PRNGKey(1), B)
    s, r = roll(key, state); s["ptr"].block_until_ready()        # compile
    t0 = time.time()
    for i in range(reps):
        state = E.init_state(random.PRNGKey(i + 2), B)
        s, r = roll(random.PRNGKey(i), state); s["ptr"].block_until_ready()
    dt = (time.time() - t0) / reps
    env_steps = B * steps
    print(f"B={B:6d} steps={steps}: {dt*1000:7.1f} ms  ->  {env_steps/dt:12,.0f} env-steps/s  "
          f"(~{B/dt:9,.0f} partial-games/s @ {steps} steps)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        for B in (256, 1024, 4096, 16384):
            bench(B)

"""Flax port of the kdens3 MCR champion policy (BN-free fused ResNet).

This is a NEW file (does not touch mahjong/jax_env.py or the torch model). It
ports ``baselines/mahjong_torch_model.py:PolicyNet`` / ``mcr_champion/
numpy_resfused.py:NumpyResFused`` into flax so the champion forward runs on-GPU
inside the vmappable JAX MCR env (``mahjong/jax_env.py``).

Architecture (identical to PolicyNet / ResFused):
  stem  Conv2d(38 -> 128, 3x3, pad1) + ReLU
  body  40 x [ Conv3x3(128->128)+ReLU, Conv3x3(128->128), +residual, ReLU ]
  foot  flatten(128*4*9=4608) -> Linear(4608->512) -> ReLU -> Linear(512->235)
  then a HARD legal mask: where(mask>0, logits, -1e9)   (matches numpy_resfused)

The input observation is (B, 38, 4, 9) == NCHW with C=38, H=4, W=9. We run the
convs with ``lax.conv_general_dilated`` in native NCHW / OIHW layout so the torch
weights load with ZERO transposes and the pre-foot flatten is naturally
channel-major (c*36 + h*9 + w) exactly like torch's ``nn.Flatten`` on NCHW.

Weights: ``kdens_s{0,1,2}_fp16.npz`` (keys ``stem.weight/bias``,
``body.{i}.c1|c2.weight/bias``, ``foot.1|3.weight/bias``, ``meta_blocks``).
``load_params`` maps those torch key names into a flax param pytree.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn

N_IN = 38
CH = 128
FLAT = CH * 4 * 9        # 4608
N_ACT = 235
BLOCKS = 40


def load_params(npz_path):
    """Read a champion .npz into a flax param pytree (torch key names -> flax
    tree). Weights are kept in fp16 on disk; we return fp32 arrays (the numpy
    champion also up-casts fp16->fp32 before the forward)."""
    z = np.load(npz_path)
    g = lambda k: jnp.asarray(z[k], dtype=jnp.float32)
    blocks = int(z["meta_blocks"][0]) if "meta_blocks" in z.files else BLOCKS
    params = {
        "stem_w": g("stem.weight"),          # (128,38,3,3) OIHW
        "stem_b": g("stem.bias"),            # (128,)
        "foot1_w": g("foot.1.weight"),       # (512,4608) torch (out,in)
        "foot1_b": g("foot.1.bias"),         # (512,)
        "foot3_w": g("foot.3.weight"),       # (235,512)
        "foot3_b": g("foot.3.bias"),         # (235,)
    }
    for i in range(blocks):
        params[f"body_{i}_c1_w"] = g(f"body.{i}.c1.weight")
        params[f"body_{i}_c1_b"] = g(f"body.{i}.c1.bias")
        params[f"body_{i}_c2_w"] = g(f"body.{i}.c2.weight")
        params[f"body_{i}_c2_b"] = g(f"body.{i}.c2.bias")
    return {"params": params, "blocks": blocks}


def _conv(x, w, b):
    """NCHW conv, OIHW kernel, 3x3 pad-1 stride-1 (matches torch/numpy)."""
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((1, 1), (1, 1)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b[None, :, None, None]


def forward(bundle, obs, mask, dtype=jnp.float32):
    """Masked logits for a single fused net.
      obs  (B, 38, 4, 9) float
      mask (B, 235) bool/float (1=legal)
    Returns (B, 235) logits with a hard legal mask (-1e9 on illegal)."""
    p = bundle["params"]
    blocks = bundle["blocks"]
    cast = lambda t: jax.tree_util.tree_map(lambda a: a.astype(dtype), t)
    p = cast(p)
    x = obs.astype(dtype)
    x = nn.relu(_conv(x, p["stem_w"], p["stem_b"]))
    for i in range(blocks):
        y = nn.relu(_conv(x, p[f"body_{i}_c1_w"], p[f"body_{i}_c1_b"]))
        y = _conv(y, p[f"body_{i}_c2_w"], p[f"body_{i}_c2_b"])
        x = nn.relu(x + y)
    B = x.shape[0]
    f = x.reshape(B, -1)                                  # channel-major flatten
    h = nn.relu(f @ p["foot1_w"].T + p["foot1_b"])
    out = (h @ p["foot3_w"].T + p["foot3_b"]).astype(jnp.float32)
    m = mask.astype(jnp.float32)
    return jnp.where(m > 0, out, -1e9)


def ensemble_logprobs(bundles, obs, mask, dtype=jnp.float32):
    """kdens3: average SOFTMAX PROBS over the legal set across nets, then return
    log-probs (matches mcr_champion/ensemble_infer.py). (B,235)."""
    m = mask.astype(jnp.float32)
    acc = jnp.zeros(mask.shape[:1] + (N_ACT,), jnp.float32)
    for b in bundles:
        lg = forward(b, obs, mask, dtype)
        lg = jnp.where(m > 0, lg, -1e30)
        lg = lg - lg.max(axis=1, keepdims=True)
        pe = jnp.exp(lg) * (m > 0)
        s = pe.sum(axis=1, keepdims=True)
        pe = jnp.where(s > 0, pe / s, m / jnp.maximum(1.0, m.sum(axis=1, keepdims=True)))
        acc = acc + pe
    avg = acc / len(bundles)
    return jnp.log(jnp.where(avg > 0, avg, 1e-30))


def make_forward(bundle, dtype=jnp.float32):
    """Return a jitted single-net masked-logit forward."""
    return jax.jit(partial(forward, bundle, dtype=dtype))

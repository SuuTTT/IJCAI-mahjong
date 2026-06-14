"""
resnet_jax.py — JAX forward of the deploy ResFused (40-block/128ch) for WARM-STARTING the RL policy
from lad_chunjiandu. Loads the validated numpy weights (cnn_lad_chunjiandu.npz, byte-exact vs torch)
into a JAX pytree and runs the same forward, batched on GPU. This is what lets RL start from a
competent policy instead of random (the Tjong "SL first" fix).

Layout (matches numpy_resfused / model_resfused): stem Conv3x3(38->C)+ReLU; `blocks` of
[Conv3x3+ReLU, Conv3x3, +residual, ReLU]; foot [flatten, Linear(C*36->512), ReLU, Linear(512->235)];
masked with log(mask). Input obs (B,38,4,9), mask (B,235) -> logits (B,235).
"""
import numpy as np
import jax, jax.numpy as jnp
from jax import lax


def load_params(npz_path):
    z = np.load(npz_path)
    p = {k: jnp.asarray(z[k].astype(np.float32)) for k in z.files if k != 'meta_blocks'}
    p['_blocks'] = int(z['meta_blocks'][0])
    return p


def _conv(x, w, b):
    # x (B,Cin,4,9), w (Cout,Cin,3,3) -> (B,Cout,4,9), SAME pad, stride 1
    y = lax.conv_general_dilated(x, w, (1, 1), 'SAME',
                                 dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    return y + b[None, :, None, None]


def forward_feats(p, obs, nb=None):
    """obs (B,38,4,9) -> (full_logits (B,235), h (B,512)). h = penultimate features (after foot.1+ReLU),
    reused by a value head for warm-started RL. full_logits[:,2:36] are the 34 Play/discard logits.
    nb = number of residual blocks; defaults to p['_blocks'] (pass explicitly when _blocks is removed)."""
    if nb is None:
        nb = int(p['_blocks'])
    x = jax.nn.relu(_conv(obs, p['stem.weight'], p['stem.bias']))
    for i in range(nb):
        y = jax.nn.relu(_conv(x, p[f'body.{i}.c1.weight'], p[f'body.{i}.c1.bias']))
        y = _conv(y, p[f'body.{i}.c2.weight'], p[f'body.{i}.c2.bias'])
        x = jax.nn.relu(x + y)
    f = x.reshape(x.shape[0], -1)                                  # flatten (B, C*4*9)
    h = jax.nn.relu(f @ p['foot.1.weight'].T + p['foot.1.bias'])   # (B,512)
    out = h @ p['foot.3.weight'].T + p['foot.3.bias']              # (B,235)
    return out, h


def forward(p, obs, mask):
    """obs (B,38,4,9) float32, mask (B,235) -> masked logits (B,235)."""
    out, _ = forward_feats(p, obs)
    m = mask.astype(jnp.float32)
    return jnp.where(m > 0, out, -1e9)

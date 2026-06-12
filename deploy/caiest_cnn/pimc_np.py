# pimc_np.py — pure-NumPy nets for the memory-fit PIMC rollout (ZERO torch -> ~91MB base instead of
# ~488MB, so the 3-net search fits under Botzone's 512MB). Reuses numpy_resfused conv for the fast
# policy; adds a numpy value-net forward (vbig: stem -> fused blocks -> GAP -> 128->64->1 head).
import numpy as np
from numpy_resfused import NumpyResFused, _conv3x3, _relu


class NumpyValueNet:
    """Numpy forward of value_search.ValueNet: stem Conv -> `blocks` fused blocks -> global-average
    pool -> MLP(channels->128->64->1). Returns a scalar value (predicted final duplicate score)."""
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        self.blocks = int(self.w['meta_blocks'][0])

    def v(self, obs):
        x = _relu(_conv3x3(obs.astype(np.float32), self.w['stem.weight'], self.w['stem.bias']))
        for i in range(self.blocks):
            y = _relu(_conv3x3(x, self.w[f'body.{i}.c1.weight'], self.w[f'body.{i}.c1.bias']))
            y = _conv3x3(y, self.w[f'body.{i}.c2.weight'], self.w[f'body.{i}.c2.bias'])
            x = _relu(x + y)
        g = x.mean(axis=(1, 2))                                   # GAP -> (C,)
        h = _relu(g @ self.w['head.0.weight'].T + self.w['head.0.bias'])
        h = _relu(h @ self.w['head.2.weight'].T + self.w['head.2.bias'])
        return float(h @ self.w['head.4.weight'].T + self.w['head.4.bias'])


class NumpyPolicy:
    """Thin wrapper: legal-discard argmax from a NumpyResFused policy net."""
    def __init__(self, npz_path):
        self.net = NumpyResFused(npz_path)

    def discard(self, obs, mask):
        lg = self.net.logits(obs, mask)
        return int(lg[2:36].argmax())                            # Play block, returns tile index 0..33

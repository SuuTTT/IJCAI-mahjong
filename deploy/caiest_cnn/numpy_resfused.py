"""
numpy_resfused.py — PURE NUMPY forward for the fused ResNet (zero torch dependency).
Research's "most reliable" deploy path: bypass torch entirely so no PyTorch-version mismatch can
ever crash on Botzone's legacy runtime. Weights come from an .npz (converted once from the fused
.pkl with convert()). Numerically identical to the torch ResFused in eval mode.

Layout (ResFused): stem Conv3x3(38->C) + ReLU; then `blocks` of [Conv3x3(C->C)+ReLU, Conv3x3(C->C),
+residual, ReLU]; foot [flatten, Linear(C*4*9->512), ReLU, Linear(512->235)]; + log(mask).
"""
import numpy as np

def _conv3x3(x, W, b):
    """x (Cin,4,9), W (Cout,Cin,3,3), b (Cout,) -> (Cout,4,9). Pad 1, stride 1."""
    Cin, H, Wd = x.shape; Cout = W.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((H * Wd, Cin * 9), np.float32)
    for i in range(H):
        for j in range(Wd):
            cols[i * Wd + j] = xp[:, i:i + 3, j:j + 3].reshape(-1)
    out = cols @ W.reshape(Cout, -1).T + b              # (H*W, Cout)
    return out.T.reshape(Cout, H, Wd)

def _relu(x): return np.maximum(x, 0.0)

class NumpyResFused:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        self.blocks = int(self.w['meta_blocks'][0])

    def logits(self, obs, mask):
        """obs (38,4,9) float, mask (235,) -> logits (235,) with log-mask applied."""
        x = _relu(_conv3x3(obs.astype(np.float32), self.w['stem.weight'], self.w['stem.bias']))
        for i in range(self.blocks):
            y = _relu(_conv3x3(x, self.w[f'body.{i}.c1.weight'], self.w[f'body.{i}.c1.bias']))
            y = _conv3x3(y, self.w[f'body.{i}.c2.weight'], self.w[f'body.{i}.c2.bias'])
            x = _relu(x + y)
        f = x.reshape(-1)                                # flatten (C*4*9)
        h = _relu(f @ self.w['foot.1.weight'].T + self.w['foot.1.bias'])
        out = h @ self.w['foot.3.weight'].T + self.w['foot.3.bias']
        m = mask.astype(np.float32)
        # hard mask (match torch's -inf): the old log(1e-30)=-69 penalty was insufficient when a raw
        # logit exceeded 69 -> a masked action (e.g. illegal Hu) could survive argmax -> phantom HU.
        return np.where(m > 0, out, -1e9)

def convert(pkl_path, npz_path):
    """One-time: torch fused .pkl -> .npz of numpy weights (run locally, ships the .npz)."""
    import torch
    sd = torch.load(pkl_path, map_location='cpu')
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.') and k.endswith('.c1.weight'))
    out = {k: v.numpy() for k, v in sd.items()}
    out['meta_blocks'] = np.array([blocks], np.int64)
    np.savez_compressed(npz_path, **out)
    print(f"converted {pkl_path} -> {npz_path} ({blocks} blocks)")

if __name__ == '__main__':
    import sys
    convert(sys.argv[1], sys.argv[2])

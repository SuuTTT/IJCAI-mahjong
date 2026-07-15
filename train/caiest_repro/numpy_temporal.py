"""
numpy_temporal.py — PURE NUMPY forward for the TEMPORAL net (CNN + GRU over ordered discards).
Zero torch dependency (Botzone py36 numpy-only runtime). Weights from an .npz produced by
convert_temporal.py, which FOLDS every BatchNorm into its preceding conv (conv has bias=False),
so the CNN branch reduces to the exact conv-relu-conv-add-relu ResFused structure.

CNN branch: stem Conv3x3(38->C)+ReLU; `blocks` of [Conv3x3+ReLU, Conv3x3, +residual, ReLU];
            flatten(C*4*9) -> Linear(->512) -> ReLU  => (512,)
GRU branch: embed[seq] (L,emb); PyTorch GRU (1 layer, gates r,z,n); take last hidden => (gru,)
head: Linear(512+gru -> 235); + hard mask (illegal -> -1e9).
Numerically matches the torch TemporalNet in eval mode (BN uses running stats).
"""
import numpy as np


def _conv3x3(x, W, b):
    """x (Cin,4,9), W (Cout,Cin,3,3), b (Cout,) -> (Cout,4,9). Pad 1, stride 1 (im2col)."""
    Cin, H, Wd = x.shape
    Cout = W.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((H * Wd, Cin * 9), np.float32)
    for i in range(H):
        for j in range(Wd):
            cols[i * Wd + j] = xp[:, i:i + 3, j:j + 3].reshape(-1)
    out = cols @ W.reshape(Cout, -1).T + b
    return out.T.reshape(Cout, H, Wd)


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyTemporal:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        self.blocks = int(self.w['meta_blocks'][0])
        self.L = int(self.w['meta_L'][0])
        self.PAD = int(self.w['meta_pad'][0])
        self.gru_dim = self.w['gru.weight_hh_l0'].shape[1]

    def _cnn(self, obs):
        w = self.w
        x = _relu(_conv3x3(obs.astype(np.float32), w['stem.weight'], w['stem.bias']))
        for i in range(self.blocks):
            y = _relu(_conv3x3(x, w[f'body.{i}.c1.weight'], w[f'body.{i}.c1.bias']))
            y = _conv3x3(y, w[f'body.{i}.c2.weight'], w[f'body.{i}.c2.bias'])
            x = _relu(x + y)
        f = x.reshape(-1)
        return _relu(f @ w['fc.weight'].T + w['fc.bias'])          # (512,)

    def _gru_last(self, seq):
        """seq: (L,) int token ids. Returns last hidden (gru_dim,). Matches PyTorch nn.GRU."""
        w = self.w
        emb = w['embed.weight']                                    # (VOCAB, emb)
        Wih = w['gru.weight_ih_l0']; Whh = w['gru.weight_hh_l0']
        bih = w['gru.bias_ih_l0']; bhh = w['gru.bias_hh_l0']
        H = self.gru_dim
        h = np.zeros(H, np.float32)
        seq = np.asarray(seq).astype(np.int64)
        for t in seq:                                              # left-padded; process all L steps
            x = emb[t]                                             # (emb,)
            gi = Wih @ x + bih                                     # (3H,)
            gh = Whh @ h + bhh                                     # (3H,)
            i_r = gi[:H]; i_z = gi[H:2 * H]; i_n = gi[2 * H:]
            h_r = gh[:H]; h_z = gh[H:2 * H]; h_n = gh[2 * H:]
            r = _sigmoid(i_r + h_r)
            zg = _sigmoid(i_z + h_z)
            n = np.tanh(i_n + r * h_n)
            h = (1.0 - zg) * n + zg * h
        return h

    def logits(self, obs, mask, seq):
        """obs (38,4,9), mask (235,), seq (L,) int -> masked logits (235,)."""
        cnn = self._cnn(obs)                                       # (512,)
        hlast = self._gru_last(seq)                                # (gru,)
        feat = np.concatenate([cnn, hlast])
        out = feat @ self.w['head.weight'].T + self.w['head.bias']
        m = mask.astype(np.float32)
        return np.where(m > 0, out, -1e9)

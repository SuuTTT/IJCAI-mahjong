"""
numpy_infer.py — NumpyMLP standalone (no PyTorch dependency).
Used by ml_bot.py at serve time. Fast startup (~100ms vs 2s with torch).
"""

import numpy as np

OBS_DIM = 240
ACT_DIM = 235


def _layer_norm(x, g, b, eps=1e-5):
    mu  = x.mean()
    std = np.sqrt(x.var() + eps)
    return (x - mu) / std * g + b


class NumpyMLP:
    """
    Faithful numpy forward pass for MahjongNet (ResBlock architecture).
    Load from .npz exported by MahjongNet.export_numpy().
    """

    def __init__(self, path: str):
        d = np.load(path)
        self.stem_w  = d["stem_w"];   self.stem_b  = d["stem_b"]
        self.stem_g  = d["stem_ln_g"]; self.stem_bg = d["stem_ln_b"]
        self.blocks  = []
        i = 0
        while f"block{i}_w1" in d:
            self.blocks.append({
                "w1": d[f"block{i}_w1"], "b1": d[f"block{i}_b1"],
                "g1": d[f"block{i}_ln1_g"], "bg1": d[f"block{i}_ln1_b"],
                "w2": d[f"block{i}_w2"], "b2": d[f"block{i}_b2"],
                "g2": d[f"block{i}_ln2_g"], "bg2": d[f"block{i}_ln2_b"],
            })
            i += 1
        self.policy_w  = d["policy_w"]; self.policy_b  = d["policy_b"]
        self.value_w0  = d.get("value_w0"); self.value_b0  = d.get("value_b0")
        self.value_w   = d["value_w"];  self.value_b   = d["value_b"]

    def forward(self, obs: np.ndarray, mask: np.ndarray = None):
        x = obs.astype(np.float32) / 255.0
        x = np.maximum(0, _layer_norm(x @ self.stem_w.T + self.stem_b,
                                      self.stem_g, self.stem_bg))
        for blk in self.blocks:
            h = _layer_norm(x @ blk["w1"].T + blk["b1"], blk["g1"], blk["bg1"])
            h = np.maximum(0, h)
            h = _layer_norm(h @ blk["w2"].T + blk["b2"], blk["g2"], blk["bg2"])
            x = np.maximum(0, x + h)
        logits = x @ self.policy_w.T + self.policy_b
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        if self.value_w0 is not None:
            h = np.maximum(0, x @ self.value_w0.T + self.value_b0)
            value = float(np.tanh(h @ self.value_w.T + self.value_b))
        else:
            value = 0.0
        return probs, value

    def best_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        probs, _ = self.forward(obs, mask)
        return int(np.argmax(probs))

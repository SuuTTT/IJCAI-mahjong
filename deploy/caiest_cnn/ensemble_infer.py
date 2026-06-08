"""
ensemble_infer.py — inference-time MIXTURE of diverse fused nets (pure NumPy, torch-free).
Exploits the campaign's non-transitivity finding: distill100b / V1 / champ2025 beat each other on
different opponents, so averaging their per-decision distributions decorrelates blunders and is more
robust against a varied field than any single model — at NumPy memory cost (~50MB/model, fits 512MB).

Averages SOFTMAX PROBABILITIES over the legal action set (arithmetic mean), then argmax. Masking is
identical across models for a given decision, so this is well-defined.

  ENSEMBLE_NPZS=/path/a.npz,/path/b.npz,/path/c.npz   # set in env; bot uses this if present
"""
import numpy as np
from numpy_resfused import NumpyResFused

class Ensemble:
    def __init__(self, npz_paths):
        self.models = [NumpyResFused(p) for p in npz_paths]
        self.n = len(self.models)

    def logits(self, obs, mask):
        """Return averaged log-probabilities (shape 235) so callers can argmax as usual."""
        acc = None
        m = mask.astype(np.float32)
        for mdl in self.models:
            lg = mdl.logits(obs, mask)               # already includes log-mask
            lg = np.where(m > 0, lg, -1e30)
            lg = lg - lg.max()                       # stabilize
            p = np.exp(lg) * (m > 0)
            s = p.sum()
            p = p / s if s > 0 else (m / max(1.0, m.sum()))
            acc = p if acc is None else acc + p
        avg = acc / self.n
        return np.log(np.where(avg > 0, avg, 1e-30))

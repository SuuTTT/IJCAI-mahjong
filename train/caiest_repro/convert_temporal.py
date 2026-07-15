"""
convert_temporal.py — fold BatchNorm into convs and export the TemporalNet state_dict to a
pure-numpy .npz consumable by numpy_temporal.NumpyTemporal (zero torch at deploy).
Usage: python3 convert_temporal.py <temporal.pkl> <out.npz>
"""
import sys
import numpy as np
import torch

EPS = 1e-5
L = 48
PAD = 136


def fold(conv_w, gamma, beta, mean, var):
    """conv (Cout,Cin,3,3) with NO bias, followed by BN -> folded (W', b')."""
    scale = gamma / np.sqrt(var + EPS)
    Wf = conv_w * scale[:, None, None, None]
    bf = beta - scale * mean
    return Wf.astype(np.float32), bf.astype(np.float32)


def main():
    pkl, out = sys.argv[1], sys.argv[2]
    sd = torch.load(pkl, map_location='cpu')
    g = {k: v.numpy() for k, v in sd.items()}
    blocks = 1 + max(int(k.split('.')[1]) for k in g if k.startswith('body.') and k.endswith('.c1.weight'))
    o = {}
    # stem: conv stem.0.weight + BN stem.1
    w, b = fold(g['stem.0.weight'], g['stem.1.weight'], g['stem.1.bias'],
                g['stem.1.running_mean'], g['stem.1.running_var'])
    o['stem.weight'] = w; o['stem.bias'] = b
    for i in range(blocks):
        w1, b1 = fold(g[f'body.{i}.c1.weight'], g[f'body.{i}.b1.weight'], g[f'body.{i}.b1.bias'],
                      g[f'body.{i}.b1.running_mean'], g[f'body.{i}.b1.running_var'])
        w2, b2 = fold(g[f'body.{i}.c2.weight'], g[f'body.{i}.b2.weight'], g[f'body.{i}.b2.bias'],
                      g[f'body.{i}.b2.running_mean'], g[f'body.{i}.b2.running_var'])
        o[f'body.{i}.c1.weight'] = w1; o[f'body.{i}.c1.bias'] = b1
        o[f'body.{i}.c2.weight'] = w2; o[f'body.{i}.c2.bias'] = b2
    o['fc.weight'] = g['cnn_fc.1.weight'].astype(np.float32)
    o['fc.bias'] = g['cnn_fc.1.bias'].astype(np.float32)
    o['embed.weight'] = g['embed.weight'].astype(np.float32)
    for k in ('gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0'):
        o[k] = g[k].astype(np.float32)
    o['head.weight'] = g['head.weight'].astype(np.float32)
    o['head.bias'] = g['head.bias'].astype(np.float32)
    o['meta_blocks'] = np.array([blocks], np.int64)
    o['meta_L'] = np.array([L], np.int64)
    o['meta_pad'] = np.array([PAD], np.int64)
    np.savez_compressed(out, **o)
    import os
    print("converted %s -> %s  blocks=%d  %.1f MB" % (pkl, out, blocks, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()

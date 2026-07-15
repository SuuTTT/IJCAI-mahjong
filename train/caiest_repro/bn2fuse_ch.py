"""
bn2fuse_ch.py — channel-aware BN->fused folding (generalizes bn2fuse.py to any channel count).
Usage: python3 bn2fuse_ch.py <nonfused_in.pkl> <fused_out.pkl>
"""
import sys, torch
from models_explore import build

def main():
    fin, fout = sys.argv[1], sys.argv[2]
    sd = torch.load(fin, map_location='cpu')
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))
    channels = sd['stem.0.weight'].shape[0]
    eps = 1e-5
    out = {}
    def fold(conv_w, bn_prefix):
        g = sd[bn_prefix + '.weight']; b = sd[bn_prefix + '.bias']
        m = sd[bn_prefix + '.running_mean']; v = sd[bn_prefix + '.running_var']
        s = g / torch.sqrt(v + eps)
        return conv_w * s.reshape(-1, 1, 1, 1), b - g * m / torch.sqrt(v + eps)
    w, b = fold(sd['stem.0.weight'], 'stem.1')
    out['stem.weight'] = w; out['stem.bias'] = b
    for i in range(blocks):
        w1, b1 = fold(sd[f'body.{i}.c1.weight'], f'body.{i}.b1')
        out[f'body.{i}.c1.weight'] = w1; out[f'body.{i}.c1.bias'] = b1
        w2, b2 = fold(sd[f'body.{i}.c2.weight'], f'body.{i}.b2')
        out[f'body.{i}.c2.weight'] = w2; out[f'body.{i}.c2.bias'] = b2
    fnet = build('resbn_fused', channels=channels, blocks=blocks)
    tsd = fnet.state_dict()
    for k, v in sd.items():
        if k in tsd and k not in out and tsd[k].shape == v.shape:
            out[k] = v
    fnet.load_state_dict(out, strict=True)
    torch.save(fnet.state_dict(), fout, _use_new_zipfile_serialization=False)
    nnet = build('resbn', channels=channels, blocks=blocks); nnet.load_state_dict(sd); nnet.eval(); fnet.eval()
    d = {'is_training': False, 'obs': {'observation': torch.zeros(2, 38, 4, 9).normal_(), 'action_mask': torch.ones(2, 235)}}
    with torch.no_grad():
        diff = (nnet(d) - fnet(d)).abs().max().item()
    print(f"folded {fin} -> {fout} (channels={channels}, blocks={blocks})  max|nonfused-fused|={diff:.2e}")

if __name__ == '__main__':
    main()

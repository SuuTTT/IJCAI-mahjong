"""
bn2fuse.py — fold a NON-fused ResBNCNN (conv bias=False + BatchNorm) into the FUSED resbn_fused
arch (conv with bias, BN removed) for deploy/bench. Inverse of fuse2bn.py. Standard BN folding:
  w_fused = w_conv * gamma/sqrt(var+eps)   (per out-channel)
  b_fused = beta - gamma*mean/sqrt(var+eps)
Needed to gauntlet/deploy the RL-league output (which trains the non-fused arch) through the
deploy bot (resbn_fused, torch-1.4-safe). Usage: python3 bn2fuse.py <nonfused_in.pkl> <fused_out.pkl>
"""
import sys, torch
from models_explore import build

def main():
    fin, fout = sys.argv[1], sys.argv[2]
    sd = torch.load(fin, map_location='cpu')
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))
    eps = 1e-5
    out = {}
    def fold(conv_w, bn_prefix):
        g = sd[bn_prefix + '.weight']; b = sd[bn_prefix + '.bias']
        m = sd[bn_prefix + '.running_mean']; v = sd[bn_prefix + '.running_var']
        s = g / torch.sqrt(v + eps)
        return conv_w * s.reshape(-1, 1, 1, 1), b - g * m / torch.sqrt(v + eps)
    # stem: conv (stem.0.weight) + BN (stem.1) -> fused stem.weight/bias
    w, b = fold(sd['stem.0.weight'], 'stem.1')
    out['stem.weight'] = w; out['stem.bias'] = b
    for i in range(blocks):
        w1, b1 = fold(sd[f'body.{i}.c1.weight'], f'body.{i}.b1')
        out[f'body.{i}.c1.weight'] = w1; out[f'body.{i}.c1.bias'] = b1
        w2, b2 = fold(sd[f'body.{i}.c2.weight'], f'body.{i}.b2')
        out[f'body.{i}.c2.weight'] = w2; out[f'body.{i}.c2.bias'] = b2
    fnet = build('resbn_fused', channels=128, blocks=blocks)
    tsd = fnet.state_dict()
    for k, v in sd.items():                       # copy foot/head + anything matching
        if k in tsd and k not in out and tsd[k].shape == v.shape:
            out[k] = v
    fnet.load_state_dict(out, strict=True)
    torch.save(fnet.state_dict(), fout, _use_new_zipfile_serialization=False)
    # sanity vs the non-fused source
    nnet = build('resbn', channels=128, blocks=blocks); nnet.load_state_dict(sd); nnet.eval(); fnet.eval()
    d = {'is_training': False, 'obs': {'observation': torch.zeros(2, 38, 4, 9).normal_(), 'action_mask': torch.ones(2, 235)}}
    with torch.no_grad():
        diff = (nnet(d) - fnet(d)).abs().max().item()
    print(f"folded {fin} -> {fout} (blocks={blocks})  max|nonfused-fused|={diff:.2e}")

if __name__ == '__main__':
    main()

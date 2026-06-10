"""
fuse2bn.py — convert a FUSED model (resbn_fused: conv+bias, no BN) back into a NON-fused ResBNCNN
state_dict (conv bias=False + identity BatchNorm that absorbs the fused conv bias). Numerically
exact (BN with running_mean=0, running_var=1-eps, weight=1, bias=fused_bias -> y = x + bias).
Needed because the data-loss left only the fused distill100b/lad_chunjiandu; RL (rl_league/rl_curriculum)
trains the NON-fused arch. Usage: python3 fuse2bn.py <fused_in.pkl> <nonfused_out.pkl> [blocks]
"""
import sys, torch
from models_explore import build

def main():
    fin, fout = sys.argv[1], sys.argv[2]
    eps = 1e-5
    fsd = torch.load(fin, map_location='cpu')
    # auto-detect block count from the fused sd (body.<i>.*) unless given
    if len(sys.argv) > 3:
        blocks = int(sys.argv[3])
    else:
        blocks = 1 + max(int(k.split('.')[1]) for k in fsd if k.startswith('body.'))
    print(f"blocks={blocks}")
    net = build('resbn', channels=128, blocks=blocks)   # non-fused ResBNCNN
    tsd = net.state_dict()
    out = {}
    def set_bn(prefix, bias):
        out[prefix + '.weight'] = torch.ones_like(bias)
        out[prefix + '.bias'] = bias.clone()
        out[prefix + '.running_mean'] = torch.zeros_like(bias)
        out[prefix + '.running_var'] = torch.full_like(bias, 1.0 - eps)
        out[prefix + '.num_batches_tracked'] = torch.tensor(0)
    # stem: fused stem.{weight,bias} -> non-fused stem.0.weight + stem.1(BN)
    out['stem.0.weight'] = fsd['stem.weight']
    set_bn('stem.1', fsd['stem.bias'])
    # body blocks: c1/c2 conv weights kept; biases -> b1/b2 BN
    for i in range(blocks):
        out[f'body.{i}.c1.weight'] = fsd[f'body.{i}.c1.weight']
        set_bn(f'body.{i}.b1', fsd[f'body.{i}.c1.bias'])
        out[f'body.{i}.c2.weight'] = fsd[f'body.{i}.c2.weight']
        set_bn(f'body.{i}.b2', fsd[f'body.{i}.c2.bias'])
    # foot/head + anything else: copy by matching name+shape
    for k, v in fsd.items():
        if k in tsd and k not in out and tsd[k].shape == v.shape:
            out[k] = v
    missing = [k for k in tsd if k not in out]
    if missing:
        print(f"WARN unmapped target keys: {missing[:5]} (+{len(missing)-5} more)" if len(missing) > 5 else f"WARN unmapped: {missing}")
    net.load_state_dict(out, strict=True)
    torch.save(net.state_dict(), fout, _use_new_zipfile_serialization=False)
    # sanity: non-fused (eval) should match fused output on random input
    fnet = build('resbn_fused', channels=128, blocks=blocks); fnet.load_state_dict(fsd); fnet.eval(); net.eval()
    import numpy as np
    d = {'is_training': False, 'obs': {'observation': torch.zeros(2, 38, 4, 9), 'action_mask': torch.ones(2, 235)}}
    d['obs']['observation'].normal_()
    with torch.no_grad():
        diff = (fnet(d) - net(d)).abs().max().item()
    print(f"converted {fin} -> {fout}  max|fused-nonfused|={diff:.2e} (should be ~0)")

if __name__ == '__main__':
    main()

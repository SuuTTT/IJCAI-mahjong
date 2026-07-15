# Temporal architectures: CNN(state) + sequence-encoder over the ordered discard-event sequence -> 235 logits.
# The CNN branch mirrors ResBNCNN (128xB, flatten->512) so with the seq branch zeroed it reduces to the
# deployed arch; the seq branch adds the discard-ORDER signal the count-collapsed CNN planes discard.
# Input dict adds d[seq] : (B,L) long token ids (tile*4+rel_seat in [0,135], pad=136).
import torch
from torch import nn

IN_PLANES = 38; GRID = 36; VOCAB = 137; PAD = 136

def _mask(logits, action_mask):
    return logits + torch.clamp(torch.log(action_mask.float()), -1e38, 1e38)

class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(x + y)

class TemporalNet(nn.Module):
    # NOTE: attribute names stem/body/cnn_fc/embed/gru/head are FROZEN (temporal_s0.pkl depends on them).
    def __init__(self, channels=128, blocks=40, emb=64, gru=256, gru_layers=1, in_planes=IN_PLANES, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.cnn_fc = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU())
        self.embed = nn.Embedding(VOCAB, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, gru, num_layers=gru_layers, batch_first=True)
        self.gru_dim = gru
        self.head = nn.Linear(512 + gru, 235)
    def forward(self, d):
        self.train(d.get("is_training", False))
        x = d["obs"]["observation"].float()
        cnn = self.cnn_fc(self.body(self.stem(x)))
        e = self.embed(d["seq"])
        _, h = self.gru(e); hlast = h[-1]
        return _mask(self.head(torch.cat([cnn, hlast], dim=1)), d["obs"]["action_mask"])

class TransformerHistNet(nn.Module):
    """CNN(state) + Transformer-encoder over the ordered discard sequence (attention instead of GRU).
    Learned positional embedding, key-padding-mask on PAD, mean-pool over non-pad tokens.
    CNN branch uses the SAME stem/body/cnn_fc names as TemporalNet for consistency."""
    def __init__(self, channels=128, blocks=40, emb=64, heads=8, tf_layers=3, ff=None,
                 in_planes=IN_PLANES, maxlen=48, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.cnn_fc = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU())
        self.embed = nn.Embedding(VOCAB, emb, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, maxlen, emb)); nn.init.normal_(self.pos, std=0.02)
        ff = ff or 4 * emb
        layer = nn.TransformerEncoderLayer(d_model=emb, nhead=heads, dim_feedforward=ff,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=tf_layers)
        self.seq_dim = emb
        self.head = nn.Linear(512 + emb, 235)
    def forward(self, d):
        self.train(d.get("is_training", False))
        x = d["obs"]["observation"].float()
        cnn = self.cnn_fc(self.body(self.stem(x)))
        seq = d["seq"]
        pad = (seq == PAD)
        e = self.embed(seq) + self.pos[:, :seq.shape[1], :]
        z = self.enc(e, src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        pooled = (z * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return _mask(self.head(torch.cat([cnn, pooled], dim=1)), d["obs"]["action_mask"])

class CNNOnlyControl(nn.Module):
    """Ablation: identical CNN branch + head fed with a ZERO seq-vector (no seq signal)."""
    def __init__(self, channels=128, blocks=40, gru=256, in_planes=IN_PLANES, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.cnn_fc = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU())
        self.head = nn.Linear(512 + gru, 235); self.gru_dim = gru
    def forward(self, d):
        self.train(d.get("is_training", False))
        cnn = self.cnn_fc(self.body(self.stem(d["obs"]["observation"].float())))
        z = torch.zeros(cnn.shape[0], self.gru_dim, device=cnn.device, dtype=cnn.dtype)
        return _mask(self.head(torch.cat([cnn, z], dim=1)), d["obs"]["action_mask"])

def build_seq(kind, **cfg):
    return {"temporal": TemporalNet, "transformer": TransformerHistNet,
            "cnnonly": CNNOnlyControl}[kind](**cfg)

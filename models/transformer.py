"""
Rain sequence transformer for delta_h_m.

The input is 104 weeks of rainfall ending at the reading date plus the static
and tabular features of that reading. The output is the change in water level
since the previous reading, positive when the level fell.

Why a transformer at all, when the tabular table already has rain_30d,
rain_90d, rain_365d and so on: those windows are a guess at which parts of the
rain history matter. The model here is given the raw weekly series and learns
its own weighting over it. Attention is the mechanism: for each reading it
decides which weeks of the past two years to read, instead of being told.

Architecture, PatchTST style:
    weekly series (104,)
      -> non-overlapping patches of 4 weeks              (26, 4)
      -> linear projection to d_model + learned position (26, d)
      -> N transformer encoder blocks, no causal mask    (26, d)
      -> mean pool                                       (d,)
    static and tabular features
      -> standardise, embed categoricals                 (f,)
      -> MLP                                             (d,)
    concat -> MLP -> scalar

Patching rather than one token per week because a 104-token sequence over
attention costs 104^2 per head for no gain: rainfall autocorrelation is far
shorter than a week is long, so a 4-week patch loses nothing and makes the
attention map readable as "which month mattered".

No recurrence anywhere, which is the point of the assignment: the BiLSTM arm
is the recurrent comparison, this arm is the attention comparison, and both
read the same table so the difference is the architecture.
"""

import numpy as np
import torch
import torch.nn as nn


class SeqEncoder(nn.Module):
    """Patch, embed, attend, pool."""

    def __init__(self, n_weeks=104, patch=4, d_model=64, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        assert n_weeks % patch == 0
        self.patch = patch
        self.n_patch = n_weeks // patch
        self.proj = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(block, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x, key_padding_mask=None):
        # x: (B, n_weeks) already scaled
        B = x.shape[0]
        p = x.view(B, self.n_patch, self.patch)
        h = self.proj(p) + self.pos
        h = self.enc(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        if key_padding_mask is None:
            return h.mean(dim=1)
        keep = (~key_padding_mask).float().unsqueeze(-1)
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def attention_maps(self, x, key_padding_mask=None):
        """Per-layer attention weights, for the interpretability slide.

        Runs the blocks by hand because nn.TransformerEncoder does not expose
        them. Returns a list of (B, n_patch, n_patch).
        """
        B = x.shape[0]
        h = self.proj(x.view(B, self.n_patch, self.patch)) + self.pos
        maps = []
        for layer in self.enc.layers:
            y = layer.norm1(h)
            out, w = layer.self_attn(
                y, y, y, key_padding_mask=key_padding_mask,
                need_weights=True, average_attn_weights=True)
            maps.append(w.detach())
            h = h + layer.dropout1(out)
            h = h + layer._ff_block(layer.norm2(h))
        return maps


class TabEncoder(nn.Module):
    def __init__(self, n_numeric, cat_sizes, d_model=64, d_cat=8, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d_cat) for n in cat_sizes])
        d_in = n_numeric + d_cat * len(cat_sizes)
        self.net = nn.Sequential(
            nn.Linear(d_in, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.GELU())

    def forward(self, num, cat):
        parts = [num] + [e(cat[:, i]) for i, e in enumerate(self.embs)]
        return self.net(torch.cat(parts, dim=1))


class RainTransformer(nn.Module):
    def __init__(self, n_numeric, cat_sizes, n_weeks=104, patch=4,
                 d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.seq = SeqEncoder(n_weeks, patch, d_model, n_heads, n_layers, dropout)
        self.tab = TabEncoder(n_numeric, cat_sizes, d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, seq, num, cat, seq_mask=None):
        z = torch.cat([self.seq(seq, seq_mask), self.tab(num, cat)], dim=1)
        return self.head(z).squeeze(-1)


class SeqOnlyTransformer(nn.Module):
    """Ablation: rain history only, no tabular features.

    Worth reporting. If this is close to the full model, the tabular block is
    decoration; if it is far worse, the static well properties carry the
    signal and the sequence is decoration. Either answer is a finding.
    """

    def __init__(self, n_weeks=104, patch=4, d_model=64, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.seq = SeqEncoder(n_weeks, patch, d_model, n_heads, n_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, seq, num=None, cat=None, seq_mask=None):
        return self.head(self.seq(seq, seq_mask)).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

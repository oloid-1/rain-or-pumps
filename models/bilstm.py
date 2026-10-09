"""
Rain sequence BiLSTM for delta_h_m, the recurrent counterpart to transformer.py.

Same contract as the transformer: weekly rain behind each reading plus the
static and tabular features in, change in water level since the previous
reading out, positive when the level fell. Both read the same rows, the same
split and the same tabular arm (TabEncoder is shared), so the difference
between them is the sequence encoder.

Architecture:
    weekly sequence (104, C)           C = 1 (rain only) or 6 (bilstm-data channels)
      -> linear to d_in                                    (104, d_in)
      -> 2-layer bidirectional LSTM, hidden h per direction (104, 2h)
      -> attention pooling over weeks                      (2h,)
      -> layer norm, linear                                (d,)
    tabular  -> TabEncoder (shared with the transformer)    (d,)
    concat -> MLP -> scalar

Why bidirectional. Nothing here is forecast: all 104 weeks are in the past at
prediction time, so the encoder may read them in both directions. The forward
pass carries the build-up of wet and dry weeks towards the reading; the
backward pass lets the most recent weeks condition how the old ones are read.
Same argument as the transformer's missing causal mask.

Why attention pooling and not the last hidden state. The last state of a
104-step LSTM is dominated by the last few weeks, and the forward and backward
final states sit at opposite ends of the series. A learned weight per week
lets the model pick the weeks that matter, and the weights are a per-week map
comparable to the transformer's attention figure.

Why no patching. An LSTM costs linear time in sequence length, so the
transformer's reason for 4-week patches does not apply, and weekly steps keep
the in_interval channel's boundary sharp.
"""

import torch
import torch.nn as nn

from transformer import TabEncoder


class BiLSTMEncoder(nn.Module):
    def __init__(self, n_channels=1, d_in=32, hidden=64, n_layers=2,
                 d_model=64, dropout=0.1, pool="attn"):
        super().__init__()
        assert pool in ("attn", "mean", "last")
        self.pool = pool
        self.inp = nn.Sequential(nn.Linear(n_channels, d_in), nn.GELU())
        self.lstm = nn.LSTM(d_in, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.score = nn.Linear(2 * hidden, 1)
        self.norm = nn.LayerNorm(2 * hidden)
        self.out = nn.Linear(2 * hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.hidden = hidden

    def encode(self, x):
        """x: (B, T) or (B, T, C) already scaled. Returns states (B, T, 2h)
        and pooling weights (B, T)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h, _ = self.lstm(self.inp(x))
        if self.pool == "attn":
            w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        elif self.pool == "mean":
            w = torch.full(h.shape[:2], 1.0 / h.shape[1], device=h.device)
        else:
            w = None
        return h, w

    def forward(self, x):
        h, w = self.encode(x)
        if w is None:
            H = self.hidden
            z = torch.cat([h[:, -1, :H], h[:, 0, H:]], dim=1)   # each direction's final state
        else:
            z = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.out(self.drop(self.norm(z)))

    @torch.no_grad()
    def pooling_weights(self, x):
        """Per-week pooling weights (B, T), for the interpretability figure."""
        return self.encode(x)[1]


class RainBiLSTM(nn.Module):
    def __init__(self, n_numeric, cat_sizes, n_channels=1, d_in=32, hidden=64,
                 n_layers=2, d_model=64, dropout=0.1, pool="attn"):
        super().__init__()
        self.seq = BiLSTMEncoder(n_channels, d_in, hidden, n_layers, d_model,
                                 dropout, pool)
        self.tab = TabEncoder(n_numeric, cat_sizes, d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, seq, num, cat):
        z = torch.cat([self.seq(seq), self.tab(num, cat)], dim=1)
        return self.head(z).squeeze(-1)


class SeqOnlyBiLSTM(nn.Module):
    """Ablation: rain history only, no tabular features."""

    def __init__(self, n_channels=1, d_in=32, hidden=64, n_layers=2,
                 d_model=64, dropout=0.1, pool="attn"):
        super().__init__()
        self.seq = BiLSTMEncoder(n_channels, d_in, hidden, n_layers, d_model,
                                 dropout, pool)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, seq, num=None, cat=None):
        return self.head(self.seq(seq)).squeeze(-1)

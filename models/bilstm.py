"""
BiLSTM arm for delta_h_m, reconstructed to the spec in
docs/BILSTM_PARAMETERS.md (Parth's `bilstm` branch, not yet merged into this
repo). The tabular side reuses `transformer.TabEncoder` unchanged, exactly
as that document says it does, so the two models differ only in how they
read the rain sequence.

    sequence (weeks, C)  -> Linear(C, 32) + GELU
                         -> BiLSTM, 2 layers, 64/direction, dropout 0.1   (weeks, 128)
                         -> attention pool: softmax(Linear(128,1)) over weeks (128,)
                         -> LayerNorm -> Dropout -> Linear(128, 64)        (64,)
    tabular              -> TabEncoder (same code as the transformer)      (64,)
    concat (128,)        -> Linear(128,64) -> GELU -> Dropout -> Linear(64,1)

One deliberate scope cut from the source document: that BiLSTM was trained
on a 6-channel rain cube (rain_mm, rain_anom_mm, in_interval, doy_sin,
doy_cos, wet_frac) built by a separate `bilstm-data` pipeline that was never
merged into rain-or-pumps - only `data/training/rain_seq.npz` (rain_mm
alone) exists here. Rebuilding that 6-channel cube is out of scope for this
pass, so this is the single-channel ("rain only") BiLSTM variant from that
document - 182,962 parameters there, with the same tabular feature count it
comes out within a few hundred parameters of that figure here too. It is
also the fairer comparison: the transformer in this repo has only ever seen
rain_mm, so a 1-channel BiLSTM isolates the recurrent-vs-attention question
instead of mixing in an extra 5 channels of input.
"""

import torch
import torch.nn as nn

from transformer import TabEncoder


class AttentionPool(nn.Module):
    """One learned score per timestep, softmax-normalised, weighted sum."""

    def __init__(self, d_in):
        super().__init__()
        self.score = nn.Linear(d_in, 1)

    def forward(self, h):
        # h: (B, T, d_in)
        s = self.score(h).squeeze(-1)          # (B, T)
        w = torch.softmax(s, dim=1)             # (B, T)
        pooled = (h * w.unsqueeze(-1)).sum(1)   # (B, d_in)
        return pooled, w


class SeqEncoderBiLSTM(nn.Module):
    def __init__(self, n_channels=1, d_in=32, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_channels, d_in), nn.GELU())
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                             bidirectional=True,
                             dropout=dropout if layers > 1 else 0.0)
        d_out = 2 * hidden
        self.pool = AttentionPool(d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_out, hidden)

    def forward(self, x, return_weights=False):
        # x: (B, weeks) for a single channel, or (B, weeks, C)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.proj(x)
        o, _ = self.lstm(h)
        pooled, w = self.pool(o)
        pooled = self.norm(pooled)
        pooled = self.drop(pooled)
        z = self.out(pooled)
        if return_weights:
            return z, w
        return z


class BiLSTMModel(nn.Module):
    def __init__(self, n_numeric, cat_sizes, n_channels=1, d_in=32,
                 hidden=64, layers=2, d_model=64, dropout=0.1):
        super().__init__()
        assert hidden == d_model, "kept equal so neither arm dominates the concat"
        self.seq = SeqEncoderBiLSTM(n_channels, d_in, hidden, layers, dropout)
        self.tab = TabEncoder(n_numeric, cat_sizes, d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def forward(self, seq, num, cat, seq_mask=None):
        z = torch.cat([self.seq(seq), self.tab(num, cat)], dim=1)
        return self.head(z).squeeze(-1)

"""
models.py — the two sequence models of the BdSL paper pipeline.

Both models take the same input: a batch of landmark sequences
    x       : float tensor (batch, time, 225)
    lengths : long tensor (batch,) — real frame count of each clip
and return logits of shape (batch, num_classes).

1. BiLSTMAttention — 2-layer bidirectional LSTM + learned attention.
   The attention layer learns which frames matter (the "peak" of the
   sign) instead of only looking at the last frame. This is the same
   family as the attention bi-LSTM in the BdSLW60 paper
   (Rubaiyeat et al. 2024), which scored 75.1% on the 60-word set.
2. SmallTransformer — 2-layer Transformer encoder, 4 heads, d_model 128.
   Small on purpose: the whole 9-word task must run in real time on a
   laptop CPU, and a big video transformer (e.g. VideoMAE) is far too
   heavy for the sliding-window demo.
3. SimpleLSTM — 1-layer unidirectional LSTM, last-hidden-state
   classifier. The BASELINE for the 1-week sprint (quickstart Day 3):
   the simplest temporal model, so the BiLSTM/Transformer gains in
   Day 4 are measured against something, not claimed in a vacuum.
"""

import math

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """Baseline: 1-layer unidirectional LSTM, last real frame's hidden
    state -> classifier. No attention, no bidirection — the floor that
    the proposed models must beat."""

    def __init__(self, input_dim=225, hidden_dim=128, num_layers=1,
                 num_classes=9, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, batch, hidden) — take the last layer.
        last = h_n[-1]  # (batch, hidden)
        return self.classifier(self.dropout(last))


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM + attention pooling."""

    def __init__(self, input_dim=225, hidden_dim=128, num_layers=2,
                 num_classes=9, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Attention: score each time step, softmax over time, weighted sum.
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # Pack so padding frames never affect the LSTM state.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: (batch, time, 2*hidden)

        scores = self.attn(out).squeeze(-1)                    # (batch, time)
        # Mask padding positions so attention cannot look at them.
        max_len = out.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)                  # (batch, time)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)  # (batch, 2*hidden)
        return self.classifier(self.dropout(context))


class SmallTransformer(nn.Module):
    """2-layer Transformer encoder with masked mean pooling."""

    def __init__(self, input_dim=225, d_model=128, nhead=4, num_layers=2,
                 num_classes=9, dropout=0.3, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        # Fixed sinusoidal positional encoding (no extra parameters).
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        h = self.input_proj(x) + self.pe[:, :x.size(1), :]
        h = self.dropout(h)
        max_len = x.size(1)
        pad_mask = (torch.arange(max_len, device=x.device)[None, :]
                    >= lengths[:, None])  # True = padding
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # Mean over real frames only.
        keep = (~pad_mask).float().unsqueeze(-1)               # (batch, time, 1)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def build_model(name, num_classes, input_dim=225, **kwargs):
    """Build one of the models by name ('lstm', 'bilstm' or 'transformer').

    'hidden_dim' is accepted for all three: it is the LSTM hidden size
    for the LSTM/BiLSTM and the Transformer d_model for the Transformer.
    """
    if name == "lstm":
        return SimpleLSTM(input_dim=input_dim,
                          num_classes=num_classes, **kwargs)
    if name == "bilstm":
        return BiLSTMAttention(input_dim=input_dim,
                               num_classes=num_classes, **kwargs)
    if name == "transformer":
        d_model = kwargs.pop("hidden_dim", 128)
        return SmallTransformer(input_dim=input_dim,
                                num_classes=num_classes,
                                d_model=d_model, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Use 'lstm', 'bilstm' or 'transformer'.")

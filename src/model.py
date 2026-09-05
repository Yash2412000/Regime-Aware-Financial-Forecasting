"""
model.py

Components

  LSTMExpert          — Expert 1: non-linear temporal dependencies
  TransformerExpert   — Expert 2: long-range multi-step dependencies
  GRUExpert           — Expert 3: lighter-weight recurrent alternative
  LinearExpert        — Expert 4: linear/AR-style baseline (fast, interpretable)
  GatingNetwork       — Soft gating: regime posteriors + fused representation
  MoEForecaster       — Full jointly-trained wrapper (y_hat = sum_k g_k * f_k(x))

"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import math

from dataset import N_REGIMES, FEATURE_COLS, SEQ_LEN

N_FEATURES = len(FEATURE_COLS)   # 11


# Expert 1: LSTM

class LSTMExpert(nn.Module):
    """Bidirectional LSTM encoder -> linear head."""

    def __init__(self, input_dim: int = N_FEATURES, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.repr_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        repr_  = self.norm(out[:, -1])
        pred   = self.head(repr_).squeeze(-1)
        return pred, repr_


# Expert 2: Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerExpert(nn.Module):
    """Input projection -> Positional Encoding -> Transformer Encoder -> mean pool -> head."""

    def __init__(self, input_dim: int = N_FEATURES, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.repr_dim = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.pos_enc(self.input_proj(x))
        z     = self.encoder(z)
        repr_ = self.norm(z.mean(dim=1))
        pred  = self.head(repr_).squeeze(-1)
        return pred, repr_


# Expert 3: GRU

class GRUExpert(nn.Module):

    def __init__(self, input_dim: int = N_FEATURES, hidden_dim: int = 96,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )
        self.repr_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x)
        repr_  = self.norm(out[:, -1])
        pred   = self.head(repr_).squeeze(-1)
        return pred, repr_


#Expert 4: Linear / AR baseline

class LinearExpert(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, seq_len: int = SEQ_LEN,
                 repr_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        flat_dim = input_dim * seq_len
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, repr_dim), nn.LayerNorm(repr_dim), nn.Dropout(dropout),
        )
        self.head = nn.Linear(repr_dim, 1)
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat  = x.flatten(start_dim=1)        # (B, T*F)
        repr_ = self.proj(flat)               # (B, repr_dim)
        pred  = self.head(repr_).squeeze(-1)
        return pred, repr_


# Gating Network

class GatingNetwork(nn.Module):
    def __init__(self, repr_total_dim: int, n_experts: int, hidden_dim: int = 64):
        super().__init__()
        gate_input_dim = repr_total_dim + N_REGIMES
        self.net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, fused_repr: torch.Tensor, regime_posterior: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([fused_repr, regime_posterior], dim=-1)
        logits     = self.net(gate_input)
        return functional.softmax(logits, dim=-1)


# Full MoE Forecaster

class MoEForecaster(nn.Module):

    REGIME_COLS_IDX = [
        FEATURE_COLS.index("regime_bull"),
        FEATURE_COLS.index("regime_bear"),
        FEATURE_COLS.index("regime_highvol"),
    ]

    def __init__(
        self,
        use_experts: tuple[str, ...] = ("lstm", "transformer", "gru", "linear"),
        lstm_hidden: int = 128, lstm_layers: int = 2,
        d_model: int = 128, nhead: int = 4, tf_layers: int = 2,
        gru_hidden: int = 96, gru_layers: int = 2,
        linear_repr: int = 32,
        dropout: float = 0.2, gate_hidden: int = 64,
    ):
        super().__init__()

        builders = {
            "lstm":        lambda: LSTMExpert(hidden_dim=lstm_hidden, num_layers=lstm_layers, dropout=dropout),
            "transformer": lambda: TransformerExpert(d_model=d_model, nhead=nhead, num_layers=tf_layers, dropout=dropout),
            "gru":         lambda: GRUExpert(hidden_dim=gru_hidden, num_layers=gru_layers, dropout=dropout),
            "linear":      lambda: LinearExpert(repr_dim=linear_repr, dropout=dropout),
        }
        self.expert_names = list(use_experts)
        self.experts = nn.ModuleList([builders[name]() for name in self.expert_names])
        self.n_experts = len(self.experts)

        repr_total = sum(e.repr_dim for e in self.experts)
        self.gate  = GatingNetwork(repr_total_dim=repr_total, n_experts=self.n_experts, hidden_dim=gate_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regime_post = x[:, -1, self.REGIME_COLS_IDX]
        regime_post = regime_post.clamp(min=0)
        regime_post = regime_post / (regime_post.sum(dim=-1, keepdim=True) + 1e-9)

        expert_preds, expert_reprs = [], []
        for expert in self.experts:
            pred, repr_ = expert(x)
            expert_preds.append(pred)
            expert_reprs.append(repr_)

        preds      = torch.stack(expert_preds, dim=1)
        fused_repr = torch.cat(expert_reprs, dim=-1)

        g_weights = self.gate(fused_repr, regime_post)
        y_hat     = (g_weights * preds).sum(dim=1)

        return y_hat, g_weights, preds


if __name__ == "__main__":
    B, T, F = 8, SEQ_LEN, N_FEATURES
    x_dummy = torch.randn(B, T, F)

    model = MoEForecaster()
    y_hat, g_w, preds = model(x_dummy)

    print(f"Experts: {model.expert_names}")
    print("MoEForecaster — forward pass check")
    print(f"  Input        : {x_dummy.shape}")
    print(f"  y_hat        : {y_hat.shape}   (expected ({B},))")
    print(f"  gate weights : {g_w.shape}    (expected ({B}, {model.n_experts}))")
    print(f"  expert preds : {preds.shape}  (expected ({B}, {model.n_experts}))")
    print(f"  Gate sums    : {g_w.sum(dim=-1)}  (all should be approx 1.0)")
    print(f"\nParameter count: {sum(p.numel() for p in model.parameters()):,}")

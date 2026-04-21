"""Proprioception projector and connector transformer.

Ported from VLANeXt encoder.py and connector.py.
"""

import torch
import torch.nn as nn


class ActionTransformerProjector(nn.Module):
    """Projects proprioception history [B, T, action_dim] into VLM embedding space."""

    def __init__(self, action_dim, hidden_size, depth=2, num_heads=4, mlp_ratio=4.0, max_len=64):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.input_proj(x)
        seq_len = x.shape[1]
        eff_len = min(seq_len, self.pos_embed.shape[1])
        pos_embed_slice = self.pos_embed[:, :eff_len, :]
        if seq_len > eff_len:
            x[:, :eff_len, :] = x[:, :eff_len, :] + pos_embed_slice
        else:
            x = x + pos_embed_slice
        x = self.encoder(x)
        return self.norm(x)


class ConnectorTransformer(nn.Module):
    """Self-attention connector for meta-query outputs (loose/soft conditioning)."""

    def __init__(self, input_dim, output_dim, depth=2, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim)
        else:
            self.input_proj = nn.Identity()
        hidden_size = output_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.norm(x)

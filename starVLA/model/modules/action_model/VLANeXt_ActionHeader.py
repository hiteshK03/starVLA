"""VLANeXt MoE Diffusion Action Head.

Ported from VLANeXt policies.py.
Key difference from Star VLA's GR00T DiT: receives per-layer VLM hidden
states (one per policy block) instead of cross-attending to the last hidden
state only.
"""

import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class MoEBlock(nn.Module):
    def __init__(self, hidden_size, vlm_hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, vlm_feat):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        v_feat = self.vlm_proj(vlm_feat)
        kv = torch.cat([x_norm, v_feat], dim=1)
        attn_out, _ = self.attn(query=x_norm, key=kv, value=kv)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ActionDiffusionTransformerMoE(nn.Module):
    """Diffusion policy head with per-layer VLM conditioning via MoE blocks.

    Unlike Star VLA's DiT which cross-attends to a single VLM hidden state,
    this head receives `vlm_hidden_states` — a list/tuple of hidden states
    from the last N VLM layers (one per MoE block).
    """

    def __init__(self, action_dim, vlm_hidden_size, hidden_size=384, depth=12,
                 num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))
        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noisy_action, timestep, vlm_hidden_states, history_actions=None):
        """Training forward.

        Args:
            noisy_action: [B, T, action_dim]
            timestep: [B] timesteps
            vlm_hidden_states: tuple/list of [B, L, vlm_hidden_size], one per VLM layer
            history_actions: optional [B, T_hist, action_dim]
        Returns:
            output: [B, T, action_dim] predicted velocity
        """
        noisy_action = noisy_action.to(dtype=self.input_proj.weight.dtype)

        if history_actions is not None:
            history_actions = history_actions.to(dtype=self.input_proj.weight.dtype)
            x_input = torch.cat([history_actions, noisy_action], dim=1)
        else:
            x_input = noisy_action

        x = self.input_proj(x_input)
        x = x + self.pos_embed[:, :x.shape[1], :]
        t = self.t_embedder(timestep)

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        for block, vlm_state in zip(self.blocks, relevant_vlm_states):
            vlm_state = vlm_state.to(dtype=x.dtype)
            x = block(x, t, vlm_state)

        output = self.final_layer(x, t)
        if history_actions is not None:
            output = output[:, -noisy_action.shape[1]:, :]
        return output


class ActionClassificationTransformerMoE(nn.Module):
    """Classification policy head with per-layer VLM conditioning via MoE blocks.

    Ported from VLANeXt's ActionClassificationTransformerMoE.
    Predicts per-dimension bin logits instead of continuous velocity.
    """

    def __init__(self, action_dim, vlm_hidden_size, num_actions=1, num_bins=256,
                 hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.total_queries = num_actions * action_dim
        self.per_dim_classes = num_bins

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, self.total_queries, hidden_size))
        self.cond_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))

        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, vlm_hidden_states, history_actions=None):
        """
        Args:
            vlm_hidden_states: tuple/list of [B, L, vlm_hidden_size]
            history_actions: optional [B, T_hist, action_dim]
        Returns:
            logits: [B, num_actions, action_dim, num_bins]
        """
        final_state = vlm_hidden_states[-1]
        dtype = self.input_proj.weight.dtype
        final_state = final_state.to(dtype=dtype)

        c_emb = final_state.mean(dim=1)
        c = self.cond_proj(c_emb)

        B = c.shape[0]
        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, :x.shape[1], :]

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        for block, vlm_state in zip(self.blocks, relevant_vlm_states):
            vlm_state = vlm_state.to(dtype=dtype)
            x = block(x, c, vlm_state)

        output = self.final_layer(x, c)
        output = output[:, -self.total_queries:, :]

        return output.view(B, self.num_actions, self.action_dim, self.per_dim_classes)

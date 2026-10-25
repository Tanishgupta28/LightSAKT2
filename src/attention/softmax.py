"""Causal attention controls for the architecture study.

Projections and output dropout follow LightSAKT. Attention probability dropout
is deliberately absent, matching the ELU+1 kernel path. The optional learned
distance penalty is an isolated recency bias, not a full AKT/ALiBi model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.attention.linear import LinearAttention


class CausalSoftmaxAttention(LinearAttention):
    def __init__(self, d_model, n_heads, dropout=0.1, recency_bias=False,
                 initial_decay=1 / 32):
        super().__init__(d_model, n_heads, dropout)
        if not math.isfinite(initial_decay) or not 0 < initial_decay <= 1:
            raise ValueError("initial_decay must be finite and in (0, 1]")
        if recency_bias:
            self.raw_decay = nn.Parameter(torch.full(
                (n_heads,), math.log(math.expm1(initial_decay))))
        else:
            self.register_parameter("raw_decay", None)

    def forward(self, query, key, value, valid=None, get_attn_weights=False):
        b, length, _ = query.shape
        if query.shape != key.shape or key.shape != value.shape or length < 1:
            raise ValueError("Aligned nonempty B,L,d tensors required")
        if valid is None:
            valid = torch.ones((b, length), dtype=torch.bool, device=query.device)
        if valid.shape != (b, length) or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean B,L")

        def heads(x):
            return x.reshape(b, length, self.n_heads, self.head_dim).transpose(1, 2)

        # Sanitize padding before projection, including NaN/garbage payloads.
        query, key, value = [x.masked_fill(~valid[..., None], 0)
                             for x in (query, key, value)]
        q = heads(self.q_proj(query))
        k = heads(self.k_proj(key))
        v = heads(self.v_proj(value))
        scores = (q @ k.transpose(-1, -2)) * self.head_dim ** -0.5
        positions = torch.arange(length, device=query.device)
        distance = positions[:, None] - positions[None, :]
        allowed = ((distance >= 0)[None, None]
                   & valid[:, None, :, None] & valid[:, None, None, :])
        if self.raw_decay is not None:
            scores = scores - F.softplus(self.raw_decay)[None, :, None, None] * distance.clamp_min(0)
        scores = scores.masked_fill(~allowed, -torch.inf)
        # A padded query can have no keys. Avoid softmax(-inf,...,-inf).
        scores = scores.masked_fill(~allowed.any(-1, keepdim=True), 0)
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0)
        out = (weights @ v).transpose(1, 2).reshape(b, length, self.d_model)
        out = self.out_proj(out).masked_fill(~valid[..., None], 0)
        return out, weights if get_attn_weights else None

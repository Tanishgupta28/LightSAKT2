"""Causal ELU+1 attention. Dense weights are visualization-only."""
import torch
from torch import nn
from torch.nn import functional as F


def prefix_sum(x):
    """Work-efficient deterministic tree scan: O(L) additions, O(log L) levels.

    CUDA cumsum is rejected by strict determinism in the supplied PyTorch build.
    Pair reduction plus reconstruction computes inclusive prefixes without an L^2 mask.
    """
    if x.device.type != "cuda":
        return x.cumsum(dim=2)
    length = x.shape[2]
    if length == 1:
        return x
    if length % 2:
        x = torch.cat([x, torch.zeros_like(x[:, :, :1])], dim=2)
    even, odd = x[:, :, ::2], x[:, :, 1::2]
    pair_prefix = prefix_sum(even + odd)
    previous = torch.cat([torch.zeros_like(pair_prefix[:, :, :1]), pair_prefix[:, :, :-1]], dim=2)
    return torch.stack([previous + even, pair_prefix], dim=3).flatten(2, 3)[:, :, :length]


class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)  # original module; no weight dropout in kernel path

    def forward(self, query, key, value, valid=None, get_attn_weights=False):
        # Batch first; aligned queries at t predict the question after history t.
        b, length, _ = query.shape
        if query.shape != key.shape or key.shape != value.shape or length < 1:
            raise ValueError("Aligned nonempty B,L,d tensors required")
        if valid is None:
            valid = torch.ones((b, length), dtype=torch.bool, device=query.device)
        if valid.shape != (b, length) or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean B,L")
        def heads(x):
            return x.reshape(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.elu(heads(self.q_proj(query))) + 1
        k = F.elu(heads(self.k_proj(key))) + 1
        v = heads(self.v_proj(value))
        keep = valid[:, None, :, None]
        k = torch.where(keep, k, 0)
        v = torch.where(keep, v, 0)
        # Both sums include precisely the same valid j<=t prefix.
        prefix_k = prefix_sum(k)
        prefix_kv = prefix_sum(k.unsqueeze(-1) * v.unsqueeze(-2))
        numerator = torch.einsum("bhld,bhlde->bhle", q, prefix_kv)
        denominator = (q * prefix_k).sum(-1, keepdim=True) + 1e-8
        out = (numerator / denominator).transpose(1, 2).reshape(b, length, self.d_model)
        out = self.out_proj(out).masked_fill(~valid[..., None], 0)
        weights = None
        if get_attn_weights:
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
            causal = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
            scores = scores.masked_fill(~causal, 0).masked_fill(~valid[:, None, :, None], 0)
            weights = scores / denominator
        return out, weights

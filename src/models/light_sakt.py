import torch
from torch import nn
from src.attention.linear import LinearAttention


class Light_SAKT(nn.Module):
    """Original trainable structure with corrected attention and explicit validity."""
    def __init__(self, num_q, embed_dim=64, num_heads=2, dropout=0.1,
                 max_seq_len=100, variant="default"):
        super().__init__()
        if variant not in {"default", "no_dropout", "no_position", "single_head", "no_attention"}:
            raise ValueError("Undefined ablation (no_regularization requires a scientific definition)")
        if variant == "no_dropout":
            dropout = 0.0
        if variant == "single_head":
            num_heads = 1
        self.num_q, self.embed_dim, self.num_heads = num_q, embed_dim, num_heads
        self.dropout, self.max_seq_len, self.variant = dropout, max_seq_len, variant
        self.M = nn.Embedding(num_q * 2, embed_dim)
        self.E = nn.Embedding(num_q, embed_dim)
        self.P = nn.Parameter(torch.empty(max_seq_len, embed_dim))
        nn.init.kaiming_normal_(self.P)
        self.attn = LinearAttention(embed_dim, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.FFN = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(embed_dim // 2, embed_dim),
                                 nn.Dropout(dropout))
        self.FFN_layer_norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, q, r, qry, valid=None, get_attn_weights=False):
        if q.ndim != 2 or q.shape != r.shape or q.shape != qry.shape:
            raise ValueError("q,r,qry must be aligned B,L")
        if not 0 < q.shape[1] <= self.max_seq_len:
            raise ValueError("Sequence length outside positional capacity")
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        if valid.shape != q.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean B,L")
        q, r, qry = [x.masked_fill(~valid, 0).long() for x in (q, r, qry)]
        m = self.M(q + self.num_q * r)
        if self.variant != "no_position":
            m = m + self.P[:q.shape[1]]
        m = m.masked_fill(~valid[..., None], 0)
        e = self.E(qry).masked_fill(~valid[..., None], 0)
        if self.variant == "no_attention":
            s, weights = torch.zeros_like(m), None
        else:
            s, weights = self.attn(e, m, m, valid, get_attn_weights)
        s = self.attn_layer_norm(self.attn_dropout(s) + m + e)
        f = self.FFN_layer_norm(self.FFN(s) + s)
        p = torch.sigmoid(self.pred(f)).squeeze(-1).masked_fill(~valid, 0)
        return p, weights


LightSAKT = Light_SAKT

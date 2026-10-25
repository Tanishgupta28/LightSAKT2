"""Separate architecture variants; the original LightSAKT stays unchanged."""
import torch
from torch import nn

from src.attention.linear import LinearAttention
from src.attention.softmax import CausalSoftmaxAttention
from src.models.light_sakt import LightSAKT


ATTENTION_TYPES = ("linear", "softmax", "recency_softmax")


def make_attention(embed_dim, num_heads, dropout, attention_type):
    if attention_type == "linear":
        return LinearAttention(embed_dim, num_heads, dropout)
    return CausalSoftmaxAttention(embed_dim, num_heads, dropout,
                                  recency_bias=attention_type == "recency_softmax")


class LightSAKTEncoderLayer(nn.Module):
    """One independent attention + FFN block with LightSAKT's post norms."""
    def __init__(self, embed_dim, num_heads, dropout, attention_type):
        super().__init__()
        self.attn = make_attention(embed_dim, num_heads, dropout, attention_type)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.FFN = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(embed_dim // 2, embed_dim),
                                 nn.Dropout(dropout))
        self.FFN_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, memory, valid, get_attn_weights=False):
        s, weights = self.attn(query, memory, memory, valid, get_attn_weights)
        s = self.attn_layer_norm(self.attn_dropout(s) + memory + query)
        f = self.FFN_layer_norm(self.FFN(s) + s)
        return f.masked_fill(~valid[..., None], 0), weights


class LightSAKTResearch(LightSAKT):
    """Configurable stack of LightSAKT blocks with a fixed target query stream.

Layer 1 is the baseline block. Each later layer reads the preceding layer's
causal hidden states as K/V and uses the same E(qry) as Q. Blocks have independent
parameters; embeddings and the prediction head are shared. No target response
is introduced. Optional visualization returns the final layer's weights.

The one-layer linear case has exactly the baseline state keys and forward path.
New layers are initialized after the baseline, retaining its initial parameters.
"""
    def __init__(self, num_q, embed_dim=64, num_heads=2, dropout=0.1,
                 max_seq_len=100, num_layers=2, attention_type="linear"):
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if attention_type not in ATTENTION_TYPES:
            raise ValueError(f"attention_type must be one of {ATTENTION_TYPES}")
        super().__init__(num_q, embed_dim, num_heads, dropout, max_seq_len)
        self.num_layers, self.attention_type = num_layers, attention_type
        if attention_type != "linear":
            # Preserve baseline projection initialization and the CPU RNG stream.
            with torch.random.fork_rng(devices=[]):
                replacement = make_attention(embed_dim, num_heads, dropout, attention_type)
            state = replacement.state_dict()
            state.update(self.attn.state_dict())
            replacement.load_state_dict(state)
            self.attn = replacement
        self.extra_layers = nn.ModuleList([
            LightSAKTEncoderLayer(embed_dim, num_heads, dropout, attention_type)
            for _ in range(num_layers - 1)])

    def forward(self, q, r, qry, valid=None, get_attn_weights=False):
        if self.num_layers == 1:
            return super().forward(q, r, qry, valid, get_attn_weights)
        if q.ndim != 2 or q.shape != r.shape or q.shape != qry.shape:
            raise ValueError("q,r,qry must be aligned B,L")
        if not 0 < q.shape[1] <= self.max_seq_len:
            raise ValueError("Sequence length outside positional capacity")
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        if valid.shape != q.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean B,L")
        q, r, qry = [x.masked_fill(~valid, 0).long() for x in (q, r, qry)]
        m = (self.M(q + self.num_q * r) + self.P[:q.shape[1]]).masked_fill(~valid[..., None], 0)
        e = self.E(qry).masked_fill(~valid[..., None], 0)
        s, weights = self.attn(e, m, m, valid, get_attn_weights)
        s = self.attn_layer_norm(self.attn_dropout(s) + m + e)
        f = self.FFN_layer_norm(self.FFN(s) + s).masked_fill(~valid[..., None], 0)
        for layer in self.extra_layers:
            f, weights = layer(e, f, valid, get_attn_weights)
        p = torch.sigmoid(self.pred(f)).squeeze(-1).masked_fill(~valid, 0)
        return p, weights

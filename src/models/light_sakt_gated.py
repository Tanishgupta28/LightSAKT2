"""One-block LightSAKT with only a query-conditioned attention-output gate."""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.light_sakt import LightSAKT


class LightSAKTGated(LightSAKT):
    """Scale the historical attention branch by 2*sigmoid(w E(qry) + b).

    One scalar per position is shared across channels. Zero initialization
    gives an identity gate, preserving baseline initialization and RNG state.
    Attention, residual memory/query, norms, FFN and prediction are unchanged.
    Only the known next-skill embedding conditions the gate; no future response
    or later hidden state enters it. There are embed_dim+1 new parameters.
    """
    def __init__(self, num_q, embed_dim=64, num_heads=2, dropout=0.1,
                 max_seq_len=100):
        super().__init__(num_q, embed_dim, num_heads, dropout, max_seq_len)
        self.history_gate_weight = nn.Parameter(torch.zeros(1, embed_dim))
        self.history_gate_bias = nn.Parameter(torch.zeros(1))

    def history_gate(self, query_embedding):
        return 2 * torch.sigmoid(F.linear(query_embedding, self.history_gate_weight,
                                         self.history_gate_bias))

    def forward(self, q, r, qry, valid=None, get_attn_weights=False):
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
        attention, weights = self.attn(e, m, m, valid, get_attn_weights)
        s = self.attn_layer_norm(self.attn_dropout(self.history_gate(e) * attention) + m + e)
        f = self.FFN_layer_norm(self.FFN(s) + s)
        p = torch.sigmoid(self.pred(f)).squeeze(-1).masked_fill(~valid, 0)
        # Optional weights describe the unchanged normalized attention kernel;
        # gate scaling is a separate output-branch operation.
        return p, weights

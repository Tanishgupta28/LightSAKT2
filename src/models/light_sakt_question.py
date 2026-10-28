"""Separate LightSAKT readout with a known-question difficulty correction."""
import torch
from torch import nn
from src.models.light_sakt import LightSAKT


class LightSAKTQuestion(LightSAKT):
    """One extra scalar multiplies a training-derived, cross-fitted feature.

    Difficulty is logit(smoothed problem success)-logit(skill success).
    Feature construction excludes the entire current student for train rows,
    and uses only training students for validation rows. No test support here.
    Zero initial scale preserves baseline initialization/RNG/forward behavior.
    """
    def __init__(self, num_q, embed_dim=64, num_heads=2, dropout=.1, max_seq_len=100):
        super().__init__(num_q, embed_dim, num_heads, dropout, max_seq_len)
        self.difficulty_scale = nn.Parameter(torch.zeros(()))

    def forward(self, q, r, qry, valid=None, get_attn_weights=False, difficulty=None):
        if q.ndim != 2 or q.shape != r.shape or q.shape != qry.shape:
            raise ValueError("q,r,qry must be aligned B,L")
        if not 0 < q.shape[1] <= self.max_seq_len:
            raise ValueError("Sequence length outside positional capacity")
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        if valid.shape != q.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean B,L")
        if difficulty is None or difficulty.shape != q.shape:
            raise ValueError("Known-question difficulty feature B,L required")
        difficulty = difficulty.masked_fill(~valid, 0)
        if not torch.isfinite(difficulty).all():
            raise ValueError("Nonfinite question difficulty")
        q, r, qry = [x.masked_fill(~valid, 0).long() for x in (q, r, qry)]
        m = (self.M(q + self.num_q * r) + self.P[:q.shape[1]]).masked_fill(~valid[..., None], 0)
        e = self.E(qry).masked_fill(~valid[..., None], 0)
        s, weights = self.attn(e, m, m, valid, get_attn_weights)
        s = self.attn_layer_norm(self.attn_dropout(s) + m + e)
        f = self.FFN_layer_norm(self.FFN(s) + s)
        logits = self.pred(f).squeeze(-1) + self.difficulty_scale * difficulty
        return torch.sigmoid(logits).masked_fill(~valid, 0), weights

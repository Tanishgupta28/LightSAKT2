import os
import numpy as np
import torch
from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
import pickle

class SAKT(Module):
    """
        Args:
            num_q: total number of questions (KCs)
            n: sequence length
            d: hidden vector dimension
            num_attn_heads: number of attention heads
            dropout: dropout rate
    """

    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        kaiming_normal_(self.P)
        self.attn = MultiheadAttention(self.d, self.num_attn_heads, dropout=self.dropout)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)
        self.FFN = Sequential(Linear(self.d, self.d), ReLU(), Dropout(self.dropout), Linear(self.d, self.d), Dropout(self.dropout))
        self.FFN_layer_norm = LayerNorm(self.d)
        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, valid=None):
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        q, r, qry = [x.masked_fill(~valid, 0) for x in (q, r, qry)]
        x = q + self.num_q * r
        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P[:q.shape[1]].unsqueeze(1)
        causal_mask = torch.triu(torch.ones([E.shape[0], M.shape[0]], device=q.device), diagonal=1).bool()
        M = M + P
        (S, attn_weights) = self.attn(E, M, M, attn_mask=causal_mask, key_padding_mask=~valid, need_weights=False)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)
        S = self.attn_layer_norm(S + M + E)
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)
        p = torch.sigmoid(self.pred(F)).squeeze(-1)
        return (p.masked_fill(~valid, 0), attn_weights)

import os
import numpy as np
import torch
import pickle
from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics

class DKVMN(Module):
    """
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    """

    def __init__(self, num_q, dim_s, size_m):
        super().__init__()
        self.num_q = num_q
        self.dim_s = dim_s
        self.size_m = size_m
        self.k_emb_layer = Embedding(self.num_q, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)
        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_s)
        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.p_layer = Linear(self.dim_s, 1)
        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r, qry, valid=None):
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        q, r, qry = [x.masked_fill(~valid, 0) for x in (q, r, qry)]
        state = self.Mv0.unsqueeze(0).expand(q.shape[0], -1, -1)
        k = self.k_emb_layer(q)
        v = self.v_emb_layer(q + self.num_q * r)
        w = torch.softmax(k @ self.Mk.T, -1)
        e, a = torch.sigmoid(self.e_layer(v)), torch.tanh(self.a_layer(v))
        next_k = self.k_emb_layer(qry)
        next_w = torch.softmax(next_k @ self.Mk.T, -1)
        predictions = []
        for t in range(q.shape[1]):
            new = state * (1 - w[:, t, :, None] * e[:, t, None, :]) + w[:, t, :, None] * a[:, t, None, :]
            state = torch.where(valid[:, t, None, None], new, state)
            read = (next_w[:, t, :, None] * state).sum(1)
            f = torch.tanh(self.f_layer(torch.cat([read, next_k[:, t]], -1)))
            predictions.append(torch.sigmoid(self.p_layer(f)).squeeze(-1))
        return torch.stack(predictions, 1).masked_fill(~valid, 0), None

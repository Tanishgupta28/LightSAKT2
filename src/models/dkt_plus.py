import os
import numpy as np
import torch
import pickle
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

class DKTPlus(Module):
    """
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
            lambda_r: the hyperparameter of this model
            lambda_w1: the hyperparameter of this model
            lambda_w2: the hyperparameter of this model
    """

    def __init__(self, num_q, emb_size, hidden_size, lambda_r, lambda_w1, lambda_w2):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r, valid=None):
        """
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        """
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        q, r = q.masked_fill(~valid, 0), r.masked_fill(~valid, 0)
        x = q + self.num_q * r
        emb = self.interaction_emb(x)
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            state = (emb.new_zeros(1, q.shape[0], self.hidden_size), emb.new_zeros(1, q.shape[0], self.hidden_size))
            outputs = []
            for t in range(q.shape[1]):
                _, new = self.lstm_layer(emb[:, t:t+1], state)
                state = tuple(torch.where(valid[:, t][None, :, None], n, old) for n, old in zip(new, state))
                outputs.append(state[0][0])
            h = torch.stack(outputs, 1)
        else:
            h, _ = self.lstm_layer(emb)
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)
        return y

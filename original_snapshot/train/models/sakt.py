import os
import numpy as np
import torch
from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
import pickle

class SAKT(Module):
    '''
        Args:
            num_q: total number of questions (KCs)
            n: sequence length
            d: hidden vector dimension
            num_attn_heads: number of attention heads
            dropout: dropout rate
    '''
    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout 

        self.M = Embedding(self.num_q * 2, self.d) # question & response embedding
        self.E = Embedding(self.num_q, d) 
        self.P = Parameter(torch.Tensor(self.n, self.d)) # positional encoding

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(self.d, self.num_attn_heads, dropout=self.dropout)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry):
        x = q + self.num_q * r
        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(torch.ones([E.shape[0], M.shape[0]]), diagonal=1).bool()

        M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        aucs = []
        loss_means = []
        accuracies = []
        f1_scores = []
        mses = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean_epoch = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean_epoch.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()

                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    y_pred = (p >= 0.5).int().numpy()
                    y_true = t.int().numpy()

                    auc = metrics.roc_auc_score(y_true=y_true, y_score=p.numpy())
                    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)
                    mse = metrics.mean_squared_error(y_true=y_true, y_pred=p.numpy())

                    loss_mean_val = np.mean(loss_mean_epoch)

                    print(
                        f"Epoch: {i}, AUC: {auc:.4f}, Accuracy: {acc:.4f}, "
                        f"F1 Score: {f1:.4f}, MSE: {mse:.4f}, Loss Mean: {loss_mean_val:.4f}"
                    )

                    if auc > max_auc:
                        torch.save(self.state_dict(), os.path.join(ckpt_path, "model.ckpt"))
                        max_auc = auc

                    aucs.append(auc)
                    accuracies.append(acc)
                    f1_scores.append(f1)
                    mses.append(mse)
                    loss_means.append(loss_mean_val)

        # Save all metrics with consistent keys 
        with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
            pickle.dump({
                "auc": aucs,
                "accuracy": accuracies,
                "f1": f1_scores,
                "mse": mses,
                "loss": loss_means,
            }, f)

        return aucs, loss_means, accuracies, f1_scores,mses
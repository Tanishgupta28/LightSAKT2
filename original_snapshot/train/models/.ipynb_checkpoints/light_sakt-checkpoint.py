import os
import numpy as np
import torch
from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error


class LinearAttention(Module):
    """
    Linear Attention mechanism with O(n) complexity instead of O(n^2)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[1]

        # Project and reshape
        q = self.q_proj(query).view(-1, batch_size * self.n_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(key).view(-1, batch_size * self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value).view(-1, batch_size * self.n_heads, self.head_dim).transpose(0, 1)

        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # Linear attention
        kv = torch.bmm(k.transpose(1, 2), v)
        qkv = torch.bmm(q, kv)

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(0).repeat(self.n_heads * batch_size, 1, 1)
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, k.size(-1))
            k_expanded = k.unsqueeze(1).expand(-1, k.size(1), -1, -1)
            k_masked = k_expanded.masked_fill(mask_expanded, 0).sum(dim=2)
            k_sum = k_masked.transpose(1, 2).sum(dim=2, keepdim=True)
        else:
            k_sum = k.transpose(1, 2).sum(dim=2, keepdim=True)

        qkv = qkv / (k_sum.transpose(1, 2) + 1e-8)
        qkv = qkv.transpose(0, 1).contiguous().view(query.shape)
        output = self.out_proj(qkv)

        return output, None


class Light_SAKT(Module):
    """
    Lightweight Self-Attentive Knowledge Tracing
    """
    def __init__(
        self,
        num_q,               # Total number of questions
        embed_dim=64,        # Embedding dimension
        num_heads=2,         # Attention heads
        dropout=0.1,         # Dropout rate
        max_seq_len=100      # Maximum sequence length
    ):
        super().__init__()
        self.num_q = num_q
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Embeddings
        self.M = Embedding(self.num_q * 2, self.embed_dim)
        self.E = Embedding(self.num_q, self.embed_dim)
        self.P = Parameter(torch.Tensor(self.max_seq_len, self.embed_dim))
        kaiming_normal_(self.P)

        # Linear Attention
        self.attn = LinearAttention(self.embed_dim, self.num_heads, dropout=self.dropout)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.embed_dim)

        # Lightweight FFN
        hidden_dim = self.embed_dim // 2
        self.FFN = Sequential(
            Linear(self.embed_dim, hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(hidden_dim, self.embed_dim),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.embed_dim)
        self.pred = Linear(self.embed_dim, 1)

    def forward(self, q, r, qry):
        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(torch.ones([E.shape[0], M.shape[0]]), diagonal=1).bool()
        M = M + P

        S, _ = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()
        return p, None

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        aucs, accs, f1s, mses, loss_means = [], [], [], [], []
        max_auc, patience, no_improve = 0, 5, 0

        for i in range(1, num_epochs + 1):
            loss_epoch = []
            self.train()
            for data in train_loader:
                q, r, qshft, rshft, m = data
                p, _ = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()
                loss_epoch.append(loss.detach().cpu().numpy())

            # Evaluation
            self.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data
                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()
                    all_p.extend(p.numpy())
                    all_t.extend(t.numpy())

            y_true = np.array(all_t)
            y_pred = np.array(all_p)
            y_pred_label = (y_pred >= 0.5).astype(int)

            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred_label)
            f1 = f1_score(y_true, y_pred_label)
            mse = mean_squared_error(y_true, y_pred)
            loss_mean = np.mean(loss_epoch)

            print(f"Epoch {i} → AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}, MSE: {mse:.4f}, Loss: {loss_mean:.4f}")

            if auc > max_auc:
                torch.save(self.state_dict(), os.path.join(ckpt_path, "model.ckpt"))
                max_auc = auc
                no_improve = 0
            else:
                no_improve += 1

            aucs.append(float(auc))
            accs.append(float(acc))
            f1s.append(float(f1))
            mses.append(float(mse))
            loss_means.append(float(loss_mean))

            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {i}")
                break

        # return {"auc": aucs, "accuracy": accs, "f1": f1s, "mse": mses, "loss": loss_means}
        return aucs, loss_means, accs, f1s, mses
        

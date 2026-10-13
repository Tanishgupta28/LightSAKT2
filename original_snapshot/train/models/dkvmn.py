import os

import numpy as np
import torch
import pickle
from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics


class DKVMN(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
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

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                p: the knowledge level about q
                Mv: the value matrices from q, r
        '''
        x = q + self.num_q * r

        batch_size = x.shape[0]
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        k = self.k_emb_layer(q)
        v = self.v_emb_layer(x)

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = torch.sigmoid(self.p_layer(f)).squeeze()

        return p, Mv

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        '''
            Args:
               train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for testing
                num_epochs: the number of epochs
                opt: the optimizer to train this model
                ckpt_path: the path to save this model's parameters
        '''

        aucs = []
        loss_means = []
        accuracies = []
        f1_scores = []
        mses = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean_epoch = []

            for data in train_loader:
                q, r, _, _, m = data

                self.train()
                p, _ = self(q.long(), r.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(r, m).float()

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean_epoch.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, _, _, m = data

                    self.eval()
                    p, _ = self(q.long(), r.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(r, m).float().detach().cpu()

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
                        torch.save(
                            self.state_dict(),
                            os.path.join(ckpt_path, "model.ckpt")
                        )
                        max_auc = auc

                    aucs.append(auc)
                    accuracies.append(acc)
                    f1_scores.append(f1)
                    mses.append(mse)
                    loss_means.append(loss_mean_val)

    # Save all metrics to one file
        with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
            pickle.dump({
                "aucs": aucs,
                "accuracies": accuracies,
                "f1_scores": f1_scores,
                "mses": mses,
                "loss_means": loss_means,
            }, f)

        return aucs, loss_means, accuracies, f1_scores, mses

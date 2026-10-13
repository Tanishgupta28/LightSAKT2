import os

import numpy as np
import torch
import pickle
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
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
                q, r, qshft, rshft, m = data

                self.train()
                y = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean_epoch.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    self.eval()
                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    y_pred = (y >= 0.5).int().numpy()
                    t_true = t.int().numpy()

                    auc = metrics.roc_auc_score(y_true=t_true, y_score=y.numpy())
                    acc = metrics.accuracy_score(y_true=t_true, y_pred=y_pred)
                    f1 = metrics.f1_score(y_true=t_true, y_pred=y_pred)
                    mse = metrics.mean_squared_error(y_true=t_true, y_pred=y.numpy())

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

                # Save all metrics for this epoch
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

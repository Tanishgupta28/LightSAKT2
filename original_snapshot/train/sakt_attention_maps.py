# import os
# import json
# import pickle
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader, random_split
# from torch.optim import Adam
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data_loaders.algebra2005 import Algebra2005
# from models.sakt import SAKT
# from models.utils import collate_fn
# from sklearn import metrics
# from torch.nn.functional import binary_cross_entropy

# class SAKTTrainer:
#     def __init__(self, hyperparams):
#         self.hyperparams = hyperparams
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.setup()

#     def setup(self):
#         hp = self.hyperparams
#         self.dataset = Algebra2005(hp["seq_len"])
#         train_size = int(len(self.dataset) * hp["train_ratio"])
#         test_size = len(self.dataset) - train_size
#         generator = torch.Generator(device=self.device)
#         self.train_dataset, self.test_dataset = random_split(
#             self.dataset, [train_size, test_size], generator=generator)
#         self.train_loader = DataLoader(
#             self.train_dataset, batch_size=hp["batch_size"], shuffle=True,
#             collate_fn=collate_fn, generator=generator
#         )
#         self.test_loader = DataLoader(
#             self.test_dataset, batch_size=test_size, shuffle=False,
#             collate_fn=collate_fn, generator=generator
#         )
#         self.model = SAKT(
#             self.dataset.num_q,
#             n=hp["seq_len"],
#             d=hp["hidden_dim"],
#             num_attn_heads=hp["num_attn_heads"],
#             dropout=hp["dropout"],
#         ).to(self.device)
#         self.optimizer = Adam(self.model.parameters(), lr=hp["learning_rate"])
#         self.ckpt_path = hp["checkpoint_dir"]
#         os.makedirs(self.ckpt_path, exist_ok=True)
#         os.makedirs(os.path.join(self.ckpt_path, "attention_heatmaps"), exist_ok=True)

#     def train(self):
#         epochs = self.hyperparams["num_epochs"]
#         for epoch in range(1, epochs + 1):
#             self.model.train()
#             train_losses = []
#             for data in self.train_loader:
#                 q, r, q_shft, r_shft, mask = data
#                 self.optimizer.zero_grad()
#                 p, _ = self.model(q.long().to(self.device), r.long().to(self.device), q_shft.long().to(self.device))
#                 p = torch.masked_select(p, mask)
#                 t = torch.masked_select(r_shft, mask)
#                 loss = binary_cross_entropy(p, t)
#                 loss.backward()
#                 self.optimizer.step()
#                 train_losses.append(loss.item())

#             avg_train_loss = np.mean(train_losses)

#             # Evaluate
#             self.model.eval()
#             with torch.no_grad():
#                 for data in self.test_loader:
#                     q, r, q_shft, r_shft, mask = data
#                     p, attn_weights = self.model(q.long().to(self.device), r.long().to(self.device), q_shft.long().to(self.device))
#                     p = torch.masked_select(p, mask).cpu()
#                     t = torch.masked_select(r_shft, mask).cpu()
#                     y_pred = (p >= 0.5).int().numpy()
#                     y_true = t.int().numpy()
#                     auc = metrics.roc_auc_score(y_true, p.numpy())
#                     print(f"Epoch {epoch}: AUC={auc:.4f}, Train Loss={avg_train_loss:.4f}")

#             # Save checkpoint and generate heatmap every 10 epochs
#             if epoch % 10 == 0:
#                 ckpt_file = os.path.join(self.ckpt_path, f"sakt_epoch{epoch}.pt")
#                 torch.save(self.model.state_dict(), ckpt_file)
#                 print(f"Saved checkpoint: {ckpt_file}")

#                 if attn_weights.ndim == 4:
#                     attn_map = attn_weights[0, 0].cpu().numpy()  # first head, first batch
#                 elif attn_weights.ndim == 3:
#                     attn_map = attn_weights[:, 0, :].cpu().numpy()  # seq_len x seq_len for first batch
#                 else:
#                     attn_map = attn_weights.cpu().numpy()

#                 print(f"Attention map shape at epoch {epoch} (should be square):", attn_map.shape)

#                 plt.figure(figsize=(10, 7))
#                 sns.heatmap(attn_map, cmap="viridis", cbar=True)
#                 plt.title(f"Attention Heatmap Epoch {epoch}")
#                 plt.xlabel("Keys (sequence positions)")
#                 plt.ylabel("Queries (sequence positions)")
#                 plt.tight_layout()
#                 heatmap_path = os.path.join(self.ckpt_path, "attention_heatmaps", f"attention_heatmap_epoch{epoch}.png")
#                 plt.savefig(heatmap_path)
#                 plt.close()
#                 print(f"Saved attention heatmap: {heatmap_path}")

# if __name__ == "__main__":
#     # Define your best hyperparameters here
#     best_hyperparams = {
#         "num_epochs": 50,
#         "train_ratio": 0.8,
#         "learning_rate": 0.0005,
#         "batch_size": 64,
#         "seq_len": 50,
#         "dropout": 0.2,
#         "hidden_dim": 128,
#         "num_attn_heads": 2,
#         "checkpoint_dir": "ckpts/sakt/Algebra2005/best_run_custom",
#     }

#     trainer = SAKTTrainer(best_hyperparams)
#     trainer.train()

    
    
    
    
    
# import os
# import json
# import pickle
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader, random_split
# from torch.optim import Adam
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data_loaders.algebra2005 import Algebra2005
# from models.sakt import SAKT
# from models.utils import collate_fn
# from sklearn import metrics
# from torch.nn.functional import binary_cross_entropy

# class SAKTTrainer:
#     def __init__(self, hyperparams):
#         self.hyperparams = hyperparams
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.setup()

#     def setup(self):
#         hp = self.hyperparams
#         self.dataset = Algebra2005(hp["seq_len"])
#         train_size = int(len(self.dataset) * hp["train_ratio"])
#         test_size = len(self.dataset) - train_size
#         generator = torch.Generator(device=self.device)
#         self.train_dataset, self.test_dataset = random_split(
#             self.dataset, [train_size, test_size], generator=generator)
#         self.train_loader = DataLoader(
#             self.train_dataset, batch_size=hp["batch_size"], shuffle=True,
#             collate_fn=collate_fn, generator=generator
#         )
#         self.test_loader = DataLoader(
#             self.test_dataset, batch_size=test_size, shuffle=False,
#             collate_fn=collate_fn, generator=generator
#         )
#         self.model = SAKT(
#             self.dataset.num_q,
#             n=hp["seq_len"],
#             d=hp["hidden_dim"],
#             num_attn_heads=hp["num_attn_heads"],
#             dropout=hp["dropout"],
#         ).to(self.device)
#         self.optimizer = Adam(self.model.parameters(), lr=hp["learning_rate"])
#         self.ckpt_path = hp["checkpoint_dir"]
#         os.makedirs(self.ckpt_path, exist_ok=True)
#         os.makedirs(os.path.join(self.ckpt_path, "attention_heatmaps"), exist_ok=True)

#     def train(self):
#         epochs = self.hyperparams["num_epochs"]
#         for epoch in range(1, epochs + 1):
#             self.model.train()
#             train_losses = []
#             for data in self.train_loader:
#                 q, r, q_shft, r_shft, mask = data
#                 self.optimizer.zero_grad()
#                 p, _ = self.model(q.long().to(self.device), r.long().to(self.device), q_shft.long().to(self.device))
#                 p = torch.masked_select(p, mask)
#                 t = torch.masked_select(r_shft, mask)
#                 loss = binary_cross_entropy(p, t)
#                 loss.backward()
#                 self.optimizer.step()
#                 train_losses.append(loss.item())

#             avg_train_loss = np.mean(train_losses)

#             # Evaluate
#             self.model.eval()
#             with torch.no_grad():
#                 for data in self.test_loader:
#                     q, r, q_shft, r_shft, mask = data
#                     p, attn_weights = self.model(q.long().to(self.device), r.long().to(self.device), q_shft.long().to(self.device))
#                     p = torch.masked_select(p, mask).cpu()
#                     t = torch.masked_select(r_shft, mask).cpu()
#                     y_pred = (p >= 0.5).int().numpy()
#                     y_true = t.int().numpy()
#                     auc = metrics.roc_auc_score(y_true, p.numpy())
#                     print(f"Epoch {epoch}: AUC={auc:.4f}, Train Loss={avg_train_loss:.4f}")

#             # Save checkpoint and generate heatmap every 10 epochs
#             if epoch % 10 == 0:
#                 ckpt_file = os.path.join(self.ckpt_path, f"sakt_epoch{epoch}.pt")
#                 torch.save(self.model.state_dict(), ckpt_file)
#                 print(f"Saved checkpoint: {ckpt_file}")

#                 if attn_weights.ndim == 4:
#                     attn_map = attn_weights[0, 0].cpu().numpy()  # first head, first batch
#                 elif attn_weights.ndim == 3:
#                     attn_map = attn_weights[:, 0, :].cpu().numpy()  # seq_len x seq_len for first batch
#                 else:
#                     attn_map = attn_weights.cpu().numpy()

#                 print(f"Attention map shape at epoch {epoch} (should be square):", attn_map.shape)

#                 plt.figure(figsize=(10, 7))
#                 sns.heatmap(attn_map, cmap="viridis", cbar=True)
#                 plt.title(f"Attention Heatmap Epoch {epoch}")
#                 plt.xlabel("Keys (sequence positions)")
#                 plt.ylabel("Queries (sequence positions)")
#                 plt.tight_layout()
#                 heatmap_path = os.path.join(self.ckpt_path, "attention_heatmaps", f"attention_heatmap_epoch{epoch}.png")
#                 plt.savefig(heatmap_path)
#                 plt.close()
#                 print(f"Saved attention heatmap: {heatmap_path}")

# if __name__ == "__main__":
#     # Define your best hyperparameters here
#     best_hyperparams = {
#         "num_epochs": 50,
#         "train_ratio": 0.8,
#         "learning_rate": 0.0005,
#         "batch_size": 64,
#         "seq_len": 50,
#         "dropout": 0.2,
#         "hidden_dim": 128,
#         "num_attn_heads": 2,
#         "checkpoint_dir": "ckpts/sakt/Algebra2005/best_run_custom",
#     }

#     trainer = SAKTTrainer(best_hyperparams)
#     trainer.train()
#                     
#                     
#                     
#                     
#                     
import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loaders.algebra2005 import Algebra2005
from models.sakt import SAKT
from models.utils import collate_fn
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy
torch.set_default_device("cpu")
class SAKTTrainer:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup()

    def setup(self):
        hp = self.hyperparams
        self.dataset = Algebra2005(hp["seq_len"])
        train_size = int(len(self.dataset) * hp["train_ratio"])
        test_size = len(self.dataset) - train_size

        generator = torch.Generator()
        generator.manual_seed(42)

        self.train_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, test_size], generator=generator
        )
        # self.train_loader = DataLoader(
        #     self.train_dataset, batch_size=hp["batch_size"], shuffle=True,
        #     collate_fn=collate_fn
        # )
        self.train_loader = DataLoader(
          self.train_dataset, 
          batch_size=hp["batch_size"], 
          shuffle=True,
          collate_fn=lambda batch: collate_fn(batch, device=self.device)  # <--- here
       )
        # self.test_loader = DataLoader(
        #     self.test_dataset, batch_size=test_size, shuffle=False,
        #     collate_fn=collate_fn
        # ) 
        self.test_loader = DataLoader(
           self.test_dataset, 
           batch_size=test_size, 
           shuffle=False,
           collate_fn=lambda batch: collate_fn(batch, device=self.device)  # <--- here
        )

        self.model = SAKT(
            self.dataset.num_q,
            n=hp["seq_len"],
            d=hp["hidden_dim"],
            num_attn_heads=hp["num_attn_heads"],
            dropout=hp["dropout"],
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=hp["learning_rate"])
        self.ckpt_path = hp["checkpoint_dir"]
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_path, "attention_heatmaps"), exist_ok=True)

    def train(self):
        epochs = self.hyperparams["num_epochs"]
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for data in self.train_loader:
                q, r, q_shft, r_shft, mask = data
                self.optimizer.zero_grad()
                p, _ = self.model(q.long().to(self.device), r.long().to(self.device), q_shft.long().to(self.device))
                p = torch.masked_select(p, mask)
                t = torch.masked_select(r_shft, mask)
                loss = binary_cross_entropy(p, t)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                for data in self.test_loader:
                    q, r, q_shft, r_shft, mask = data
                    p, attn_weights = self.model(q.long().to(self.device), r.long().to(self.device), q_shft.long().to(self.device))
                    p = torch.masked_select(p, mask).cpu()
                    t = torch.masked_select(r_shft, mask).cpu()
                    y_pred = (p >= 0.5).int().numpy()
                    y_true = t.int().numpy()
                    auc = metrics.roc_auc_score(y_true, p.numpy())
                    print(f"Epoch {epoch}: AUC={auc:.4f}, Train Loss={avg_train_loss:.4f}")

            # Save checkpoint and generate heatmap every 10 epochs
            if epoch % 10 == 0:
                ckpt_file = os.path.join(self.ckpt_path, f"sakt_epoch{epoch}.pt")
                torch.save(self.model.state_dict(), ckpt_file)
                print(f"Saved checkpoint: {ckpt_file}")

               
                if attn_weights.ndim == 4:  
                    # (batch, heads, seq, seq)
                    num_heads = attn_weights.shape[1]
                    for head in range(num_heads):
                        attn_map = attn_weights[0, head].cpu().numpy()
                        plt.figure(figsize=(10, 7))
                        sns.heatmap(attn_map, cmap="viridis", cbar=True)
                        plt.title(f"Attention Heatmap Epoch {epoch} - Head {head}")
                        plt.xlabel("Keys (sequence positions)")
                        plt.ylabel("Queries (sequence positions)")
                        plt.tight_layout()
                        heatmap_path = os.path.join(
                            self.ckpt_path, "attention_heatmaps",
                            f"attention_heatmap_epoch{epoch}_head{head}.png"
                        )
                        plt.savefig(heatmap_path)
                        plt.close()
                        print(f"Saved attention heatmap: {heatmap_path}")

                elif attn_weights.ndim == 3:
                    # (batch, seq, seq) -> single head only
                    attn_map = attn_weights[0].cpu().numpy()
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(attn_map, cmap="viridis", cbar=True)
                    plt.title(f"Attention Heatmap Epoch {epoch}")
                    plt.xlabel("Keys (sequence positions)")
                    plt.ylabel("Queries (sequence positions)")
                    plt.tight_layout()
                    heatmap_path = os.path.join(
                        self.ckpt_path, "attention_heatmaps",
                        f"attention_heatmap_epoch{epoch}.png"
                    )
                    plt.savefig(heatmap_path)
                    plt.close()
                    print(f"Saved attention heatmap: {heatmap_path}")

if __name__ == "__main__":
    best_hyperparams = {
        "num_epochs": 50,
        "train_ratio": 0.8,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "seq_len": 50,
        "dropout": 0.2,
        "hidden_dim": 128,
        "num_attn_heads": 2,
        "checkpoint_dir": "ckpts/sakt/Algebra2005/best_run_custom",
    }

    trainer = SAKTTrainer(best_hyperparams)
    trainer.train()


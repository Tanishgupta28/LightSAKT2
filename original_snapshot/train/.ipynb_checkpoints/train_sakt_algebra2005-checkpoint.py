# import os
# import argparse
# import json
# import pickle

# import torch
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from torch.utils.data import DataLoader, random_split
# from torch.optim import SGD, Adam

# from data_loaders.assist2009 import ASSIST2009
# from data_loaders.assist2018 import ASSIST2018
# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.statics2011 import Statics2011
# from models.dkt import DKT
# from models.dkt_plus import DKTPlus
# from models.dkvmn import DKVMN
# from models.sakt import SAKT
# from models.gkt import PAM, MHA       
# from models.utils import collate_fn


# def main(model_name, dataset_name):
#     if not os.path.isdir("ckpts"):
#         os.mkdir("ckpts")

#     ckpt_path = os.path.join("ckpts", model_name)
#     if not os.path.isdir(ckpt_path):
#         os.mkdir(ckpt_path)

#     ckpt_path = os.path.join(ckpt_path, dataset_name)
#     if not os.path.isdir(ckpt_path):
#         os.mkdir(ckpt_path)

#     with open("./config.json") as f:
#         config = json.load(f)
#         model_config = config[model_name]
#         train_config = config["train_config"]

#     batch_size = train_config["batch_size"]
#     num_epochs = train_config["num_epochs"]
#     train_ratio = train_config["train_ratio"]
#     learning_rate = train_config["learning_rate"]
#     optimizer = train_config["optimizer"]  # can be [sgd, adam]
#     seq_len = train_config["seq_len"]

#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "Statics2011":
#         dataset = Statics2011(seq_len)

#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"

#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     if model_name == "dkt":
#         model = DKT(dataset.num_q, **model_config).to(device)
#     elif model_name == "dkt+":
#         model = DKTPlus(dataset.num_q, **model_config).to(device)
#     elif model_name == "dkvmn":
#         model = DKVMN(dataset.num_q, **model_config).to(device)
#     elif model_name == "sakt":
#         model = SAKT(dataset.num_q, **model_config).to(device)
#     elif model_name == "gkt":
#         if model_config["method"] == "PAM":
#             model = PAM(dataset.num_q, **model_config).to(device)
#         elif model_config["method"] == "MHA":
#             model = MHA(dataset.num_q, **model_config).to(device)
#     else:
#         print("The wrong model name was used...")
#         return

#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     generator = torch.Generator(device=device)
#     train_dataset, test_dataset = random_split(
#         dataset, [train_size, test_size],generator=generator
#     )

#     if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
#         with open(
#             os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
#         ) as f:
#             train_dataset.indices = pickle.load(f)
#         with open(
#             os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
#         ) as f:
#             test_dataset.indices = pickle.load(f)
#     else:
#         with open(
#             os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
#         ) as f:
#             pickle.dump(train_dataset.indices, f)
#         with open(
#             os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
#         ) as f:
#             pickle.dump(test_dataset.indices, f)

#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         collate_fn=collate_fn,
#         generator=generator
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=test_size, shuffle=True,
#         collate_fn=collate_fn,
#         generator=generator
#     )

#     if optimizer == "sgd":
#         opt = SGD(model.parameters(), learning_rate, momentum=0.9)
#     elif optimizer == "adam":
#         opt = Adam(model.parameters(), learning_rate)

#     aucs, loss_means = \
#         model.train_model(
#             train_loader, test_loader, num_epochs, opt, ckpt_path
#         )

#     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
#         pickle.dump(aucs, f)
#     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
#         pickle.dump(loss_means, f)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="sakt",
#         help="The name of the model to train. \
#             The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
#             The default model is dkt."
#     )
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default="Algebra2005",
#         help="The name of the dataset to use in training. \
#             The possible datasets are in \
#             [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
#             The default dataset is ASSIST2009."
#     )
#     args = parser.parse_args()
#     print("trainig of SAKT on algebra 2005")
#     main(args.model_name, args.dataset_name)

# import os
# import sys
# import json
# import pickle
# import argparse
# import torch
# import itertools
# from torch.utils.data import DataLoader, random_split
# from torch.optim import Adam

# import matplotlib.pyplot as plt
# import numpy as np

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from data_loaders.assist2009 import ASSIST2009
# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.assist2018 import ASSIST2018

# from models.light_sakt import LightSAKT
# from models.utils import collate_fn


# def plot_training_curves(metrics, save_path):
#     plt.figure(figsize=(15, 10))
#     metric_names = ["auc", "accuracy", "f1", "mse", "loss"]
#     titles = {
#         "auc": "AUC over Epochs",
#         "accuracy": "Accuracy over Epochs",
#         "f1": "F1 Score over Epochs",
#         "mse": "MSE over Epochs",
#         "loss": "Loss over Epochs"
#     }
#     for i, metric in enumerate(metric_names):
#         plt.subplot(3, 2, i + 1)
#         plt.plot(metrics.get(metric, []), label=metric.upper())
#         plt.title(titles[metric])
#         plt.xlabel("Epoch")
#         plt.ylabel(metric.upper())
#         plt.grid(True)
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "training_curves.png"))
#     plt.close()


# def main(model_name, dataset_name, config_file):
#     with open(config_file) as f:
#         config = json.load(f)

#     model_config = config[model_name]
#     train_config = config["train_config"]

#     batch_size = train_config["batch_size"]
#     num_epochs = train_config["num_epochs"]
#     train_ratio = train_config["train_ratio"]
#     learning_rate = train_config["learning_rate"]
#     seq_len = train_config["seq_len"]

#     unique_folder = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_dropout{model_config.get('dropout', 0.1)}"
#     base_ckpt_dir = "ckpts"
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, unique_folder)
#     os.makedirs(ckpt_path, exist_ok=True)

#     print(f"Training {model_name} on {dataset_name} for {num_epochs} epochs with dropout {model_config.get('dropout', 0.1)}")

#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model_config["dropout"] = model_config.get("dropout", 0.1)
#     model = LightSAKT(dataset.num_q, **model_config).to(device)

#     print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     generator = torch.Generator(device=device)

#     train_idx_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
#     test_idx_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")

#     if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
#         with open(train_idx_path, "rb") as f:
#             train_indices = pickle.load(f)
#         with open(test_idx_path, "rb") as f:
#             test_indices = pickle.load(f)
#     else:
#         train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
#         with open(train_idx_path, "wb") as f:
#             pickle.dump(train_dataset.indices, f)
#         with open(test_idx_path, "wb") as f:
#             pickle.dump(test_dataset.indices, f)
#         train_indices, test_indices = train_dataset.indices, test_dataset.indices

#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         collate_fn=collate_fn, generator=generator
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=test_size, shuffle=False,
#         collate_fn=collate_fn, generator=generator
#     )

#     optimizer = Adam(model.parameters(), learning_rate)

#     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
#         train_loader, test_loader, num_epochs, optimizer, ckpt_path
#     )

#     # Convert aucs to floats if needed, to avoid formatting errors
#     aucs = [float(a) for a in aucs]

#     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
#         pickle.dump(aucs, f)
#     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
#         pickle.dump(loss_means, f)

#     metrics = {
#         "auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses,
#         "loss": loss_means
#     }

#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)

#     plot_training_curves(metrics, ckpt_path)

#     print("\n✅ Training completed!")
#     print(f"Best AUC:       {max(aucs):.4f}")
#     print(f"Final Accuracy: {accuracies[-1]:.4f}")
#     print(f"Final F1:       {f1_scores[-1]:.4f}")
#     print(f"Final MSE:      {mses[-1]:.4f}")
#     print(f"Final Loss:     {loss_means[-1]:.4f}")
#     print(f"Results saved to: {ckpt_path}")


# def grid_search(model_name, dataset_name, base_config_path="./config.json"):
#     with open(base_config_path) as f:
#         base_config = json.load(f)

#     param_grid = {
#         "train_config.num_epochs": [50, 100],
#         "train_config.train_ratio": [0.8, 0.9],
#         "train_config.learning_rate": [0.0005, 0.001],
#         "sakt.dropout": [0.1, 0.2]
#     }

#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_auc = -float("inf")
#     best_params = None

#     for params in all_combinations:
#         config = json.loads(json.dumps(base_config))  # Deep copy

#         for key, value in params.items():
#             section, param = key.split(".")
#             config[section][param] = value

#         temp_config_path = "./temp_config.json"
#         with open(temp_config_path, "w") as f:
#             json.dump(config, f, indent=4)

#         print(f"\nRunning grid search with params: {params}")
#         main(model_name, dataset_name, temp_config_path)

#         folder = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']['train_ratio']}_lr{config['train_config']['learning_rate']}_dropout{config['sakt'].get('dropout', 0.1)}"
#         metrics_path = os.path.join("ckpts", model_name, dataset_name, folder, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print("⚠️ Metrics file missing! Skipping this combination.")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         auc = max(metrics.get("auc", [0]))
#         print(f"AUC: {auc}")

#         if auc > best_auc:
#             best_auc = auc
#             best_params = params

#     print("\nGrid Search Complete!")
#   print(f"Best AUC: {max(map(float, aucs)):.4f}")
#     print(f"Best Hyperparameters: {best_params}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train or grid search Lightweight SAKT model")
#     parser.add_argument("--model_name", type=str, default="sakt", help="Model name (use 'sakt' for LightSAKT)")
#     parser.add_argument("--dataset_name", type=str, default="Algebra2005",
#                         choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "ASSIST2018"])
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
#     parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")

#     args = parser.parse_args()

#     if args.mode == "grid_search":
#         grid_search(args.model_name, args.dataset_name, args.config)
#     else:
#         main(args.model_name, args.dataset_name, args.config)




# import os
# import sys
# import json
# import pickle
# import argparse
# import itertools
# import torch
# from torch.utils.data import DataLoader, random_split
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# import copy

# # Add project root for imports
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # Datasets
# from data_loaders.assist2009 import ASSIST2009
# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.assist2018 import ASSIST2018

# # Model
# from models.sakt import SAKT  # Use SAKT model now as requested
# from models.utils import collate_fn


# def plot_training_curves(metrics, save_path):
#     os.makedirs(save_path, exist_ok=True)
#     plt.figure(figsize=(15, 10))
#     metric_names = ["auc", "accuracy", "f1", "mse", "loss"]
#     titles = {
#         "auc": "AUC over Epochs",
#         "accuracy": "Accuracy over Epochs",
#         "f1": "F1 Score over Epochs",
#         "mse": "MSE over Epochs",
#         "loss": "Loss over Epochs"
#     }
#     for i, metric in enumerate(metric_names):
#         plt.subplot(3, 2, i + 1)
#         if metrics.get(metric):
#             plt.plot(metrics[metric], label=metric.upper())
#         plt.title(titles[metric])
#         plt.xlabel("Epoch")
#         plt.ylabel(metric.upper())
#         plt.grid(True)
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "training_curves.png"))
#     plt.close()


# def safe_floats(lst, metric_name="value"):
#     floats = []
#     for x in lst:
#         try:
#             floats.append(float(x))
#         except (ValueError, TypeError):
#             print(f"⚠️ Warning: Skipping invalid {metric_name} value: {x}")
#     return floats


# def main(model_name, dataset_name, config_file):
#     with open(config_file) as f:
#         config = json.load(f)

#     model_config = config[model_name]
#     train_config = config["train_config"]

#     batch_size = train_config["batch_size"]
#     num_epochs = train_config["num_epochs"]
#     train_ratio = train_config["train_ratio"]
#     learning_rate = train_config["learning_rate"]
#     seq_len = train_config["seq_len"]

#     unique_folder = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_dropout{model_config.get('dropout', 0.1)}"
#     ckpt_path = os.path.join("ckpts", model_name, dataset_name, unique_folder)
#     os.makedirs(ckpt_path, exist_ok=True)

#     print(f"\n🚀 Training {model_name} on {dataset_name} for {num_epochs} epochs, dropout={model_config.get('dropout', 0.1)}")

#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     else:
#         raise ValueError(f"❌ Unsupported dataset: {dataset_name}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     generator = torch.Generator(device=device)

#     # Pass required args to SAKT constructor (example: num_q, n, d, num_attn_heads, dropout)
#     # Example values for d and num_attn_heads, adjust as per your config or defaults
#     d = model_config.get("d", 128)
#     num_attn_heads = model_config.get("num_attn_heads", 8)
#     seq_length = seq_len

#     model = SAKT(dataset.num_q, seq_length, d, num_attn_heads, model_config.get("dropout", 0.1)).to(device)

#     print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size

#     train_idx_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
#     test_idx_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")

#     if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
#         with open(train_idx_path, "rb") as f:
#             train_indices = pickle.load(f)
#         with open(test_idx_path, "rb") as f:
#             test_indices = pickle.load(f)
#     else:
#         train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
#         train_indices, test_indices = train_dataset.indices, test_dataset.indices
#         with open(train_idx_path, "wb") as f:
#             pickle.dump(train_indices, f)
#         with open(test_idx_path, "wb") as f:
#             pickle.dump(test_indices, f)

#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator)
#     test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, collate_fn=collate_fn, generator=generator)

#     optimizer = Adam(model.parameters(), learning_rate)

#     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
#         train_loader, test_loader, num_epochs, optimizer, ckpt_path
#     )

#     # Safely convert all metric lists to floats
#     aucs = safe_floats(aucs, "AUC")
#     accuracies = safe_floats(accuracies, "accuracy")
#     f1_scores = safe_floats(f1_scores, "F1")
#     mses = safe_floats(mses, "MSE")
#     loss_means = safe_floats(loss_means, "loss")

#     # Save metrics with keys consistent with train_model
#     metrics = {
#         "auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses,
#         "loss": loss_means,
#     }
#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)

#     plot_training_curves(metrics, ckpt_path)

#     print("\n✅ Training completed!")
#     print(f"Best AUC:       {max(aucs):.4f}" if aucs else "No valid AUC values")
#     print(f"Final Accuracy: {accuracies[-1]:.4f}" if accuracies else "No Accuracy recorded")
#     print(f"Final F1:       {f1_scores[-1]:.4f}" if f1_scores else "No F1 recorded")
#     print(f"Final MSE:      {mses[-1]:.4f}" if mses else "No MSE recorded")
#     print(f"Final Loss:     {loss_means[-1]:.4f}" if loss_means else "No Loss recorded")
#     print(f"Results saved to: {ckpt_path}")

#     return max(aucs) if aucs else -1.0


# def grid_search(model_name, dataset_name, base_config_path="./config.json"):
#     with open(base_config_path) as f:
#         base_config = json.load(f)

#     param_grid = {
#         "train_config.num_epochs": [50],
#         "train_config.train_ratio": [0.8],
#         "train_config.learning_rate": [0.0005, 0.001, 0.005],
#         f"{model_name}.dropout": [0.1, 0.2],
#     }

#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_auc = -float("inf")
#     best_params = None

#     for idx, params in enumerate(all_combinations):
#         config = copy.deepcopy(base_config)

#         for key, value in params.items():
#             section, param = key.split(".")
#             config[section][param] = value

#         temp_config_path = "./temp_config.json"
#         with open(temp_config_path, "w") as f:
#             json.dump(config, f, indent=4)

#         print(f"\n🔎 Grid search run {idx+1}/{len(all_combinations)} | Params: {params}")
#         auc = main(model_name, dataset_name, temp_config_path)

#         folder = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']['train_ratio']}_lr{config['train_config']['learning_rate']}_dropout{config[model_name].get('dropout', 0.1)}"
#         metrics_path = os.path.join("ckpts", model_name, dataset_name, folder, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print("⚠️ Metrics missing, skipping this run...")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         auc = max(metrics.get("auc", [0]))
#         print(f"AUC: {auc}")

#         if auc > best_auc:
#             best_auc = auc
#             best_params = params

#     print("\n🎯 Grid Search Complete!")
#     print(f"Best AUC: {best_auc}")
#     print(f"Best Hyperparameters: {best_params}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train or grid search SAKT model")
#     parser.add_argument("--model_name", type=str, default="sakt", help="Model name (use 'sakt')")
#     parser.add_argument("--dataset_name", type=str, default="Algebra2005",
#                         choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "ASSIST2018"])
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
#     parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")

#     args = parser.parse_args()

#     if args.mode == "grid_search":
#         grid_search(args.model_name, args.dataset_name, args.config)
#     else:
#         main(args.model_name, args.dataset_name, args.config)





# import os
# import sys
# import json
# import pickle
# import argparse
# import itertools
# import torch
# from torch.utils.data import DataLoader, random_split
# from torch.optim import SGD, Adam

# # Fix import path to allow data_loaders to be found
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from data_loaders.assist2009 import ASSIST2009
# from data_loaders.assist2018 import ASSIST2018
# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.statics2011 import Statics2011
# from models.dkvmn import DKVMN
# from models.utils import collate_fn


# def main(model_name, dataset_name, config_file):
#     with open(config_file) as f:
#         config = json.load(f)

#     model_config = config[model_name]
#     train_config = config["train_config"]

#     batch_size = train_config["batch_size"]
#     num_epochs = train_config["num_epochs"]
#     train_ratio = train_config["train_ratio"]
#     learning_rate = train_config["learning_rate"]
#     optimizer_type = train_config["optimizer"]
#     seq_len = train_config["seq_len"]

#     # Default dimensions/size for DKVMN if not set
#     if "dim_s" not in model_config:
#         model_config["dim_s"] = 50
#     if "size_m" not in model_config:
#         model_config["size_m"] = 20

#     # Create unique checkpoint folder to include dim_s and size_m
#     unique_folder_name = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_dim{model_config['dim_s']}_size{model_config['size_m']}"
#     base_ckpt_dir = "ckpts"
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, unique_folder_name)
#     os.makedirs(ckpt_path, exist_ok=True)

#     print(f"Training DKVMN on {dataset_name} for {num_epochs} epochs with dim_s={model_config['dim_s']} and size_m={model_config['size_m']}")

#     # Load dataset
#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "Statics2011":
#         dataset = Statics2011(seq_len)
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Save config copies
#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     # Initialize DKVMN model with dim_s and size_m included
#     model = DKVMN(dataset.num_q, **model_config).to(device)

#     # Split dataset
#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     generator = torch.Generator(device=device)
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

#     # Save or reuse split indices
#     train_idx_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
#     test_idx_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")
#     if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
#         with open(train_idx_path, "rb") as f:
#             train_dataset.indices = pickle.load(f)
#         with open(test_idx_path, "rb") as f:
#             test_dataset.indices = pickle.load(f)
#     else:
#         with open(train_idx_path, "wb") as f:
#             pickle.dump(train_dataset.indices, f)
#         with open(test_idx_path, "wb") as f:
#             pickle.dump(test_dataset.indices, f)

#     # Data loaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         collate_fn=collate_fn, generator=generator
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=test_size, shuffle=True,
#         collate_fn=collate_fn, generator=generator
#     )

#     # Setup optimizer
#     if optimizer_type == "sgd":
#         opt = SGD(model.parameters(), learning_rate, momentum=0.9)
#     elif optimizer_type == "adam":
#         opt = Adam(model.parameters(), learning_rate)
#     else:
#         raise ValueError(f"Unknown optimizer: {optimizer_type}")

#     # Train and collect metrics
#     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
#         train_loader, test_loader, num_epochs, opt, ckpt_path
#     )

#     # Save metrics
#     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
#         pickle.dump(aucs, f)
#     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
#         pickle.dump(loss_means, f)

#     metrics = {
#         "train_loss": loss_means,
#         "valid_auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses
#     }
#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)

#     print(f"✅ DKVMN Training complete. Metrics saved in {ckpt_path}")


# def grid_search(model_name, dataset_name, config_file="./config.json"):
#     with open(config_file) as f:
#         base_config = json.load(f)

#     # Define grid for dim_s and size_m in addition to standard params
#     param_grid = {
#         "train_config.num_epochs": [50],
#         "train_config.train_ratio": [0.8],
#         "train_config.learning_rate": [0.0005, 0.001,0.005],
#         "dkvmn.dim_s": [50, 100],
#         "dkvmn.size_m": [20, 40]
#     }

#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

#     best_auc = -float("inf")
#     best_params = None

#     for params in all_combinations:
#         config = json.loads(json.dumps(base_config))  # deep copy
#         for key, value in params.items():
#             section, param = key.split(".")
#             config[section][param] = value

#         temp_config_path = "./temp_config.json"
#         with open(temp_config_path, "w") as f:
#             json.dump(config, f, indent=4)

#         print(f"\nRunning grid search with params: {params}")
#         main(model_name, dataset_name, temp_config_path)

#         folder_name = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']['train_ratio']}_lr{config['train_config']['learning_rate']}_dim{config['dkvmn']['dim_s']}_size{config['dkvmn']['size_m']}"
#         metrics_path = os.path.join("ckpts", model_name, dataset_name, folder_name, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print("⚠️ Metrics file missing! Skipping this combination.")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         auc = max(metrics.get("valid_auc", [0]))
#         print(f"AUC: {auc}")

#         if auc > best_auc:
#             best_auc = auc
#             best_params = params

#     print("\nGrid Search Complete!")
#     print(f"Best AUC: {best_auc}")
#     print(f"Best Hyperparameters: {best_params}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, default="train", help="Train or grid_search mode")
#     parser.add_argument("--model_name", type=str, default="sakt", help="Model to train (default: dkvmn)")
#     parser.add_argument("--dataset_name", type=str, default="Algebra2005", help="Dataset to use (default: Algebra2005)")
#     parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
#     args = parser.parse_args()

#     if args.mode == "grid_search":
#         grid_search(args.model_name, args.dataset_name, args.config)
#     else:
#         main(args.model_name, args.dataset_name, args.config)




# import os
# import argparse
# import json
# import pickle
# import torch
# import sys
# import itertools
# import copy
# from torch.utils.data import DataLoader, random_split
# from torch.optim import SGD, Adam

# # Add parent directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from data_loaders.assist2009 import ASSIST2009
# from data_loaders.assist2018 import ASSIST2018
# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.statics2011 import Statics2011
# from models.dkt import DKT
# from models.dkt_plus import DKTPlus
# from models.dkvmn import DKVMN
# from models.sakt import SAKT
# from models.gkt import PAM, MHA
# from models.utils import collate_fn


# def train_and_evaluate(model_name, dataset_name, config):
#     base_ckpt_dir = os.path.join("ckpts")
#     os.makedirs(os.path.join(base_ckpt_dir, model_name, dataset_name), exist_ok=True)
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name)

#     model_config = config[model_name]
#     train_config = config["train_config"]

#     batch_size = train_config["batch_size"]
#     num_epochs = train_config["num_epochs"]
#     train_ratio = train_config["train_ratio"]
#     learning_rate = train_config["learning_rate"]
#     optimizer_type = train_config["optimizer"]
#     seq_len = train_config["seq_len"]

#     # Load dataset
#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "Statics2011":
#         dataset = Statics2011(seq_len)
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Save config copies
#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     # Initialize model
#     if model_name == "dkt":
#         model = DKT(dataset.num_q, **model_config).to(device)
#     elif model_name == "dkt+":
#         model = DKTPlus(dataset.num_q, **model_config).to(device)
#     elif model_name == "dkvmn":
#         model = DKVMN(dataset.num_q, **model_config).to(device)
#     elif model_name == "sakt":
#         model = SAKT(dataset.num_q, **model_config).to(device)
#     elif model_name == "gkt":
#         if model_config["method"] == "PAM":
#             model = PAM(dataset.num_q, **model_config).to(device)
#         elif model_config["method"] == "MHA":
#             model = MHA(dataset.num_q, **model_config).to(device)
#         else:
#             raise ValueError(f"Invalid method for GKT: {model_config['method']}")
#     else:
#         raise ValueError(f"Unknown model: {model_name}")

#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     generator = torch.Generator(device=device)

#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

#     # Store/reuse split indices
#     train_idx_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
#     test_idx_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")
#     if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
#         with open(train_idx_path, "rb") as f:
#             train_dataset.indices = pickle.load(f)
#         with open(test_idx_path, "rb") as f:
#             test_dataset.indices = pickle.load(f)
#     else:
#         with open(train_idx_path, "wb") as f:
#             pickle.dump(train_dataset.indices, f)
#         with open(test_idx_path, "wb") as f:
#             pickle.dump(test_dataset.indices, f)

#     # Data loaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         collate_fn=collate_fn, generator=generator
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=test_size, shuffle=False,
#         collate_fn=collate_fn, generator=generator
#     )

#     # Optimizer
#     if optimizer_type == "sgd":
#         opt = SGD(model.parameters(), learning_rate, momentum=0.9)
#     elif optimizer_type == "adam":
#         opt = Adam(model.parameters(), learning_rate)
#     else:
#         raise ValueError(f"Unknown optimizer: {optimizer_type}")

#     # Train model and collect metrics
#     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
#         train_loader, test_loader, num_epochs, opt, ckpt_path
#     )

#     # Save individual metric files
#     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
#         pickle.dump(aucs, f)
#     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
#         pickle.dump(loss_means, f)

#     # Save combined metrics
#     metrics = {
#         "train_loss": loss_means,
#         "valid_auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses,
#     }
#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)

#     return aucs, metrics, ckpt_path


# def grid_search(model_name, dataset_name, base_config_path="./config.json"):
#     with open(base_config_path) as f:
#         base_config = json.load(f)

#     # Define hyperparameter grid for search
#     param_grid = {
#         "train_config.num_epochs": [50],
#         "train_config.train_ratio": [0.8],
#         "train_config.learning_rate": [0.0005, 0.001, 0.005],
#         # "train_config.optimizer": ["adam", "sgd"],
#         f"{model_name}.dropout": [0.1, 0.2],
#     }

#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_auc = -float("inf")
#     best_params = None

#     for idx, params in enumerate(all_combinations):
#         config = copy.deepcopy(base_config)

#         # Update config with current params
#         for key, value in params.items():
#             section, param = key.split(".", 1)
#             if section in config:
#                 config[section][param] = value
#             else:
#                 config[section] = {param: value}

#         print(f"\n🔎 Grid search run {idx+1}/{len(all_combinations)} | Params: {params}")
#         aucs, metrics, ckpt_path = train_and_evaluate(model_name, dataset_name, config)

#         max_auc = max(aucs) if aucs else 0
#         print(f"AUC: {max_auc}")

#         if max_auc > best_auc:
#             best_auc = max_auc
#             best_params = params

#     print("\n🎯 Grid Search Complete!")
#     print(f"Best AUC: {best_auc}")
#     print(f"Best Hyperparameters: {best_params}")


# def main(model_name, dataset_name, mode, config_path="./config.json"):
#     with open(config_path) as f:
#         config = json.load(f)

#     if mode == "grid_search":
#         grid_search(model_name, dataset_name, config_path)
#     else:
#         aucs, metrics, ckpt_path = train_and_evaluate(model_name, dataset_name, config)
#         print(f"✅ Training complete. Metrics saved in {ckpt_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="sakt",
#                         help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
#     parser.add_argument("--dataset_name", type=str, default="Algebra2005",
#                         help="Dataset name")
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
#     parser.add_argument("--config", type=str, default="./config.json",
#                         help="Path to config file")

#     args = parser.parse_args()

#     print(f"🚀 {args.mode.capitalize()} {args.model_name} on {args.dataset_name}")
#     main(args.model_name, args.dataset_name, args.mode, args.config)



import os
import argparse
import json
import pickle
import torch
import sys
import itertools
import copy
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2018 import ASSIST2018
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA
from models.utils import collate_fn


def train_and_evaluate(model_name, dataset_name, config, run_name=None):
    base_ckpt_dir = os.path.join("ckpts")
    os.makedirs(base_ckpt_dir, exist_ok=True)

    # Unique run folder
    if run_name:
        ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, run_name)
    else:
        ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name)

    os.makedirs(ckpt_path, exist_ok=True)

    model_config = config[model_name]
    train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer_type = train_config["optimizer"]
    seq_len = train_config["seq_len"]

    # Load dataset
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2018":
        dataset = ASSIST2018(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save config copies
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    # Initialize model
    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(dataset.num_q, **model_config).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(dataset.num_q, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset.num_q, **model_config).to(device)
    elif model_name == "gkt":
        if model_config["method"] == "PAM":
            model = PAM(dataset.num_q, **model_config).to(device)
        elif model_config["method"] == "MHA":
            model = MHA(dataset.num_q, **model_config).to(device)
        else:
            raise ValueError(f"Invalid method for GKT: {model_config['method']}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator(device=device)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    # Store/reuse split indices
    train_idx_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
    test_idx_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")
    if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
        with open(train_idx_path, "rb") as f:
            train_dataset.indices = pickle.load(f)
        with open(test_idx_path, "rb") as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(train_idx_path, "wb") as f:
            pickle.dump(train_dataset.indices, f)
        with open(test_idx_path, "wb") as f:
            pickle.dump(test_dataset.indices, f)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False,
        collate_fn=collate_fn, generator=generator
    )

    # Optimizer
    if optimizer_type == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer_type == "adam":
        opt = Adam(model.parameters(), learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Train model and collect metrics
    aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
        train_loader, test_loader, num_epochs, opt, ckpt_path
    )

    # Save individual metric files
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)

    # Save combined metrics
    metrics = {
        "train_loss": loss_means,
        "valid_auc": aucs,
        "accuracy": accuracies,
        "f1": f1_scores,
        "mse": mses,
    }
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    return aucs, metrics, ckpt_path


def grid_search(model_name, dataset_name, base_config_path="./config.json"):
    with open(base_config_path) as f:
        base_config = json.load(f)

    # Define hyperparameter grid
    param_grid = {
        "train_config.num_epochs": [50],
        "train_config.train_ratio": [0.8],
        "train_config.learning_rate": [0.0005, 0.001, 0.005],
        f"{model_name}.dropout": [0.1, 0.2],
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_auc = -float("inf")
    best_params = None
    best_ckpt_path = None

    for idx, params in enumerate(all_combinations):
        config = copy.deepcopy(base_config)

        # Update config with current params
        for key, value in params.items():
            section, param = key.split(".", 1)
            if section in config:
                config[section][param] = value
            else:
                config[section] = {param: value}

        # Create readable run name with hyperparams
        run_name = f"run_{idx+1}_" + "_".join([f"{k.replace('.', '')}{v}" for k, v in params.items()])

        print(f"\n🔎 Grid search run {idx+1}/{len(all_combinations)} | Params: {params}")

        aucs, metrics, ckpt_path = train_and_evaluate(model_name, dataset_name, config, run_name=run_name)

        max_auc = max(aucs) if aucs else 0
        print(f"AUC: {max_auc} | Saved in: {ckpt_path}")

        if max_auc > best_auc:
            best_auc = max_auc
            best_params = params
            best_ckpt_path = ckpt_path

    print("\n🎯 Grid Search Complete!")
    print(f"Best AUC: {best_auc}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"🏆 Best run saved in: {best_ckpt_path}")


def main(model_name, dataset_name, mode, config_path="./config.json"):
    with open(config_path) as f:
        config = json.load(f)

    if mode == "grid_search":
        grid_search(model_name, dataset_name, config_path)
    else:
        aucs, metrics, ckpt_path = train_and_evaluate(model_name, dataset_name, config)
        print(f"✅ Training complete. Metrics saved in {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sakt",
                        help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
    parser.add_argument("--dataset_name", type=str, default="Algebra2005",
                        help="Dataset name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
    parser.add_argument("--config", type=str, default="./config.json",
                        help="Path to config file")

    args = parser.parse_args()

    print(f"🚀 {args.mode.capitalize()} {args.model_name} on {args.dataset_name}")
    main(args.model_name, args.dataset_name, args.mode, args.config)

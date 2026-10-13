# # import os
# # import argparse
# # import json
# # import pickle
# # import torch
# # from torch.utils.data import DataLoader, random_split
# # from torch.optim import Adam
# # import sys
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# # from data_loaders.assist2009 import ASSIST2009
# # from data_loaders.assist2015 import ASSIST2015
# # from data_loaders.algebra2005 import Algebra2005
# # from data_loaders.assist2018 import ASSIST2018
# # from models.light_sakt import LightSAKT
# # from models.utils import collate_fn
# # import matplotlib.pyplot as plt
# # import numpy as np

# # def plot_training_curves(metrics, save_path):
# #     """Plot and save training curves for all metrics"""
# #     plt.figure(figsize=(15, 10))

# #     metric_names = ["auc", "accuracy", "f1", "mse", "loss"]
# #     titles = {
# #         "auc": "AUC over Epochs",
# #         "accuracy": "Accuracy over Epochs",
# #         "f1": "F1 Score over Epochs",
# #         "mse": "MSE over Epochs",
# #         "loss": "Loss over Epochs"
# #     }

# #     for i, metric in enumerate(metric_names):
# #         plt.subplot(3, 2, i + 1)
# #         plt.plot(metrics[metric], label=metric.upper())
# #         plt.title(titles[metric])
# #         plt.xlabel("Epoch")
# #         plt.ylabel(metric.upper())
# #         plt.grid(True)
# #         plt.legend()

# #     plt.tight_layout()
# #     plt.savefig(os.path.join(save_path, 'training_curves.png'))
# #     plt.close()

# # def main(dataset_name, config_path="./config.json"):
# #     # Create checkpoint directory
# #     ckpt_path = os.path.join("ckpts", "light_sakt", dataset_name)
# #     os.makedirs(ckpt_path, exist_ok=True)

# #     # Load configuration
# #     with open(config_path) as f:
# #         config = json.load(f)
# #         model_config = config["sakt"]
# #         train_config = config["train_config"]
    
# #     model_config["dropout"] = 0.1  # Adjust dropout

# #     # Training parameters
# #     batch_size = train_config["batch_size"]
# #     num_epochs = train_config["num_epochs"]
# #     train_ratio = train_config["train_ratio"]
# #     learning_rate = train_config["learning_rate"]
# #     seq_len = train_config["seq_len"]

# #     # Load dataset
# #     if dataset_name == "ASSIST2009":
# #         dataset = ASSIST2009(seq_len)
# #     elif dataset_name == "ASSIST2015":
# #         dataset = ASSIST2015(seq_len)
# #     elif dataset_name == "Algebra2005":
# #         dataset = Algebra2005(seq_len)
# #     elif dataset_name == "ASSIST2018":
# #         dataset = ASSIST2018(seq_len)
# #     else:
# #         raise ValueError(f"Unknown dataset: {dataset_name}")

# #     # Set device
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print(f"Using device: {device}")

# #     # Save config files
# #     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
# #         json.dump(model_config, f, indent=4)
# #     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
# #         json.dump(train_config, f, indent=4)

# #     # Initialize model
# #     model = LightSAKT(dataset.num_q, **model_config).to(device)
# #     print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# #     # Train/test split
# #     train_size = int(len(dataset) * train_ratio)
# #     test_size = len(dataset) - train_size
# #     generator = torch.Generator(device=device)
# #     train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)


# #     # Save split indices
# #     indices_path = os.path.join(dataset.dataset_dir, "light_sakt_indices")
# #     os.makedirs(indices_path, exist_ok=True)
# #     with open(os.path.join(indices_path, "train_indices.pkl"), "wb") as f:
# #         pickle.dump(train_dataset.indices, f)
# #     with open(os.path.join(indices_path, "test_indices.pkl"), "wb") as f:
# #         pickle.dump(test_dataset.indices, f)

# #     # Data loaders
# #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,generator=generator)
# #     test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, collate_fn=collate_fn,generator=generator)

# #     # Optimizer
# #     optimizer = Adam(model.parameters(), learning_rate)

# #     # Train
# #     print(f"Starting training on {dataset_name}...")
# #     metrics = model.train_model(train_loader, test_loader, num_epochs, optimizer, ckpt_path)

# #     # Save metrics
# #     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
# #         pickle.dump(metrics, f)

# #     # Plot metrics
# #     plot_training_curves(metrics, ckpt_path)

# #     # Print final results
# #     print("\nTraining completed!")
# #     print(f"Best AUC:      {max(metrics['auc']):.4f}")
# #     print(f"Final Accuracy:{metrics['accuracy'][-1]:.4f}")
# #     print(f"Final F1:      {metrics['f1'][-1]:.4f}")
# #     print(f"Final MSE:     {metrics['mse'][-1]:.4f}")
# #     print(f"Final Loss:    {metrics['loss'][-1]:.4f}")
# #     print(f"Results saved to: {ckpt_path}")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Train Lightweight SAKT model")
# #     parser.add_argument(
# #         "--dataset",
# #         type=str,
# #         default="Algebra2005",
# #         choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "ASSIST2018"],
# #         help="Dataset to use for training"
# #     )
# #     parser.add_argument(
# #         "--config",
# #         type=str,
# #         default="./config.json",
# #         help="Path to configuration file"
# #     )

# #     args = parser.parse_args()
# #     main(args.dataset, args.config)


# # 
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

#     # Checkpoint directory with hyperparameter naming
#     unique_folder = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_dropout{model_config.get('dropout', 0.1)}"
#     base_ckpt_dir = "ckpts"
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, unique_folder)
#     os.makedirs(ckpt_path, exist_ok=True)

#     print(f"Training {model_name} on {dataset_name} for {num_epochs} epochs...")

#     # Load dataset
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

#     # Add default dropout if missing
#     model_config["dropout"] = model_config.get("dropout", 0.1)
#     model = LightSAKT(dataset.num_q, **model_config).to(device)
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

#     # Save config copies
#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     # Dataset split with fixed seed
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

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               collate_fn=collate_fn, generator=generator)
#     test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False,
#                              collate_fn=collate_fn, generator=generator)

#     optimizer = Adam(model.parameters(), lr=learning_rate)

#     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
#         train_loader, test_loader, num_epochs, optimizer, ckpt_path
#     )

#     # Save metrics
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
#     print(f"Best AUC: {max(map(float, aucs)):.4f}")
#     print(f"Final Accuracy: {accuracies[-1]:.4f}")
#     print(f"Final F1: {f1_scores[-1]:.4f}")
#     print(f"Final MSE: {mses[-1]:.4f}")
#     print(f"Final Loss: {loss_means[-1]:.4f}")
#     print(f"Results saved to: {ckpt_path}")


# def grid_search(model_name, dataset_name, config_file):
#     with open(config_file) as f:
#         base_config = json.load(f)

#     param_grid = {
#         "train_config.num_epochs": [50],
#         "train_config.train_ratio": [0.8],
#         "train_config.learning_rate": [0.0005, 0.001],
#         "sakt.dropout": [0.1, 0.2]
#     }

#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     best_auc = -float('inf')
#     best_params = None

#     for params in all_combinations:
#         config = json.loads(json.dumps(base_config))  # deep copy

#         for key, value in params.items():
#             section, param = key.split('.')
#             config[section][param] = value

#         temp_config_path = "./temp_config.json"
#         with open(temp_config_path, "w") as f:
#             json.dump(config, f, indent=4)

#         print(f"\nRunning grid search with params: {params}")
#         main(model_name, dataset_name, temp_config_path)

#         folder = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']['train_ratio']}_lr{config['train_config']['learning_rate']}_dropout{config['sakt'].get('dropout', 0.1)}"
#         metrics_path = os.path.join("ckpts", model_name, dataset_name, folder, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print("⚠️ Metrics file missing! Skipping this combo.")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         auc = max(metrics.get("auc", [0]))
#         print(f"AUC: {auc}")

#         if auc > best_auc:
#             best_auc = auc
#             best_params = params

#     print("\nGrid Search Complete!")
#     print(f"Best AUC: {best_auc}")
#     print(f"Best Hyperparameters: {best_params}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train or grid search Lightweight SAKT model")
#     parser.add_argument("--model_name", type=str, default="sakt", help="Model name (always 'sakt' or 'light_sakt')")
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

# # Ensure data loaders and models are in the path
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
#     plt.savefig(os.path.join(save_path, 'training_curves.png'))
#     plt.close()


# def safe_floats(lst):
#     floats = []
#     for x in lst:
#         try:
#             floats.append(float(x))
#         except (ValueError, TypeError):
#             continue
#     return floats


# def main(model_name, dataset_name, config_path):
#     with open(config_path) as f:
#         config = json.load(f)
#     model_config = config[model_name]
#     train_config = config["train_config"]

#     batch_size = train_config["batch_size"]
#     num_epochs = train_config["num_epochs"]
#     train_ratio = train_config["train_ratio"]
#     learning_rate = train_config["learning_rate"]
#     seq_len = train_config["seq_len"]

#     # Set default dropout if not present
#     model_config["dropout"] = model_config.get("dropout", 0.1)

#     unique_folder = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_d{model_config.get('d','NA')}_drop{model_config.get('dropout',0.1)}"
#     base_ckpt_dir = "ckpts"
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, unique_folder)
#     os.makedirs(ckpt_path, exist_ok=True)

#     print(f"Training {model_name} on {dataset_name} for {num_epochs} epochs...")

#     # Load dataset
#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Save configs used to train
#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     model = LightSAKT(dataset.num_q, **model_config).to(device)
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     generator = torch.Generator(device=device)

#     indices_path = os.path.join(dataset.dataset_dir, f"{model_name}_indices")
#     os.makedirs(indices_path, exist_ok=True)
#     train_idx_path = os.path.join(indices_path, "train_indices.pkl")
#     test_idx_path = os.path.join(indices_path, "test_indices.pkl")

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

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               collate_fn=collate_fn, generator=generator)
#     test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False,
#                              collate_fn=collate_fn, generator=generator)

#     optimizer = Adam(model.parameters(), lr=learning_rate)

#     metrics = model.train_model(train_loader, test_loader, num_epochs, optimizer, ckpt_path)

#     aucs = metrics["auc"]
#     loss_means = metrics["loss"]
#     accuracies = metrics["accuracy"]
#     f1_scores = metrics["f1"]
#     mses = metrics["mse"]
    
#     metrics = {
#     "auc": aucs,
#     "accuracy": accuracies,
#     "f1": f1_scores,
#     "mse": mses,
#     "loss": loss_means,
#     }

#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)
#         # pickle.dump({
#         #     "auc": aucs,
#         #     "accuracy": accuracies,
#         #     "f1": f1_scores,
#         #     "mse": mses,
#         #     "loss": loss_means,
#         # }, f)

#     plot_training_curves({
#         "auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses,
#         "loss": loss_means
#     }, ckpt_path)

#     # Safely print best AUC
#     valid_aucs = safe_floats(aucs)

#     if valid_aucs:
#         print(f"Best AUC: {max(valid_aucs):.4f}")
#     else:
#         print("No valid AUC values found.")

#     print(f"Final Accuracy: {accuracies[-1]:.4f}")
#     print(f"Final F1: {f1_scores[-1]:.4f}")
#     print(f"Final MSE: {mses[-1]:.4f}")
#     print(f"Final Loss: {loss_means[-1]:.4f}")
#     print(f"Results saved to: {ckpt_path}")


# def grid_search(model_name, dataset_name, config_file):
#     with open(config_file) as f:
#         base_config = json.load(f)

#     param_grid = {
#         "train_config.num_epochs": [50],
#         "train_config.train_ratio": [0.8],
#         "train_config.learning_rate": [0.0005, 0.001,0.005],
#         "light_sakt.d": [64, 100, 128],
#         "light_sakt.dropout": [0.2],
#     }

#     keys, values = zip(*param_grid.items())
#     all_combinations = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

#     best_auc = -float('inf')
#     best_params = None

#     for params in all_combinations:
#         config = json.loads(json.dumps(base_config))  # Deep copy

#         for key, value in params.items():
#             section, param = key.split('.')
#             config[section][param] = value

#         temp_config_path = "./temp_config.json"
#         with open(temp_config_path, "w") as f:
#             json.dump(config, f, indent=4)

#         print(f"\nRunning grid search with params: {params}")
#         main(model_name, dataset_name, temp_config_path)

#         folder = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']      ['train_ratio']}_lr{config['train_config']['learning_rate']}_d{config['light_sakt']['d']}_drop{config['light_sakt'].get('dropout', 0.1)}"
#         metrics_path = os.path.join("ckpts", model_name, dataset_name, folder, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print("⚠️ Metrics missing for this parameter combination, skipping.")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         valid_aucs = safe_floats(metrics.get("auc", []))
#         if not valid_aucs:
#             print("⚠️ No valid AUC metrics found, skipping.")
#             continue

#         auc = max(valid_aucs)
#         print(f"AUC: {auc}")

#         if auc > best_auc:
#             best_auc = auc
#             best_params = params

#     print("\nGrid Search complete!")
#     print(f"Best AUC: {best_auc}")
#     print(f"Best Hyperparameters: {best_params}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Lightweight SAKT training and hyperparameter grid search")
#     parser.add_argument("--model_name", type=str, default="light_sakt",help="Model name key in config")
#     parser.add_argument("--dataset_name", type=str, default="Algebra2005", choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "ASSIST2018"])
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
#     parser.add_argument("--config", type=str, default="./config.json", help="Path to JSON config")

#     args = parser.parse_args()

#     if args.mode == "grid_search":
#         grid_search(args.model_name, args.dataset_name, args.config)
#     else:
#         main(args.model_name, args.dataset_name, args.config)
import os
import sys
import json
import pickle
import argparse
import torch
import itertools
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loaders.algebra2005 import Algebra2005
from models.light_sakt import Light_SAKT  # Make sure Light_SAKT is correctly imported
from models.utils import collate_fn

def main(model_name, dataset_name, config_file):
    with open(config_file) as f:
        config = json.load(f)

    model_config = config[model_name]
    train_config = config["train_config"]


    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    seq_len = train_config["seq_len"]
    base_ckpt_dir = "ckpts"
    folder_name = (
        f"epochs{num_epochs}"
        f"_tr{train_ratio}"
        f"_lr{learning_rate}"
        f"_embed{model_config['embed_dim']}"
        f"_heads{model_config['num_heads']}"
        f"_drop{model_config['dropout']}"
    )
    ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, folder_name)
    os.makedirs(ckpt_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# Split dataset
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator(device=device)  # <- fix here
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False,
        collate_fn=collate_fn, generator=generator
    )


    # Initialize model
    model = Light_SAKT(dataset.num_q, **model_config).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train model
    aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
        train_loader, test_loader, num_epochs, optimizer, ckpt_path
    )

    # Save metrics
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump({
            "train_loss": loss_means,
            "valid_auc": aucs,
            "accuracy": accuracies,
            "f1": f1_scores,
            "mse": mses
        }, f)

    print(f"✅ Training complete. Metrics saved in {ckpt_path}")
def grid_search(model_name, dataset_name, config_file):
    with open(config_file) as f:
        base_config = json.load(f)

    param_grid = {
     "train_config.num_epochs": [50],
     "train_config.train_ratio": [0.8],
     "train_config.learning_rate": [0.0005, 0.001, 0.005],
     "light_sakt.embed_dim": [32, 64, 128],
     "light_sakt.num_heads": [4],
     "light_sakt.dropout": [0.1, 0.2],
  }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    best_auc = -float("inf")
    best_params = None
    best_run_dir = None

    for params in all_combinations:
        config = json.loads(json.dumps(base_config))  # deep copy

        for key, value in params.items():
            section, param = key.split(".")
            config[section][param] = value

        temp_config_path = "./temp_config.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"\nRunning grid search with params: {params}")
        main(model_name, dataset_name, temp_config_path)

        # reconstruct folder name (same as in main)
        folder = (
        f"epochs{config['train_config']['num_epochs']}"
        f"_tr{config['train_config']['train_ratio']}"
        f"_lr{config['train_config']['learning_rate']}"
        f"_embed{config['light_sakt']['embed_dim']}"
        f"_heads{config['light_sakt']['num_heads']}"
        f"_drop{config['light_sakt']['dropout']}"
       )
        run_dir = os.path.join("ckpts", model_name, dataset_name, folder)
        metrics_path = os.path.join(run_dir, "metrics.pkl")

        if not os.path.exists(metrics_path):
            print("⚠️ Metrics missing, skipping.")
            continue

        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

        auc = max(metrics.get("valid_auc", [0]))
        print(f"AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_run_dir = run_dir

    print("\n✅ Grid Search complete!")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"📂 Best run saved in: {best_run_dir}")

    # save best directory for easy lookup
    with open("best_run.txt", "w") as f:
        f.write(best_run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or grid_search")
    parser.add_argument("--model_name", type=str, default="light_sakt", help="Model to train")
    parser.add_argument("--dataset_name", type=str, default="Algebra2005", help="Dataset to use")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
    args = parser.parse_args()

    if args.mode == "grid_search":
        grid_search(args.model_name, args.dataset_name, args.config)
    else:
        main(args.model_name, args.dataset_name, args.config)

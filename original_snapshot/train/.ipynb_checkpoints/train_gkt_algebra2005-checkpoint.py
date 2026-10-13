# # import os
# # import argparse
# # import json
# # import pickle

# # import torch
# # import sys
# # import os
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# # from torch.utils.data import DataLoader, random_split
# # from torch.optim import SGD, Adam

# # from data_loaders.assist2009 import ASSIST2009
# # from data_loaders.assist2018 import ASSIST2018
# # from data_loaders.assist2015 import ASSIST2015
# # from data_loaders.algebra2005 import Algebra2005
# # from data_loaders.statics2011 import Statics2011
# # from models.dkt import DKT
# # from models.dkt_plus import DKTPlus
# # from models.dkvmn import DKVMN
# # from models.sakt import SAKT
# # from models.gkt import GKT, PAM, MHA
# # from models.utils import collate_fn


# # def main(model_name, dataset_name):
# #     print("Entered main()")
# #     if not os.path.isdir("ckpts"):
# #         os.mkdir("ckpts")

# #     ckpt_path = os.path.join("ckpts", model_name)
# #     if not os.path.isdir(ckpt_path):
# #         os.mkdir(ckpt_path)

# #     ckpt_path = os.path.join(ckpt_path, dataset_name)
# #     if not os.path.isdir(ckpt_path):
# #         os.mkdir(ckpt_path)

# #     with open("./config.json") as f:
# #         config = json.load(f)
# #         model_config = config[model_name]
# #         train_config = config["train_config"]

# #     batch_size = train_config["batch_size"]
# #     num_epochs = train_config["num_epochs"]
# #     train_ratio = train_config["train_ratio"]
# #     learning_rate = train_config["learning_rate"]
# #     optimizer = train_config["optimizer"]  # can be [sgd, adam]
# #     seq_len = train_config["seq_len"]

# #     if dataset_name == "ASSIST2009":
# #         dataset = ASSIST2009(seq_len)
# #     elif dataset_name == "ASSIST2018":
# #         dataset = ASSIST2018(seq_len)
# #     elif dataset_name == "ASSIST2015":
# #         dataset = ASSIST2015(seq_len)
# #     elif dataset_name == "Algebra2005":
# #         dataset = Algebra2005(seq_len)
# #     elif dataset_name == "Statics2011":
# #         dataset = Statics2011(seq_len)

# #     if torch.cuda.is_available():
# #         device = "cuda"
# #     else:
# #         device = "cpu"

# #     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
# #         json.dump(model_config, f, indent=4)
# #     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
# #         json.dump(train_config, f, indent=4)

# #     if model_name == "dkt":
# #         model = DKT(dataset.num_q, **model_config).to(device)
# #     elif model_name == "dkt+":
# #         model = DKTPlus(dataset.num_q, **model_config).to(device)
# #     elif model_name == "dkvmn":
# #         model = DKVMN(dataset.num_q, **model_config).to(device)
# #     elif model_name == "sakt":
# #         model = SAKT(dataset.num_q, **model_config).to(device)
# #     elif model_name == "gkt":
# #         if model_config["method"] == "PAM":
# #             model = PAM(dataset.num_q, **model_config).to(device)
# #         elif model_config["method"] == "MHA":
# #             model = MHA(dataset.num_q, **model_config).to(device)
# #     else:
# #         print("The wrong model name was used...")
# #         return

# #     train_size = int(len(dataset) * train_ratio)
# #     test_size = len(dataset) - train_size
# #     generator = torch.Generator(device=device)
# #     train_dataset, test_dataset = random_split(
# #         dataset, [train_size, test_size],generator=generator
# #     )

# #     if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
# #         with open(
# #             os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
# #         ) as f:
# #             train_dataset.indices = pickle.load(f)
# #         with open(
# #             os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
# #         ) as f:
# #             test_dataset.indices = pickle.load(f)
# #     else:
# #         with open(
# #             os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
# #         ) as f:
# #             pickle.dump(train_dataset.indices, f)
# #         with open(
# #             os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
# #         ) as f:
# #             pickle.dump(test_dataset.indices, f)

# #     train_loader = DataLoader(
# #         train_dataset, batch_size=batch_size, shuffle=True,
# #         collate_fn=collate_fn,
# #         generator=generator
# #     )
# #     test_loader = DataLoader(
# #         test_dataset, batch_size=test_size, shuffle=True,
# #         collate_fn=collate_fn,
# #         generator=generator
# #     )

# #     if optimizer == "sgd":
# #         opt = SGD(model.parameters(), learning_rate, momentum=0.9)
# #     elif optimizer == "adam":
# #         opt = Adam(model.parameters(), learning_rate)

# #     aucs, loss_means = \
# #         model.train_model(
# #             train_loader, test_loader, num_epochs, opt, ckpt_path
# #         )

# #     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
# #         pickle.dump(aucs, f)
# #     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
# #         pickle.dump(loss_means, f)


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument(
# #         "--model_name",
# #         type=str,
# #         default="gkt",
# #         help="The name of the model to train. \
# #             The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
# #             The default model is dkt."
# #     )
# #     parser.add_argument(
# #         "--dataset_name",
# #         type=str,
# #         default="Algebra2005",
# #         help="The name of the dataset to use in training. \
# #             The possible datasets are in \
# #             [ASSIST2009, ASSIST2015, Algebra2005, Statics2011, ASSIST2018]. \
# #             The default dataset is ASSIST2009."
# #     )
# #     args = parser.parse_args()
# #     print("trainig of gkt on algebra 2005")
# #     main(args.model_name, args.dataset_name)

# # import os
# # import argparse
# # import json
# # import pickle
# # import torch
# # import sys

# # from torch.utils.data import DataLoader, random_split
# # from torch.optim import SGD, Adam

# # # Add parent directory to sys.path
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # from data_loaders.assist2009 import ASSIST2009
# # from data_loaders.assist2018 import ASSIST2018
# # from data_loaders.assist2015 import ASSIST2015
# # from data_loaders.algebra2005 import Algebra2005
# # from data_loaders.statics2011 import Statics2011
# # from models.dkt import DKT
# # from models.dkt_plus import DKTPlus
# # from models.dkvmn import DKVMN
# # from models.sakt import SAKT
# # from models.gkt import PAM, MHA
# # from models.utils import collate_fn


# # def main(model_name, dataset_name):
# #     base_ckpt_dir = os.path.join( "ckpts")

# #     # Create folder structure
# #     os.makedirs(os.path.join(base_ckpt_dir, model_name, dataset_name), exist_ok=True)
# #     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name)

# #     # Load config
# #     with open("./config.json") as f:
# #         config = json.load(f)
# #         model_config = config[model_name]
# #         train_config = config["train_config"]

# #     batch_size = train_config["batch_size"]
# #     num_epochs = train_config["num_epochs"]
# #     train_ratio = train_config["train_ratio"]
# #     learning_rate = train_config["learning_rate"]
# #     optimizer_type = train_config["optimizer"]
# #     seq_len = train_config["seq_len"]

# #     # Load dataset
# #     if dataset_name == "ASSIST2009":
# #         dataset = ASSIST2009(seq_len)
# #     elif dataset_name == "ASSIST2018":
# #         dataset = ASSIST2018(seq_len)
# #     elif dataset_name == "ASSIST2015":
# #         dataset = ASSIST2015(seq_len)
# #     elif dataset_name == "Algebra2005":
# #         dataset = Algebra2005(seq_len)
# #     elif dataset_name == "Statics2011":
# #         dataset = Statics2011(seq_len)
# #     else:
# #         raise ValueError(f"Unknown dataset: {dataset_name}")

# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     # Save config copies
# #     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
# #         json.dump(model_config, f, indent=4)
# #     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
# #         json.dump(train_config, f, indent=4)

# #     # Load model
# #     if model_name == "dkt":
# #         model = DKT(dataset.num_q, **model_config).to(device)
# #     elif model_name == "dkt+":
# #         model = DKTPlus(dataset.num_q, **model_config).to(device)
# #     elif model_name == "dkvmn":
# #         model = DKVMN(dataset.num_q, **model_config).to(device)
# #     elif model_name == "sakt":
# #         model = SAKT(dataset.num_q, **model_config).to(device)
# #     elif model_name == "gkt":
# #         if model_config["method"] == "PAM":
# #             model = PAM(dataset.num_q, **model_config).to(device)
# #         elif model_config["method"] == "MHA":
# #             model = MHA(dataset.num_q, **model_config).to(device)
# #         else:
# #             raise ValueError(f"Invalid method for GKT: {model_config['method']}")
# #     else:
# #         raise ValueError(f"Unknown model: {model_name}")

# #     # Split dataset
# #     train_size = int(len(dataset) * train_ratio)
# #     test_size = len(dataset) - train_size
# #     generator = torch.Generator(device=device)
# #     train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# #     # Store/reuse split indices
# #     train_idx_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
# #     test_idx_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")
# #     if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
# #         with open(train_idx_path, "rb") as f:
# #             train_dataset.indices = pickle.load(f)
# #         with open(test_idx_path, "rb") as f:
# #             test_dataset.indices = pickle.load(f)
# #     else:
# #         with open(train_idx_path, "wb") as f:
# #             pickle.dump(train_dataset.indices, f)
# #         with open(test_idx_path, "wb") as f:
# #             pickle.dump(test_dataset.indices, f)

# #     # Data loaders
# #     train_loader = DataLoader(
# #         train_dataset, batch_size=batch_size, shuffle=True,
# #         collate_fn=collate_fn, generator=generator
# #     )
# #     test_loader = DataLoader(
# #         test_dataset, batch_size=test_size, shuffle=True,
# #         collate_fn=collate_fn, generator=generator
# #     )

# #     # Optimizer
# #     if optimizer_type == "sgd":
# #         opt = SGD(model.parameters(), learning_rate, momentum=0.9)
# #     elif optimizer_type == "adam":
# #         opt = Adam(model.parameters(), learning_rate)
# #     else:
# #         raise ValueError(f"Unknown optimizer: {optimizer_type}")

# #     # Train model and collect metrics
# #     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
# #         train_loader, test_loader, num_epochs, opt, ckpt_path
# #     )

# #     # Save individual metric files
# #     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
# #         pickle.dump(aucs, f)
# #     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
# #         pickle.dump(loss_means, f)

# #     # Save combined metrics
# #     metrics = {
# #         "train_loss": loss_means,  # if you have separate train/val, change here
# #         "valid_auc": aucs,
# #         "accuracy": accuracies,
# #         "f1": f1_scores,
# #         "mse": mses
# #     }
# #     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
# #         pickle.dump(metrics, f)

# #     print(f"✅ Training complete. Metrics saved in {ckpt_path}")


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--model_name", type=str, default="gkt",
# #                         help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
# #     parser.add_argument("--dataset_name", type=str, default="Algebra2005",
# #                         help="Dataset: [ASSIST2009, ASSIST2018, ASSIST2015, Algebra2005, Statics2011]")
# #     args = parser.parse_args()

# #     print(f"🚀 Training {args.model_name} on {args.dataset_name}")
# #     main(args.model_name, args.dataset_name)
# import os
# import sys
# import json
# import pickle
# import argparse
# import torch
# import itertools
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

# from models.dkt import DKT
# from models.dkt_plus import DKTPlus
# from models.dkvmn import DKVMN
# from models.sakt import SAKT
# from models.gkt import PAM, MHA

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

#     # Unique folder for run results based on hyperparameters
#     unique_folder = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_hs{model_config.get('hidden_size', 'NA')}"
#     base_ckpt_dir = "ckpts"
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, unique_folder)
#     os.makedirs(ckpt_path, exist_ok=True)

#     print(f"Training for {num_epochs} epochs")

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

#     # Save config copy
#     with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
#         json.dump(model_config, f, indent=4)
#     with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
#         json.dump(train_config, f, indent=4)

#     # Load model
#     if model_name == "dkt":
#         model = DKT(dataset.num_q, **model_config).to(device)
#     elif model_name == "dkt+":
#         model = DKTPlus(dataset.num_q, **model_config).to(device)
#     elif model_name == "dkvmn":
#         model = DKVMN(dataset.num_q, **model_config).to(device)
#     elif model_name == "sakt":
#         model = SAKT(dataset.num_q, **model_config).to(device)
#     elif model_name == "gkt":
#         method = model_config.get("method", None)
#         if method == "PAM":
#             model = PAM(dataset.num_q, **model_config).to(device)
#         elif method == "MHA":
#             model = MHA(dataset.num_q, **model_config).to(device)
#         else:
#             raise ValueError(f"Invalid method for GKT: {method}")
#     else:
#         raise ValueError(f"Unknown model: {model_name}")

#     # Dataset split
#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     generator = torch.Generator(device=device)
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

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

#     # Train the model
#     aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
#         train_loader, test_loader, num_epochs, opt, ckpt_path
#     )

#     # Save results
#     with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
#         pickle.dump(aucs, f)
#     with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
#         pickle.dump(loss_means, f)

#     metrics = {
#         "train_loss": loss_means,
#         "valid_auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses,
#     }
#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)

#     print(f"✅ Training complete. Metrics saved in {ckpt_path}")


# def grid_search(model_name, dataset_name, config_file):
#     with open(config_file) as f:
#         base_config = json.load(f)

#     param_grid = {
#         'train_config.num_epochs': [50],
#         'train_config.train_ratio': [0.8],
#         'train_config.learning_rate': [0.0005, 0.001, 0.005],
#         f'gkt.hidden_size': [64, 100, 128],
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

#         folder = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']['train_ratio']}_lr{config['train_config']['learning_rate']}_hs{config[model_name].get('hidden_size', 'NA')}"
#         metrics_path = os.path.join("ckpts", model_name, dataset_name, folder, "metrics.pkl")
#         if not os.path.exists(metrics_path):
#             print("⚠️ Metrics file missing! Skipping this combination.")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         auc = max(metrics.get('valid_auc', [0]))
#         print(f"AUC: {auc}")

#         if auc > best_auc:
#             best_auc = auc
#             best_params = params

#     print("\nGrid Search Complete!")
#     print(f"Best AUC: {best_auc}")
#     print(f"Best Hyperparameters: {best_params}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, default="train", help="train or grid_search")
#     parser.add_argument("--model_name", type=str, default="dkt", help="Model to train")
#     parser.add_argument("--dataset_name", type=str, default="Algebra2005", help="Dataset to use")
#     parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
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
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

# Fix import path to allow data_loaders to be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2018 import ASSIST2018
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from models.gkt import PAM, MHA
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
    optimizer_type = train_config["optimizer"]
    seq_len = train_config["seq_len"]
    # Unique folder name for checkpoint
    unique_folder_name = f"epochs{num_epochs}_tr{train_ratio}_lr{learning_rate}_hs{model_config.get('hidden_size','NA')}_heads{model_config.get('num_attn_heads','NA')}_method{model_config.get('method','NA')}"
    base_ckpt_dir = "ckpts"
    ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name, unique_folder_name)
    os.makedirs(ckpt_path, exist_ok=True)
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {dataset_name} for {num_epochs} epochs with method {model_config.get('method')} on {device}")
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
    # Save config
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    # Model selection
    method = model_config.get("method", None)
    if method == "PAM":
        model = PAM(dataset.num_q, **model_config).to(device)
    elif method == "MHA":
        model = MHA(dataset.num_q, **model_config).to(device)
    else:
        raise ValueError(f"Invalid method for GKT: {method}")
    # Reproducible manual split
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator(device="cpu").manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator,device="cpu").tolist()

    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    test_dataset = torch.utils.data.Subset(dataset, indices[train_size:])
    generator = torch.Generator(device=device).manual_seed(42)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
        generator=generator #
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn,
        generator=generator #
    )
    # Optimizer
    if optimizer_type == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer_type == "adam":
        opt = Adam(model.parameters(), learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    # Training
    aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
        train_loader, test_loader, num_epochs, opt, ckpt_path
    )
    # Save metrics
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)
    metrics = {
        "train_loss": loss_means,
        "valid_auc": aucs,
        "accuracy": accuracies,
        "f1": f1_scores,
        "mse": mses
    }
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print(f"✅ Training complete. Metrics saved in {ckpt_path}")

def grid_search(model_name, dataset_name, base_config_file="./config.json"):
    with open(base_config_file) as f:
        base_config = json.load(f)
    # Define grid (tune as needed)
    param_grid = {
        'train_config.num_epochs': [50],
        'train_config.train_ratio': [0.8],
        'train_config.learning_rate': [0.0005, 0.001],
        f'{model_name}.hidden_size': [64],
        f'{model_name}.num_attn_heads': [4],
        f'{model_name}.method': ["PAM", "MHA"]
    }
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_auc = -float('inf')
    best_params = None
    for params in all_combinations:
        config = json.loads(json.dumps(base_config))  # deep copy
        for key, value in params.items():
            section, param = key.split('.')
            config[section][param] = value
        temp_config_path = "./temp_config.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"\nRunning grid search with params: {params}")
        main(model_name, dataset_name, temp_config_path)
        folder = f"epochs{config['train_config']['num_epochs']}_tr{config['train_config']['train_ratio']}_lr{config['train_config']['learning_rate']}_hs{config[model_name].get('hidden_size','NA')}_heads{config[model_name].get('num_attn_heads','NA')}_method{config[model_name].get('method','NA')}"
        run_dir = os.path.join("ckpts", model_name, dataset_name, folder)
        metrics_path = os.path.join(run_dir, "metrics.pkl")
        if not os.path.exists(metrics_path):
            print("⚠️ Metrics file missing! Skipping this combination.")
            continue
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
        auc = max(metrics.get('valid_auc', [0]))
        print(f"AUC: {auc}")
        if auc > best_auc:
            best_auc = auc
            best_params = params
            
    print("\nGrid Search Complete!")
    print(f"Best AUC: {best_auc}")
    print(f"Best Hyperparameters: {best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or grid_search")
    parser.add_argument("--model_name", type=str, default="gkt", help="Model to train")
    parser.add_argument("--dataset_name", type=str, default="Algebra2005", help="Dataset to use")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
    args = parser.parse_args()
    if args.mode == "grid_search":
        grid_search(args.model_name, args.dataset_name, args.config)
    else:
        main(args.model_name, args.dataset_name, args.config)

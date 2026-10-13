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
#         default="ASSIST2015",
#         help="The name of the dataset to use in training. \
#             The possible datasets are in \
#             [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
#             The default dataset is ASSIST2009."
#     )
#     args = parser.parse_args()
#     print("trainig of SAKT on assist 2015")
#     main(args.model_name, args.dataset_name)

# import os
# import argparse
# import json
# import pickle
# import torch
# import sys

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


# def main(model_name, dataset_name):
#     base_ckpt_dir = os.path.join( "ckpts")

#     # Create folder structure
#     os.makedirs(os.path.join(base_ckpt_dir, model_name, dataset_name), exist_ok=True)
#     ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name)

#     # Load config
#     with open("./config.json") as f:
#         config = json.load(f)
#         model_config = config[model_name]
#         train_config = config["train_config"]

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
#         if model_config["method"] == "PAM":
#             model = PAM(dataset.num_q, **model_config).to(device)
#         elif model_config["method"] == "MHA":
#             model = MHA(dataset.num_q, **model_config).to(device)
#         else:
#             raise ValueError(f"Invalid method for GKT: {model_config['method']}")
#     else:
#         raise ValueError(f"Unknown model: {model_name}")

#     # Split dataset
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
#         test_dataset, batch_size=test_size, shuffle=True,
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
#         "train_loss": loss_means,  # if you have separate train/val, change here
#         "valid_auc": aucs,
#         "accuracy": accuracies,
#         "f1": f1_scores,
#         "mse": mses
#     }
#     with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
#         pickle.dump(metrics, f)

#     print(f"✅ Training complete. Metrics saved in {ckpt_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="sakt",
#                         help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
#     parser.add_argument("--dataset_name", type=str, default="ASSIST2015",
#                         help="Dataset: [ASSIST2009, ASSIST2018, ASSIST2015, Algebra2005, Statics2011]")
#     args = parser.parse_args()

#     print(f"🚀 Training {args.model_name} on {args.dataset_name}")
#     main(args.model_name, args.dataset_name)
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


def train_single_run(model_name, dataset_name, config):
    base_ckpt_dir = os.path.join("ckpts")
    os.makedirs(os.path.join(base_ckpt_dir, model_name, dataset_name), exist_ok=True)
    ckpt_path = os.path.join(base_ckpt_dir, model_name, dataset_name)

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

    # Load model
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

    # Split dataset
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False,
                             collate_fn=collate_fn, generator=generator)

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
        "mse": mses
    }
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    return aucs, metrics, ckpt_path


def grid_search(model_name, dataset_name, base_config_path="./config.json"):
    with open(base_config_path) as f:
        base_config = json.load(f)

    param_grid = {
        "train_config.num_epochs": [50],
        "train_config.train_ratio": [0.8],
        "train_config.learning_rate": [0.0005, 0.001, 0.005],
        "train_config.optimizer": ["adam"],
        f"{model_name}.dropout": [0.1, 0.2],
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_auc = -float("inf")
    best_params = None

    for idx, params in enumerate(all_combinations):
        config = copy.deepcopy(base_config)

        # Update config with current params
        for key, value in params.items():
            section, param = key.split(".", 1)
            if section in config:
                config[section][param] = value
            else:
                config[section] = {param: value}

        print(f"\n🔎 Grid search run {idx+1}/{len(all_combinations)} | Params: {params}")
        aucs, metrics, ckpt_path = train_single_run(model_name, dataset_name, config)

        max_auc = max(aucs) if aucs else 0
        print(f"AUC: {max_auc}")

        if max_auc > best_auc:
            best_auc = max_auc
            best_params = params

    print("\n🎯 Grid Search Complete!")
    print(f"Best AUC: {best_auc}")
    print(f"Best Hyperparameters: {best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sakt",
                        help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
    parser.add_argument("--dataset_name", type=str, default="ASSIST2015",
                        help="Dataset: [ASSIST2009, ASSIST2018, ASSIST2015, Algebra2005, Statics2011]")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
    parser.add_argument("--config", type=str, default="./config.json",
                        help="Path to config file")

    args = parser.parse_args()

    print(f"🚀 {args.mode.capitalize()} {args.model_name} on {args.dataset_name}")
    if args.mode == "grid_search":
        grid_search(args.model_name, args.dataset_name, args.config)
    else:
        train_single_run(args.model_name, args.dataset_name, json.load(open(args.config)))
        print("✅ Training complete.")

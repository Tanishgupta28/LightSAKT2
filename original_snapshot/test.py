
import os
import sys
import json
import pickle
import argparse
import itertools
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loaders.assist2015 import ASSIST2015
from models.light_sakt import Light_SAKT
from models.utils import collate_fn


def main(model_name, dataset_name, config_file):
    """Train a single Light_SAKT model run using 70% training and 10% validation (no test set)."""
    with open(config_file) as f:
        config = json.load(f)

    model_config = config[model_name]
    train_config = config["train_config"]

    # Load training hyperparameters
    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]  # should be 0.7 from config.json
    learning_rate = train_config["learning_rate"]
    seq_len = train_config["seq_len"]

    # Set up checkpoint directory
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

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    if dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # ---- MAIN SPLIT: 70% training, 10% validation ----
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)  # 70%
    val_size = int(total_size * 0.1)            # 10%
    unused_size = total_size - (train_size + val_size)
    generator = torch.Generator(device=device).manual_seed(42)

    # Only keep 70% train and 10% val (ignore unused portion)
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, unused_size], generator=generator
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,generator=generator)

    # ---- Initialize model ----
    model = Light_SAKT(dataset.num_q, **model_config).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---- Train model ----
    aucs, loss_means, accuracies, f1_scores, mses = model.train_model(
        train_loader, val_loader, num_epochs, optimizer, ckpt_path
    )

    # ---- Save metrics ----
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump(
            {
                "train_loss": loss_means,
                "valid_auc": aucs,
                "accuracy": accuracies,
                "f1": f1_scores,
                "mse": mses,
            },
            f,
        )

    print(f"✅ Training complete. Metrics saved in {ckpt_path}")


def grid_search(model_name, dataset_name, config_file):
    """Perform grid search over hyperparameters for Light_SAKT using 70% training and 10% validation."""
    with open(config_file) as f:
        base_config = json.load(f)

    # Hyperparameter grid
    param_grid = {
        "train_config.learning_rate": [0.0005, 0.001, 0.005],
        "light_sakt.embed_dim": [32, 64, 128],
        "light_sakt.num_heads": [2, 4],
        "light_sakt.dropout": [0.1, 0.2],
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    best_auc = -float("inf")
    best_params = None
    best_run_dir = None

    for params in all_combinations:
        # Deep copy base config
        config = json.loads(json.dumps(base_config))

        # Update config with current parameter combination
        for key, value in params.items():
            section, param = key.split(".")
            config[section][param] = value

        # Ensure train ratio fixed at 0.7
        config["train_config"]["train_ratio"] = 0.7

        # Save temporary config
        temp_config_path = "./temp_config.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"\n🚀 Running grid search with params: {params}")
        main(model_name, dataset_name, temp_config_path)

        # Folder name reconstruction for metrics lookup
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
            print("⚠ Metrics missing, skipping.")
            continue

        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

        auc = max(metrics.get("valid_auc", [0]))
        print(f"📈 Validation AUC: {auc:.4f}")

        # Track best run
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_run_dir = run_dir

    # ---- Final summary ----
    print("\n✅ Grid Search complete!")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"📂 Best run saved in: {best_run_dir}")

    with open("best_run.txt", "w") as f:
        f.write(best_run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or grid_search")
    parser.add_argument("--model_name", type=str, default="light_sakt", help="Model to train")
    parser.add_argument("--dataset_name", type=str, default="ASSIST2015", help="Dataset to use")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to config file")
    args = parser.parse_args()

    if args.mode == "grid_search":
        grid_search(args.model_name, args.dataset_name, args.config)
    else:
        main(args.model_name,args.dataset_name,args.config)
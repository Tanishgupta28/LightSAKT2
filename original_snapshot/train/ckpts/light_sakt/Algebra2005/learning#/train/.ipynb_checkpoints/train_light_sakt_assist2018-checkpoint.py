import os
import argparse
import json
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2018 import ASSIST2018
from models.light_sakt import LightSAKT
from models.utils import collate_fn
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(metrics, save_path):
    """Plot and save training curves for all metrics"""
    plt.figure(figsize=(15, 10))

    metric_names = ["auc", "accuracy", "f1", "mse", "loss"]
    titles = {
        "auc": "AUC over Epochs",
        "accuracy": "Accuracy over Epochs",
        "f1": "F1 Score over Epochs",
        "mse": "MSE over Epochs",
        "loss": "Loss over Epochs"
    }

    for i, metric in enumerate(metric_names):
        plt.subplot(3, 2, i + 1)
        plt.plot(metrics[metric], label=metric.upper())
        plt.title(titles[metric])
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

def main(dataset_name, config_path="./config.json"):
    # Create checkpoint directory
    ckpt_path = os.path.join("ckpts", "light_sakt", dataset_name)
    os.makedirs(ckpt_path, exist_ok=True)

    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
        model_config = config["sakt"]
        train_config = config["train_config"]

    model_config["dropout"] = 0.1  # Adjust dropout

    # Training parameters
    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    seq_len = train_config["seq_len"]

    # Load dataset
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "ASSIST2018":
        dataset = ASSIST2018(seq_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Save config files
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    # Initialize model
    model = LightSAKT(dataset.num_q, **model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Train/test split
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator(device=device)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    # Save split indices
    indices_path = os.path.join(dataset.dataset_dir, "light_sakt_indices")
    os.makedirs(indices_path, exist_ok=True)
    with open(os.path.join(indices_path, "train_indices.pkl"), "wb") as f:
        pickle.dump(train_dataset.indices, f)
    with open(os.path.join(indices_path, "test_indices.pkl"), "wb") as f:
        pickle.dump(test_dataset.indices, f)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, collate_fn=collate_fn,generator=generator)

    # Optimizer
    optimizer = Adam(model.parameters(), learning_rate)

    # Train
    print(f"Starting training on {dataset_name}...")
    metrics = model.train_model(train_loader, test_loader, num_epochs, optimizer, ckpt_path)

    # Save metrics
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    # Plot metrics
    plot_training_curves(metrics, ckpt_path)

    # Print final results
    print("\nTraining completed!")
    print(f"Best AUC:      {max(metrics['auc']):.4f}")
    print(f"Final Accuracy:{metrics['accuracy'][-1]:.4f}")
    print(f"Final F1:      {metrics['f1'][-1]:.4f}")
    print(f"Final MSE:     {metrics['mse'][-1]:.4f}")
    print(f"Final Loss:    {metrics['loss'][-1]:.4f}")
    print(f"Results saved to: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lightweight SAKT model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ASSIST2018",
        choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "ASSIST2018"],
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.json",
        help="Path to configuration file"
    )

    args = parser.parse_args()
    main(args.dataset, args.config)

"""
Kaggle-Compatible Training Script for Knowledge Tracing Models
Works seamlessly on both local and Kaggle environments
"""
import os
import argparse
import json
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import sys
import matplotlib.pyplot as plt
import numpy as np

# Setup paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'train')))

# Import utilities
from kaggle_utils import (
    detect_environment, 
    get_dataset_path, 
    print_environment_info,
    setup_kaggle_paths
)

# Setup Kaggle paths
setup_kaggle_paths()

# Import data loaders (try Kaggle version first, fallback to original)
try:
    from data_loaders.assist2009_kaggle import ASSIST2009
except ImportError:
    from data_loaders.assist2009 import ASSIST2009

try:
    from data_loaders.assist2015 import ASSIST2015
except ImportError:
    ASSIST2015 = None

try:
    from data_loaders.algebra2005 import Algebra2005
except ImportError:
    Algebra2005 = None

try:
    from data_loaders.assist2018 import ASSIST2018
except ImportError:
    ASSIST2018 = None

# Import models
from models.light_sakt import LightSAKT
from models.sakt import SAKT
from models.dkt import DKT
from models.utils import collate_fn


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
        if metric in metrics:
            plt.plot(metrics[metric], label=metric.upper())
            plt.title(titles[metric])
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.grid(True)
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {os.path.join(save_path, 'training_curves.png')}")


def load_dataset(dataset_name, seq_len, env_config):
    """Load dataset based on name"""
    print(f"\nLoading dataset: {dataset_name}")
    
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len, env_config=env_config)
    elif dataset_name == "ASSIST2015":
        if ASSIST2015 is None:
            raise ImportError("ASSIST2015 loader not available")
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        if Algebra2005 is None:
            raise ImportError("Algebra2005 loader not available")
        dataset = Algebra2005(seq_len)
    elif dataset_name == "ASSIST2018":
        if ASSIST2018 is None:
            raise ImportError("ASSIST2018 loader not available")
        dataset = ASSIST2018(seq_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def load_model(model_name, num_q, model_config, device):
    """Load model based on name"""
    print(f"\nInitializing model: {model_name}")
    
    if model_name == "light_sakt":
        model = LightSAKT(num_q, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(num_q, **model_config).to(device)
    elif model_name == "dkt":
        model = DKT(num_q, **model_config).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model


def main(args):
    """Main training function"""
    
    # Print environment information
    print_environment_info()
    
    # Detect environment
    env_config = detect_environment()
    
    # Create checkpoint directory
    ckpt_path = os.path.join(env_config['ckpt_dir'], args.model, args.dataset)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"\nCheckpoint directory: {ckpt_path}")
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        # Try alternative paths
        alt_paths = [
            'train/config.json',
            '/kaggle/working/train/config.json',
            '/kaggle/input/kt-code/train/config.json'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    
    # Get model and training configs
    if args.model in config:
        model_config = config[args.model]
    else:
        model_config = config.get("light_sakt", {})  # Default
    
    train_config = config["train_config"]
    
    # Override with command line arguments if provided
    if args.batch_size:
        train_config["batch_size"] = args.batch_size
    if args.epochs:
        train_config["num_epochs"] = args.epochs
    if args.learning_rate:
        train_config["learning_rate"] = args.learning_rate
    
    # Training parameters
    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    seq_len = train_config["seq_len"]
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Sequence length: {seq_len}")
    
    # Load dataset
    dataset = load_dataset(args.dataset, seq_len, env_config)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Save config files
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    
    # Initialize model
    model = load_model(args.model, dataset.num_q, model_config, device)
    
    # Train/test split
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Test samples: {test_size}")
    
    # Save split indices
    indices_dir = os.path.join(env_config['cache_dir'], args.dataset, f"{args.model}_indices")
    os.makedirs(indices_dir, exist_ok=True)
    
    with open(os.path.join(indices_dir, "train_indices.pkl"), "wb") as f:
        pickle.dump(train_dataset.indices, f)
    with open(os.path.join(indices_dir, "test_indices.pkl"), "wb") as f:
        pickle.dump(test_dataset.indices, f)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = Adam(model.parameters(), learning_rate)
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training: {args.model} on {args.dataset}")
    print(f"{'='*60}\n")
    
    metrics = model.train_model(train_loader, test_loader, num_epochs, optimizer, ckpt_path)
    
    # Save metrics
    with open(os.path.join(ckpt_path, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    
    # Plot metrics
    plot_training_curves(metrics, ckpt_path)
    
    # Print final results
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Best AUC:       {max(metrics['auc']):.4f}")
    print(f"Final Accuracy: {metrics['accuracy'][-1]:.4f}")
    print(f"Final F1:       {metrics['f1'][-1]:.4f}")
    print(f"Final MSE:      {metrics['mse'][-1]:.4f}")
    print(f"Final Loss:     {metrics['loss'][-1]:.4f}")
    print(f"\nResults saved to: {ckpt_path}")
    print(f"{'='*60}\n")
    
    # Save summary
    summary = {
        'model': args.model,
        'dataset': args.dataset,
        'best_auc': float(max(metrics['auc'])),
        'final_accuracy': float(metrics['accuracy'][-1]),
        'final_f1': float(metrics['f1'][-1]),
        'final_mse': float(metrics['mse'][-1]),
        'final_loss': float(metrics['loss'][-1]),
        'num_epochs': num_epochs,
        'num_params': sum(p.numel() for p in model.parameters()),
    }
    
    with open(os.path.join(ckpt_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Knowledge Tracing models on Kaggle or locally")
    
    parser.add_argument(
        "--model",
        type=str,
        default="light_sakt",
        choices=["light_sakt", "sakt", "dkt", "dkt+", "dkvmn", "gkt"],
        help="Model to train"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ASSIST2009",
        choices=["ASSIST2009", "ASSIST2015", "Algebra2005", "ASSIST2018"],
        help="Dataset to use for training"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="train/config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )

    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise

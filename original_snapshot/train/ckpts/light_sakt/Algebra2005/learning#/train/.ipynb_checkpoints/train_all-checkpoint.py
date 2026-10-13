import os
import argparse
import json
from itertools import product
import torch
from torch.utils.data import DataLoader

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA
from models.utils import collate_fn
from models.visualization import plot_dataset_comparison, create_summary_table

DATASETS = {
    "ASSIST2009": ASSIST2009,
    "ASSIST2015": ASSIST2015,
    "Algebra2005": Algebra2005
}

MODELS = {
    "dkt": DKT,
    "dkt+": DKTPlus,
    "dkvmn": DKVMN,
    "sakt": SAKT,
    "gkt_pam": lambda num_q, **kwargs: PAM(num_q, **{**kwargs, "method": "PAM"}),
    "gkt_mha": lambda num_q, **kwargs: MHA(num_q, **{**kwargs, "method": "MHA"})
}

def train_model_on_dataset(model_name, dataset_name, config):
    """Train a specific model on a specific dataset."""
    print(f"\nTraining {model_name} on {dataset_name}")
    
    # Create checkpoint directory
    ckpt_path = os.path.join("ckpts", model_name, dataset_name)
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Load configurations
    model_config = config[model_name.split('_')[0]]  # Handle gkt variants
    train_config = config["train_config"]
    dataset_config = config["dataset_config"][dataset_name]
    
    # Initialize dataset
    dataset = DATASETS[dataset_name](dataset_config["seq_len"])
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODELS[model_name](dataset.num_q, **model_config).to(device)
    
    # Split dataset and create data loaders
    train_size = int(len(dataset) * train_config["train_ratio"])
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), train_config["learning_rate"])
    
    # Train model
    aucs, loss_means = model.train_model(
        train_loader,
        test_loader,
        train_config["num_epochs"],
        optimizer,
        ckpt_path
    )
    
    return aucs, loss_means

def test_critical_path():
    """Run critical path testing with DKT model on ASSIST2009 dataset"""
    print("Running critical path test...")
    
    with open("config.json") as f:
        config = json.load(f)
    
    try:
        aucs, loss_means = train_model_on_dataset("dkt", "ASSIST2009", config)
        print("\nCritical path test results:")
        print(f"Final AUC: {aucs[-1]:.4f}")
        print(f"Final Loss: {loss_means[-1]:.4f}")
        
        # Generate test plot
        os.makedirs("results", exist_ok=True)
        plot_dataset_comparison(
            "ckpts",
            ["ASSIST2009"],
            ["dkt"],
            save_dir="results"
        )
        print("\nTest plot saved to results/ASSIST2009_comparison.png")
        return True
    except Exception as e:
        print(f"\nCritical path test failed: {str(e)}")
        return False

def main(args):
    # Load configuration
    with open("config.json") as f:
        config = json.load(f)
    
    if args.test:
        success = test_critical_path()
        if not success:
            return
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Get list of models and datasets to train
    models_to_train = args.models.split(',') if args.models else MODELS.keys()
    datasets_to_train = args.datasets.split(',') if args.datasets else DATASETS.keys()
    
    # Train all specified combinations
    for model_name, dataset_name in product(models_to_train, datasets_to_train):
        try:
            train_model_on_dataset(model_name, dataset_name, config)
        except Exception as e:
            print(f"Error training {model_name} on {dataset_name}: {str(e)}")
    
    # Generate comparison plots
    plot_dataset_comparison(
        "ckpts",
        datasets_to_train,
        models_to_train,
        save_dir="results"
    )
    
    # Create and save summary table
    summary = create_summary_table("ckpts", datasets_to_train, models_to_train)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to train. Default: all models"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to train on. Default: all datasets"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run critical path test (DKT on ASSIST2009)"
    )
    args = parser.parse_args()
    
    main(args)

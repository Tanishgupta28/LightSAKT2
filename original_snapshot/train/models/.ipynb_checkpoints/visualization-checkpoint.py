import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(models_data, dataset_name, save_path=None):
    """
    Plot learning curves (AUC vs Epoch) for multiple models.
    
    Args:
        models_data (dict): Dictionary containing model names and their AUC scores
        dataset_name (str): Name of the dataset for the plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define colors and line styles matching the example graphs
    model_styles = {
        'dkt': ('blue', '-'),
        'dkt+': ('orange', '-'),
        'dkvmn': ('green', '-'),
        'kon': ('red', '-'),
        'sakt': ('purple', '-'),
        'gkt_pam': ('brown', '--'),
        'gkt_mha': ('pink', '-')
    }
    
    # Set up the plot style to match the example
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.50, 0.85)  # Set y-axis limits like in example
    plt.xlim(0, None)     # Start x-axis from 0
    
    # Plot each model's learning curve
    for model_name, aucs in models_data.items():
        color, linestyle = model_styles.get(model_name.lower(), ('gray', '-'))
        epochs = range(len(aucs))
        plt.plot(epochs, aucs, label=model_name.upper(), 
                color=color, linestyle=linestyle, linewidth=1.5)
    
    # Customize the plot
    plt.title(dataset_name, pad=20, fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Test AUC', fontsize=10)
    
    # Customize grid
    plt.grid(True, which='both', linestyle='--', alpha=0.2)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def load_training_metrics(ckpt_dir, model_name, dataset_name):
    """
    Load training metrics from checkpoint directory.
    
    Args:
        ckpt_dir (str): Base checkpoint directory
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
    
    Returns:
        tuple: (aucs, loss_means)
    """
    model_path = os.path.join(ckpt_dir, model_name, dataset_name)
    
    with open(os.path.join(model_path, "aucs.pkl"), "rb") as f:
        aucs = pickle.load(f)
    with open(os.path.join(model_path, "loss_means.pkl"), "rb") as f:
        loss_means = pickle.load(f)
    
    return aucs, loss_means

def plot_dataset_comparison(ckpt_dir, dataset_names, model_names, save_dir=None):
    """
    Create comparison plots for all models on each dataset.
    
    Args:
        ckpt_dir (str): Base checkpoint directory
        dataset_names (list): List of dataset names
        model_names (list): List of model names
        save_dir (str, optional): Directory to save plots
    """
    # Create a figure with subplots for each dataset
    if len(dataset_names) > 1:
        fig = plt.figure(figsize=(20, 10))
        rows = (len(dataset_names) + 1) // 2
        cols = min(len(dataset_names), 2)
    
    for idx, dataset in enumerate(dataset_names, 1):
        if len(dataset_names) > 1:
            plt.subplot(rows, cols, idx)
        
        models_data = {}
        for model in model_names:
            try:
                aucs, _ = load_training_metrics(ckpt_dir, model, dataset)
                models_data[model] = aucs
            except FileNotFoundError:
                print(f"No data found for {model} on {dataset}")
                continue
        
        if models_data:
            save_path = os.path.join(save_dir, f"{dataset}_comparison.png") if save_dir else None
            plot_learning_curves(models_data, dataset, save_path)
    
    if len(dataset_names) > 1:
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "all_datasets_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def create_summary_table(ckpt_dir, dataset_names, model_names):
    """
    Create a summary table of final AUC scores for all models on all datasets.
    
    Args:
        ckpt_dir (str): Base checkpoint directory
        dataset_names (list): List of dataset names
        model_names (list): List of model names
    
    Returns:
        dict: Nested dictionary containing final AUC scores
    """
    summary = {}
    for dataset in dataset_names:
        summary[dataset] = {}
        for model in model_names:
            try:
                aucs, _ = load_training_metrics(ckpt_dir, model, dataset)
                summary[dataset][model] = aucs[-1]  # Get final AUC score
            except FileNotFoundError:
                summary[dataset][model] = None
    
    return summary

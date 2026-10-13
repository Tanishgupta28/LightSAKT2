import os
import pickle
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def load_results(model_path):
    """Load training results from checkpoint directory"""
    try:
        with open(os.path.join(model_path, "aucs.pkl"), "rb") as f:
            aucs = pickle.load(f)
        with open(os.path.join(model_path, "losses.pkl"), "rb") as f:
            losses = pickle.load(f)
        return aucs, losses
    except FileNotFoundError:
        return None, None

def plot_single_model_results(dataset_name, save_dir="results"):
    """Plot results for Light SAKT on a single dataset"""
    model_path = os.path.join("ckpts", "light_sakt", dataset_name)
    aucs, losses = load_results(model_path)
    
    if aucs is None:
        print(f"No results found for Light SAKT on {dataset_name}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Plot AUC curve
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(aucs) + 1), aucs, 'b-', linewidth=2, label='Light SAKT')
    plt.title(f'AUC Progression - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.5, 1.0)
    
    # Plot Loss curve
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(losses) + 1), losses, 'r-', linewidth=2, label='Light SAKT')
    plt.title(f'Loss Progression - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot performance summary
    plt.subplot(1, 3, 3)
    max_auc = max(aucs)
    final_auc = aucs[-1]
    min_loss = min(losses)
    final_loss = losses[-1]
    
    metrics = ['Max AUC', 'Final AUC', 'Min Loss', 'Final Loss']
    values = [max_auc, final_auc, min_loss, final_loss]
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title(f'Performance Summary - {dataset_name}')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'light_sakt_{dataset_name.lower()}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved for {dataset_name}")

def compare_with_original_sakt(datasets, save_dir="results"):
    """Compare Light SAKT with original SAKT"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, len(datasets), figsize=(5*len(datasets), 10))
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, dataset in enumerate(datasets):
        # Load Light SAKT results
        light_path = os.path.join("ckpts", "light_sakt", dataset)
        light_aucs, light_losses = load_results(light_path)
        
        # Load original SAKT results
        sakt_path = os.path.join("ckpts", "sakt", dataset)
        sakt_aucs, sakt_losses = load_results(sakt_path)
        
        # Plot AUC comparison
        ax1 = axes[0, i]
        if light_aucs:
            ax1.plot(range(1, len(light_aucs) + 1), light_aucs, 
                    'b-', linewidth=2, label='Light SAKT')
        if sakt_aucs:
            ax1.plot(range(1, len(sakt_aucs) + 1), sakt_aucs, 
                    'g--', linewidth=2, label='Original SAKT')
        
        ax1.set_title(f'AUC Comparison - {dataset}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('AUC')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0.5, 1.0)
        
        # Plot Loss comparison
        ax2 = axes[1, i]
        if light_losses:
            ax2.plot(range(1, len(light_losses) + 1), light_losses, 
                    'r-', linewidth=2, label='Light SAKT')
        if sakt_losses:
            ax2.plot(range(1, len(sakt_losses) + 1), sakt_losses, 
                    'orange', linestyle='--', linewidth=2, label='Original SAKT')
        
        ax2.set_title(f'Loss Comparison - {dataset}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'light_vs_original_sakt.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plot saved")

def create_performance_table(datasets, save_dir="results"):
    """Create a performance comparison table"""
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        
        # Light SAKT results
        light_path = os.path.join("ckpts", "light_sakt", dataset)
        light_aucs, light_losses = load_results(light_path)
        
        if light_aucs:
            results[dataset]['Light SAKT'] = {
                'Max AUC': max(light_aucs),
                'Final AUC': light_aucs[-1],
                'Min Loss': min(light_losses),
                'Final Loss': light_losses[-1]
            }
        
        # Original SAKT results
        sakt_path = os.path.join("ckpts", "sakt", dataset)
        sakt_aucs, sakt_losses = load_results(sakt_path)
        
        if sakt_aucs:
            results[dataset]['Original SAKT'] = {
                'Max AUC': max(sakt_aucs),
                'Final AUC': sakt_aucs[-1],
                'Min Loss': min(sakt_losses),
                'Final Loss': sakt_losses[-1]
            }
    
    # Save results to JSON
    os.makedirs(save_dir, exist_ok=True)

    # Convert numpy.float32 to float for JSON serialization
    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError

    with open(os.path.join(save_dir, "light_sakt_comparison.json"), "w") as f:
        json.dump(results, f, indent=4, default=convert)
    
    # Print table
    print("\nPerformance Comparison Table:")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        print("-" * 40)
        
        if 'Light SAKT' in results[dataset]:
            light_data = results[dataset]['Light SAKT']
            print(f"Light SAKT    - Max AUC: {light_data['Max AUC']:.4f}, "
                  f"Final AUC: {light_data['Final AUC']:.4f}")
        
        if 'Original SAKT' in results[dataset]:
            sakt_data = results[dataset]['Original SAKT']
            print(f"Original SAKT - Max AUC: {sakt_data['Max AUC']:.4f}, "
                  f"Final AUC: {sakt_data['Final AUC']:.4f}")
    
    return results

import os
import pickle
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def load_results(model_path):
    """Load training results from checkpoint directory"""
    try:
        with open(os.path.join(model_path, "aucs.pkl"), "rb") as f:
            aucs = pickle.load(f)
        with open(os.path.join(model_path, "losses.pkl"), "rb") as f:
            losses = pickle.load(f)
        return aucs, losses
    except FileNotFoundError:
        return None, None

def plot_single_model_results(dataset_name, save_dir="results"):
    """Plot results for Light SAKT on a single dataset"""
    model_path = os.path.join("ckpts", "light_sakt", dataset_name)
    aucs, losses = load_results(model_path)
    
    if aucs is None:
        print(f"No results found for Light SAKT on {dataset_name}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Plot AUC curve
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(aucs) + 1), aucs, 'b-', linewidth=2, label='Light SAKT')
    plt.title(f'AUC Progression - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.5, 1.0)
    
    # Plot Loss curve
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(losses) + 1), losses, 'r-', linewidth=2, label='Light SAKT')
    plt.title(f'Loss Progression - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot performance summary
    plt.subplot(1, 3, 3)
    max_auc = max(aucs)
    final_auc = aucs[-1]
    min_loss = min(losses)
    final_loss = losses[-1]
    
    metrics = ['Max AUC', 'Final AUC', 'Min Loss', 'Final Loss']
    values = [max_auc, final_auc, min_loss, final_loss]
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title(f'Performance Summary - {dataset_name}')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'light_sakt_{dataset_name.lower()}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved for {dataset_name}")

def plot_combined_light_sakt_results(datasets, save_dir="results"):
    """Plot combined AUC and Loss curves for Light SAKT on multiple datasets"""
    plt.figure(figsize=(12, 6))
    
    for dataset in datasets:
        model_path = os.path.join("ckpts", "light_sakt", dataset)
        aucs, losses = load_results(model_path)
        if aucs is None:
            print(f"No results found for Light SAKT on {dataset}")
            continue
        
        epochs = range(1, len(aucs) + 1)
        plt.plot(epochs, aucs, label=f'AUC - {dataset}')
        plt.plot(epochs, losses, label=f'Loss - {dataset}', linestyle='--')
    
    plt.title("Combined Light SAKT Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "light_sakt_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to {os.path.join(save_dir, 'light_sakt_combined.png')}")

def main():
    parser = argparse.ArgumentParser(description="Plot Light SAKT results")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ASSIST2009,ASSIST2015,Algebra2005",
        help="Comma-separated list of datasets"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with original SAKT"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    
    print("Generating plots for Light SAKT...")
    
    # Plot individual results
    for dataset in datasets:
        plot_single_model_results(dataset, args.save_dir)
    
    # Plot combined results
    plot_combined_light_sakt_results(datasets, args.save_dir)
    
    # Compare with original SAKT if requested
    if args.compare:
        print("Comparing with original SAKT...")
        compare_with_original_sakt(datasets, args.save_dir)
    
    # Create performance table
    create_performance_table(datasets, args.save_dir)
    
    print(f"\nAll plots saved to '{args.save_dir}' directory")

if __name__ == "__main__":
    main()

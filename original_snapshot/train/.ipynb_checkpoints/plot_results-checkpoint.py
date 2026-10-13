# import os
# import sys
# import json
# import argparse
# from models.visualization import plot_dataset_comparison, create_summary_table

# def main(args):
#     # Load configuration
#     config_path = "./config.json"
#     with open(config_path) as f:
#         config = json.load(f)

#     os.makedirs("results", exist_ok=True)

#     models = args.models.split(',') if args.models else [
#         "dkt", "dkt+", "dkvmn", "sakt", "gkt_pam", "gkt_mha","light_sakt"
#     ]
#     datasets = args.datasets.split(',') if args.datasets else [
#         "ASSIST2009", "ASSIST2015", "Algebra2005"
#     ]

#     plot_dataset_comparison("ckpts", models, datasets, save_dir="results")
#     summary = create_summary_table("ckpts", models, datasets)

#     print("\nFinal AUC Scores:")
#     print("-" * 80)
#     header = "Dataset".ljust(15) + " | " + " | ".join(m.ljust(10) for m in models)
#     print(header)
#     print("-" * len(header))
    
#     for dataset in datasets:
#         row = dataset.ljust(15) + " | "
#         row += " | ".join(
#             f"{summary.get(dataset, {}).get(model, 'N/A'):.4f}".ljust(10)
#             if summary.get(dataset, {}).get(model) is not None
#             else "N/A".ljust(10)
#             for model in models
#         )
#         print(row)

#     with open("results/summary.json", "w") as f:
#         json.dump(summary, f, indent=4)

#     print(f"\nPlots and summary have been saved to the 'results' directory.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate comparison plots for trained models")
#     parser.add_argument("--models", type=str)
#     parser.add_argument("--datasets", type=str)
#     args = parser.parse_args(args=[])  # <-- for testing
#     main(args) 



# import os
# import pickle
# import matplotlib.pyplot as plt
# import pandas as pd

# # Define datasets and models
# datasets = ["Algebra2005", "ASSIST2009","ASSIST2015"]
# models = ["dkt", "light_sakt", "sakt","dkvmn","dkt+","gkt"]

# # Base path
# BASE_DIR = os.path.join(os.path.dirname(__file__), "ckpts")

# def plot_dataset_comparison(dataset):
#     plt.figure(figsize=(15, 5))

#     metrics_list = []
#     available_models = []

#     for model in models:
#         metrics_path = os.path.join(BASE_DIR, model, dataset, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print(f"⚠ No data found for {model} on {dataset}")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         available_models.append(model)
#         metrics_list.append(metrics)

#         # Plot train loss
#         plt.subplot(1, 3, 1)
#         plt.plot(metrics.get("train_loss", []), label=model)
#         plt.title(f"{dataset} - Train Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()

#         # Plot validation loss
#         plt.subplot(1, 3, 2)
#         plt.plot(metrics.get("valid_loss", []), label=model)
#         plt.title(f"{dataset} - Validation Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()

#         # Plot validation AUC
#         plt.subplot(1, 3, 3)
#         plt.plot(metrics.get("valid_auc", []), label=model)
#         plt.title(f"{dataset} - Validation AUC")
#         plt.xlabel("Epoch")
#         plt.ylabel("AUC")
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Print final epoch values for comparison
#     if metrics_list:
#         summary = {
#             "Model": [],
#             "Final Train Loss": [],
#             "Final Val Loss": [],
#             "Final Val AUC": []
#         }
#         for model, metrics in zip(available_models, metrics_list):
#             summary["Model"].append(model)
#             summary["Final Train Loss"].append(metrics.get("train_loss", [None])[-1])
#             summary["Final Val Loss"].append(metrics.get("valid_loss", [None])[-1])
#             summary["Final Val AUC"].append(metrics.get("valid_auc", [None])[-1])

#         df = pd.DataFrame(summary)
#         print(f"\n📊 Final metrics for dataset: {dataset}")
#         print(df.to_string(index=False))

# # Run for all datasets
# for ds in datasets:
#     plot_dataset_comparison(ds)


# import os
# import pickle
# import matplotlib.pyplot as plt
# import pandas as pd

# # Define datasets and models
# datasets = ["Algebra2005", "ASSIST2009", "ASSIST2015"]
# models = ["dkt", "light_sakt", "sakt", "dkvmn", "dkt+", "gkt"]

# # Base path
# BASE_DIR = os.path.join(os.path.dirname(__file__), "ckpts")

# def plot_dataset_comparison(dataset):
#     plt.figure(figsize=(15, 5))

#     metrics_list = []
#     available_models = []

#     for model in models:
#         metrics_path = os.path.join(BASE_DIR, model, dataset, "metrics.pkl")

#         if not os.path.exists(metrics_path):
#             print(f"⚠ No data found for {model} on {dataset}")
#             continue

#         with open(metrics_path, "rb") as f:
#             metrics = pickle.load(f)

#         available_models.append(model)
#         metrics_list.append(metrics)

#         # Plot train loss
#         plt.subplot(1, 3, 1)
#         plt.plot(metrics.get("train_loss", []), label=model)
#         plt.title(f"{dataset} - Train Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()

#         # Plot validation loss
#         plt.subplot(1, 3, 2)
#         plt.plot(metrics.get("valid_loss", []), label=model)
#         plt.title(f"{dataset} - Validation Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()

#         # Plot validation AUC
#         plt.subplot(1, 3, 3)
#         plt.plot(metrics.get("valid_auc", []), label=model)
#         plt.title(f"{dataset} - Validation AUC")
#         plt.xlabel("Epoch")
#         plt.ylabel("AUC")
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # ✅ NEW: Combined AUC vs Epochs comparison for all models
#     if metrics_list:
#         plt.figure(figsize=(8, 6))
#         for model, metrics in zip(available_models, metrics_list):
#             plt.plot(metrics.get("valid_auc", []), label=model, marker='o')
#         plt.title(f"AUC vs Epochs - {dataset}")
#         plt.xlabel("Epoch")
#         plt.ylabel("Validation AUC")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     # Print final epoch values for comparison
#     if metrics_list:
#         summary = {
#             "Model": [],
#             "Final Train Loss": [],
#             "Final Val Loss": [],
#             "Final Val AUC": []
#         }
#         for model, metrics in zip(available_models, metrics_list):
#             summary["Model"].append(model)
#             summary["Final Train Loss"].append(metrics.get("train_loss", [None])[-1])
#             summary["Final Val Loss"].append(metrics.get("valid_loss", [None])[-1])
#             summary["Final Val AUC"].append(metrics.get("valid_auc", [None])[-1])

#         df = pd.DataFrame(summary)
#         print(f"\n📊 Final metrics for dataset: {dataset}")
#         print(df.to_string(index=False))


# # Run for all datasets
# for ds in datasets:
#     plot_dataset_comparison(ds)


import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Define datasets and models
datasets = ["Algebra2005", "ASSIST2009", "ASSIST2015"]
models = ["dkt", "light_sakt", "sakt", "dkvmn", "dkt+", "gkt"]

# Base path for metrics
BASE_DIR = os.path.join(os.path.dirname(__file__), "ckpts")

# Path to save plots
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_dataset_comparison(dataset):
    plt.figure(figsize=(14, 3.5)) #(widht , height)

    metrics_list = []
    available_models = []

    for model in models:
        metrics_path = os.path.join(BASE_DIR, model, dataset, "metrics.pkl")

        if not os.path.exists(metrics_path):
            print(f"⚠ No data found for {model} on {dataset}")
            continue

        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

        available_models.append(model)
        metrics_list.append(metrics)

        # Plot train loss
        plt.subplot(1, 3, 1)
        plt.plot(metrics.get("train_loss", []), label=model)
        plt.title(f"{dataset} - Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="center left",bbox_to_anchor=(1,0.5))
        plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.7)

        # # Plot validation loss
        # plt.subplot(1, 3, 2)
        # plt.plot(metrics.get("valid_loss", []), label=model)
        # plt.title(f"{dataset} - Validation Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()

        # Plot validation AUC
        plt.subplot(1, 3, 2)
        plt.plot(metrics.get("valid_auc", []), label=model)
        plt.title(f"{dataset} - Validation AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend(loc="center left",bbox_to_anchor=(1, 0.5))
        plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(RESULTS_DIR, f"{dataset}_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved plot: {save_path}")

    plt.close()  # Close figure to free memory

    if metrics_list:
        summary = {
            "Model": [],
            "Final Train Loss": [],
            "Final Val Loss": [],
            "Final Val AUC": []
        }
        for model, metrics in zip(available_models, metrics_list):
            summary["Model"].append(model)
            summary["Final Train Loss"].append(metrics.get("train_loss", [None])[-1])
            summary["Final Val Loss"].append(metrics.get("valid_loss", [None])[-1])
            summary["Final Val AUC"].append(metrics.get("valid_auc", [None])[-1])

        df = pd.DataFrame(summary)
        print(f"\n📊 Final metrics for dataset: {dataset}")
        print(df.to_string(index=False))


# Run for all datasets
for ds in datasets:
    plot_dataset_comparison(ds)


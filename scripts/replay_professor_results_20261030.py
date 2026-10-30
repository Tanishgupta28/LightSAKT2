"""Independent CPU validation/test replay with checkpoint-bound cache."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.architecture_study import read_json, digest
from scripts.architecture_pilot import PROJECT, SOURCE, write_json
from scripts.pilot_metric_extensions import extended_metrics
from scripts.professor_data_20261030 import FeatureSequences
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.engine import evaluate

OUT = PROJECT / "reports/professor_results_20261030"


def verify_run(dataset, model_name, seed, directory, root, features=None):
    directory, root = Path(directory), Path(root)
    if not (directory / "metrics.json").exists() or read_json(directory / "status.json")["status"] != "completed":
        return None
    recorded = read_json(directory / "metrics.json"); config = read_json(directory / "config.json")
    checkpoint = Path(recorded["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == recorded["checkpoint_sha256"]
    cache = OUT / "replays" / f"{dataset}-{model_name}-{seed}.json"
    if cache.exists():
        previous = read_json(cache)
        assert previous["checkpoint_sha256"] == checkpoint_hash
        return previous
    stats = read_json(SOURCE / f"reports/{dataset}_statistics.json")
    data = SOURCE / f"data/processed/{dataset}/interactions.csv"
    assert digest(data) == stats["processed_sha256"] == recorded["processed_sha256"]
    assert digest(SOURCE / f"data/splits/{dataset}.json") == stats["split_sha256"] == recorded["split_sha256"]
    best = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert best["epoch"] == recorded["best_epoch"]
    for key in ["dataset", "model", "seed", "seq_len", "batch_size", "learning_rate", "betas", "weight_decay", "epsilon", "max_epochs", "patience", "model_config"]:
        assert best["config"][key] == config[key], (str(checkpoint), key)
    assert config["dataset"] == dataset and config["seed"] == seed
    assert best["processed_sha256"] == stats["processed_sha256"] and best["split_sha256"] == stats["split_sha256"]
    assert all(torch.isfinite(x).all() for x in best["model"].values())
    history_path = root / "results/raw" / checkpoint.parent.name / "history.json"
    if history_path.exists():
        history = read_json(history_path)
        assert max(history, key=lambda r: r["validation"]["auc"])["epoch"] == best["epoch"]
    model = build_model(config["model"], stats["num_q"], config["model_config"]).cpu()
    model.load_state_dict(best["model"], strict=True)
    params = sum(p.numel() for p in model.parameters())
    assert params == recorded["parameters"]
    replayed = {}; differences = {}; prediction_hashes = {}
    for split in ["validation", "test"]:
        if config["model"] == "LightSAKTQuestion":
            path = Path(features["train_validation_feature_path" if split == "validation" else "test_feature_path"])
            assert digest(path) == features["train_validation_feature_sha256" if split == "validation" else "test_feature_sha256"]
            dataset_object = FeatureSequences(data, split, config["seq_len"], path)
        else:
            dataset_object = Sequences(data, split, config["seq_len"])
        _, target, prediction = evaluate(model, config["model"], DataLoader(dataset_object, batch_size=8, shuffle=False), "cpu")
        m = extended_metrics(target, prediction)
        assert m["n_targets"] == dataset_object.n_targets
        replayed[split] = m
        reference = recorded.get(split)
        if reference is not None:
            difference = {k: abs(m[k] - reference[k]) for k in ["auc", "accuracy", "f1", "bce"]}
            assert difference["auc"] < 2e-6 and difference["bce"] < 2e-6, difference
            assert difference["accuracy"] < 1e-4 and difference["f1"] < 1e-4, difference
            assert m["n_targets"] == reference["n_targets"]
            differences[split] = difference
        local = PROJECT / ".tools/professor_campaign_20261030/predictions" / f"{dataset}-{model_name}-{seed}-{split}.npz"
        local.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(local, target=target, prediction=prediction)
        prediction_hashes[split] = {"path": str(local), "sha256": digest(local)}
    row = {"status": "VERIFIED_CPU_REPLAY", "dataset": dataset, "model": model_name, "seed": seed,
           "run_id": config["run_id"], "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
           "config": config, "config_file": str(directory / "config.json"), "config_sha256": digest(directory / "config.json"),
           "recorded_metrics_path": str(directory / "metrics.json"), "recorded_metrics_sha256": digest(directory / "metrics.json"),
           "validation": recorded["validation"], "test": replayed["test"], "cpu_replay": replayed,
           "recorded_comparison_absolute_differences": differences, "parameters": params,
           "best_epoch": recorded["best_epoch"], "epochs_trained": recorded["epochs_trained"],
           "training_seconds": recorded["training_seconds"], "peak_allocated_bytes": recorded.get("peak_allocated_bytes"),
           "processed_sha256": stats["processed_sha256"], "split_sha256": stats["split_sha256"],
           "selection": "maximum validation AUC,strict improvement,patience5,max50;test only after selection",
           "recorded_training_code_sha256": recorded.get("training_code_sha256"), "selected_git_commit": best.get("git_commit"),
           "precision_recall": "CPU replay,threshold0.5,globally pooled masked targets", "prediction_artifacts": prediction_hashes,
           "verification_utc": datetime.now(timezone.utc).isoformat(), "inference_device": "cpu", "inference_batch_size": 8}
    write_json(cache, row)
    print(f"VERIFIED {dataset} {model_name} seed{seed}: val={row['validation']['auc']:.9f} test={row['test']['auc']:.9f}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--baselines", action="store_true"); args = parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    plan = read_json(PROJECT / "experiments/professor_campaign_20261030/plan.json")
    if args.baselines:
        for dataset in ["ASSIST2009", "ASSIST2015", "Algebra2005"]:
            for name in ["SAKT", "LightSAKT"]:
                for seed in [42, 123, 2026, 3407, 7777]:
                    verify_run(dataset, name, seed, SOURCE / f"results/raw/{dataset}-{name}-final-{seed}", SOURCE)
    for entry in plan["entries"]:
        verify_run(entry["dataset"], "LightSAKT+Difficulty", entry["seed"], Path(entry["work"]) / "results/raw" / entry["run_id"],
                   Path(entry["work"]), plan["feature_specs"][entry["dataset"]])


if __name__ == "__main__":
    main()

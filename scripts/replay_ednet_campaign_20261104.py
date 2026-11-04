"""CPU-only checkpoint restoration, independent metrics and recovery-state audit.

Never trains, changes selection, or writes into historical result directories.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from scripts.ednet_replacement_data import ROOT, WORK, OUT, DS, write
from scripts.architecture_study import digest
from scripts.professor_data_20261030 import FeatureSequences
from scripts.pilot_metric_extensions import extended_metrics
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.engine import evaluate

PLAN = ROOT / "experiments/ednet_replacement_20261104/plan.json"


def independent_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    assert y.shape == p.shape and y.ndim == 1
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    truth, predicted = y == 1, p >= .5
    tp = int(np.count_nonzero(truth & predicted)); fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted)); tn = int(np.count_nonzero(~truth & ~predicted))
    positive = int(truth.sum()); negative = len(y) - positive
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {"auc": float((rankdata(p)[truth].sum() - positive * (positive + 1) / 2) / (positive * negative)),
            "accuracy": (tp + tn) / len(y), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn) if tp + fn else 0.,
            "bce": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log(1 - clipped))),
            "n_targets": len(y)}


def state_audit(entry, config, recorded, train_chunks):
    raw = WORK / "results/raw" / entry["run_id"]
    best_path, latest_path = [WORK / "checkpoints" / entry["run_id"] / f"{name}.pt" for name in ["best", "latest"]]
    best, latest = [torch.load(path, map_location="cpu", weights_only=False) for path in [best_path, latest_path]]
    history = json.loads((raw / "history.json").read_text())
    assert config == best["config"] == latest["config"] == json.loads(Path(entry["config"]).read_text())
    assert latest["history"] == history
    assert [h["epoch"] for h in history] == list(range(1, latest["epoch"] + 1))
    selected = max(history, key=lambda h: h["validation"]["auc"])
    assert selected["epoch"] == best["epoch"] == latest["best_epoch"] == recorded["best_epoch"]
    assert selected["validation"] == recorded["validation"] == best["validation"]
    assert latest["epoch"] == recorded["epochs_trained"]
    assert latest["epoch"] == config["max_epochs"] or latest["bad_epochs"] == config["patience"]
    assert recorded["test"] is None and config["evaluate_test"] is False
    assert digest(best_path) == recorded["checkpoint_sha256"]
    for state in [best, latest]:
        assert state["processed_sha256"] == recorded["processed_sha256"]
        assert state["split_sha256"] == recorded["split_sha256"]
        assert all(torch.isfinite(v).all() for v in state["model"].values())
    for state in latest["optimizer"]["state"].values():
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    rng = latest["rng"]
    random.Random().setstate(rng["python"]); np.random.RandomState().set_state(rng["numpy"])
    torch.Generator().set_state(rng["torch"].cpu())
    assert len(rng["cuda"]) == 1 and rng["cuda"][0].dtype == torch.uint8
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(range(train_chunks), batch_size=config["batch_size"], shuffle=True, num_workers=0, generator=generator)
    for _ in history:
        for _ in loader:
            pass
    assert torch.equal(generator.get_state(), latest["loader_rng"].cpu())
    expected_steps = len(history) * math.ceil(train_chunks / config["batch_size"])
    assert sorted(set(int(v["step"]) for v in latest["optimizer"]["state"].values())) == [expected_steps]
    assert recorded["training_seconds"] < 900 and recorded["peak_allocated_bytes"] <= 4 * 1024**3
    return best, {"run_id": entry["run_id"], "status": "PASS", "best_sha256": digest(best_path),
        "latest_sha256": digest(latest_path), "epochs": len(history), "best_epoch": best["epoch"],
        "optimizer_finite": True, "saved_rng_states_valid": True, "loader_rng_matches_seeded_epochs": True,
        "adam_steps": expected_steps, "no_test_selection": True,
        "difficulty_scale": float(best["model"]["difficulty_scale"]) if "difficulty_scale" in best["model"] else None,
        "training_git_commit": best["git_commit"]}


def verify(entry, plan, frame):
    raw = WORK / "results/raw" / entry["run_id"]
    if not (raw / "status.json").exists() or json.loads((raw / "status.json").read_text())["status"] != "completed":
        return None
    config = json.loads((raw / "config.json").read_text()); recorded = json.loads((raw / "metrics.json").read_text())
    assert digest(entry["config"]) == entry["config_sha256"]
    # The native trainer serializes the same JSON without the preparation
    # helper's trailing newline. Bind the frozen file bytes and parsed values.
    assert config == json.loads(Path(entry["config"]).read_text())
    assert config["dataset"] == DS and config["seed"] == entry["seed"]
    cache = OUT / "replays" / f"{DS}-{entry['model']}-{entry['seed']}.json"
    checkpoint = Path(recorded["checkpoint"])
    assert digest(checkpoint) == recorded["checkpoint_sha256"]
    if cache.exists():
        row = json.loads(cache.read_text())
        assert row["checkpoint_sha256"] == recorded["checkpoint_sha256"]
        for item in row["prediction_artifacts"].values():
            assert digest(item["path"]) == item["sha256"]
        return row
    chunks = plan["feature_audit"]["training_chunks"]
    best, audit = state_audit(entry, config, recorded, chunks)
    model = build_model(config["model"], plan["data_stats"]["num_q"], config["model_config"]).cpu()
    model.load_state_dict(best["model"], strict=True)
    params = sum(p.numel() for p in model.parameters()); assert params == recorded["parameters"]
    replayed, artifacts, differences = {}, {}, {}
    for split in ["validation", "test"]:
        data_path = WORK / f"data/processed/{DS}/interactions.csv"
        if config["model"] == "LightSAKTQuestion":
            key = "train_validation" if split == "validation" else "test"
            feature_path = Path(plan["feature_spec"][key + "_feature_path"])
            assert digest(feature_path) == plan["feature_spec"][key + "_feature_sha256"]
            data = FeatureSequences(data_path, split, config["seq_len"], feature_path)
        else:
            data = Sequences(data_path, split, config["seq_len"])
        _, y, p = evaluate(model, config["model"], DataLoader(data, batch_size=8, shuffle=False), "cpu")
        m = extended_metrics(y, p); independently = independent_metrics(y, p)
        assert max(abs(m[k] - v) for k, v in independently.items()) < 1e-12
        expected = np.concatenate([g.label.to_numpy()[1:] for _, g in frame[frame.split == split].groupby("student", sort=True)])
        assert np.array_equal(y, expected) and len(y) == plan["data_stats"]["split_targets"][split]
        replayed[split] = m
        if split == "validation":
            difference = {k: abs(m[k] - recorded[split][k]) for k in ["auc", "accuracy", "f1", "bce"]}
            assert difference["auc"] < 2e-6 and difference["bce"] < 2e-6
            assert difference["accuracy"] < 1e-4 and difference["f1"] < 1e-4
            differences[split] = difference
        destination = WORK / "results/predictions" / f"{entry['run_id']}-{split}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, target=y, prediction=p)
        artifacts[split] = {"path": str(destination), "sha256": digest(destination)}
    row = {"status": "VERIFIED_CPU_REPLAY", "dataset": DS,
        "model": "LightSAKT+Difficulty" if config["model"] == "LightSAKTQuestion" else config["model"],
        "seed": config["seed"], "run_id": config["run_id"], "config": config,
        "config_file": str(raw / "config.json"), "config_sha256": digest(raw / "config.json"),
        "checkpoint": str(checkpoint), "checkpoint_sha256": recorded["checkpoint_sha256"],
        "recorded_metrics_path": str(raw / "metrics.json"), "recorded_metrics_sha256": digest(raw / "metrics.json"),
        "validation": recorded["validation"], "test": replayed["test"], "cpu_replay": replayed,
        "recorded_comparison_absolute_differences": differences, "prediction_artifacts": artifacts,
        "parameters": params, "best_epoch": recorded["best_epoch"], "epochs_trained": recorded["epochs_trained"],
        "training_seconds": recorded["training_seconds"], "peak_allocated_bytes": recorded["peak_allocated_bytes"],
        "peak_reserved_bytes": recorded["peak_reserved_bytes"], "processed_sha256": recorded["processed_sha256"],
        "split_sha256": recorded["split_sha256"], "selected_git_commit": best["git_commit"],
        "recorded_training_code_sha256": recorded["training_code_sha256"], "checkpoint_state_audit": audit,
        "verification_utc": datetime.now(timezone.utc).isoformat(), "inference_device": "cpu", "inference_batch_size": 8,
        "selection": "strict validation AUC maximum;patience5,max50;test after frozen selection",
        "mask_and_metrics": "exact original per-student labels[1:]; independent rank-sum AUC and confusion/BCE match <1e-12"}
    write(cache, row)
    print(f"VERIFIED {row['model']} seed{row['seed']}: val={row['validation']['auc']:.9f} test={row['test']['auc']:.9f}", flush=True)
    return row


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    plan = json.loads(PLAN.read_text())
    for mapping in ["data_binding_sha256", "scientific_code_sha256", "hold_sha256", "protected_sha256"]:
        for path, expected in plan[mapping].items():
            assert digest(path) == expected, path
    frame = pd.read_csv(WORK / f"data/processed/{DS}/interactions.csv", dtype={"student": str})
    rows = [verify(entry, plan, frame) for entry in plan["entries"]]
    rows = [r for r in rows if r is not None]
    write(OUT / "checkpoint_state_audit.json", {"status": "PASS_COMPLETED_RUNS", "completed": len(rows),
        "expected": 15, "runs": [r["checkpoint_state_audit"] for r in rows]})
    write(OUT / "protected_artifacts_audit.json", {"status": "PASS", "unchanged_protected_files": len(plan["protected_sha256"]),
        "frozen_science_data_configs_and_holds": True})


if __name__ == "__main__":
    main()

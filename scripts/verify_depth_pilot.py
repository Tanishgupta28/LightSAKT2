"""Independent CPU checkpoint replay of the completed validation-only depth pair."""
import hashlib
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import DataLoader

from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL, PILOT, write_json
from scripts.architecture_study import read_json, digest
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.engine import evaluate


def verify():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    work = PROJECT / "pilot-workspace"
    pilot = read_json(PROJECT / CONTROL / "pilot.json")
    stats = read_json(SOURCE / "reports/ASSIST2009_statistics.json")
    code_hash = hashlib.sha256(b"".join(
        str(p).encode() + p.read_bytes() for p in sorted(Path("src").rglob("*.py"))
        if "visualization" not in p.parts)).hexdigest()
    executed_grant_hash = digest(PROJECT / CONTROL / "gpu_authorization_20261025.executed.json")
    data = SOURCE / "data/processed/ASSIST2009/interactions.csv"
    assert digest(data) == stats["processed_sha256"]
    assert digest(SOURCE / "data/splits/ASSIST2009.json") == stats["split_sha256"]
    validation = Sequences(data, "validation", 100)
    loader = DataLoader(validation, batch_size=64, shuffle=False, num_workers=0)
    rows = []
    for arm in ("baseline", "two_layer_linear"):
        config = pilot["configurations"][arm]
        directory = work / "results/raw" / config["run_id"]
        result = read_json(directory / "metrics.json")
        history = read_json(directory / "history.json")
        assert read_json(directory / "status.json")["status"] == "completed"
        assert read_json(directory / "config.json") == config and config["evaluate_test"] is False
        assert result["test"] is None and not (work / "results/predictions").exists()
        assert digest(result["checkpoint"]) == result["checkpoint_sha256"]
        best = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        latest_path = work / "checkpoints" / config["run_id"] / "latest.pt"
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert best["config"] == latest["config"] == config
        assert latest["epoch"] == len(history) == result["epochs_trained"]
        assert best["epoch"] == max(history, key=lambda row: row["validation"]["auc"])["epoch"]
        assert latest["environment"]["training_code_sha256"] == result["training_code_sha256"] == code_hash
        assert latest["environment"]["observer"]["gpu_authorization_sha256"] == executed_grant_hash
        assert all(key in latest for key in ("optimizer", "rng", "loader_rng", "bad_epochs", "history"))
        for state in (best, latest):
            assert state["processed_sha256"] == stats["processed_sha256"]
            assert state["split_sha256"] == stats["split_sha256"]
            assert all(torch.isfinite(tensor).all() for tensor in state["model"].values())
        for optimizer_state in latest["optimizer"]["state"].values():
            assert all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                       for value in optimizer_state.values())
        model = build_model(config["model"], stats["num_q"], config["model_config"]).cpu()
        model.load_state_dict(best["model"], strict=True)
        replay, _, _ = evaluate(model, config["model"], loader, "cpu")
        differences = {key: abs(replay[key] - result["validation"][key])
                       for key in ("auc", "f1", "accuracy", "bce")}
        assert differences["auc"] <= 2e-6 and differences["bce"] <= 2e-6
        assert differences["f1"] <= 1e-4 and differences["accuracy"] <= 1e-4
        assert replay["n_targets"] == result["validation"]["n_targets"] == 27456
        rows.append({"arm": arm, "status": "PASS", "validation_replay": replay,
                     "absolute_differences": differences, "best_checkpoint_sha256": digest(result["checkpoint"]),
                     "latest_checkpoint_sha256": digest(latest_path), "selected_epoch": best["epoch"],
                     "completed_epochs": latest["epoch"], "complete_resume_state_present": True})
    historical = read_json(SOURCE / "results/raw/ASSIST2009-LightSAKT-grid-42-f08e9494d270/history.json")
    fresh = read_json(work / "results/raw" / pilot["configurations"]["baseline"]["run_id"] / "history.json")
    maximum_baseline_difference = max(abs(a["validation"]["auc"] - b["validation"]["auc"])
                                      for a, b in zip(historical, fresh))
    assert len(historical) == len(fresh) and maximum_baseline_difference == 0
    record = {"status": "PASS", "device": "cpu", "optimizer_updates_performed": 0,
              "split": "validation", "test_loader_constructed": False, "test_predictions_accessed": False,
              "baseline_epoch_auc_maximum_difference_from_original": maximum_baseline_difference,
              "training_code_sha256": code_hash, "rows": rows}
    write_json(PROJECT / PILOT / "checkpoint_replay_verification.json", record)
    print("PASS: both selected checkpoints replay validation metrics; full resume states are finite and source-bound.")


if __name__ == "__main__":
    verify()

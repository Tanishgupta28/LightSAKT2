"""Replay only validation and verify the bounded seed-42 softmax artifacts."""
import hashlib
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import DataLoader

from scripts.architecture_pilot import CONTROL, PROJECT, SOURCE, SOFTMAX_PILOT, write_json
from scripts.architecture_study import digest, read_json
from scripts.pilot_metric_extensions import extended_metrics
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.engine import evaluate


def verify(arm="one_layer_softmax", report_dir=SOFTMAX_PILOT,
           grant_name="gpu_authorization_softmax_20261027.executed.json"):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    work = PROJECT / "pilot-workspace"
    config = read_json(PROJECT / CONTROL / "configs" / f"{arm}.json")
    directory = work / "results/raw" / config["run_id"]
    result = read_json(directory / "metrics.json")
    history = read_json(directory / "history.json")
    stats = read_json(SOURCE / "reports/ASSIST2009_statistics.json")
    assert read_json(directory / "status.json")["status"] == "completed"
    assert read_json(directory / "config.json") == config
    assert config["evaluate_test"] is False and result["test"] is None
    assert not (work / "results/predictions").exists()
    assert digest(result["checkpoint"]) == result["checkpoint_sha256"]
    best = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    latest_path = work / "checkpoints" / config["run_id"] / "latest.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert best["config"] == latest["config"] == config
    assert latest["epoch"] == len(history) == result["epochs_trained"]
    assert best["epoch"] == result["best_epoch"] == max(
        history, key=lambda row: row["validation"]["auc"])["epoch"]
    code_hash = hashlib.sha256(b"".join(
        str(p).encode() + p.read_bytes() for p in sorted(Path("src").rglob("*.py"))
        if "visualization" not in p.parts)).hexdigest()
    assert latest["environment"]["training_code_sha256"] == result["training_code_sha256"] == code_hash
    grant = PROJECT / CONTROL / grant_name
    assert latest["environment"]["observer"]["gpu_authorization_sha256"] == digest(grant)
    assert all(key in latest for key in ("optimizer", "rng", "loader_rng", "bad_epochs", "history"))
    assert latest["bad_epochs"] >= config["patience"] or len(history) == config["max_epochs"]
    data = SOURCE / "data/processed/ASSIST2009/interactions.csv"
    assert digest(data) == stats["processed_sha256"] == config["pilot"]["processed_sha256"]
    assert digest(SOURCE / "data/splits/ASSIST2009.json") == stats["split_sha256"] == config["pilot"]["split_sha256"]
    for state in (best, latest):
        for key in ("processed_sha256", "split_sha256"):
            assert state[key] == stats[key]
        assert all(torch.isfinite(tensor).all() for tensor in state["model"].values())
    for state in latest["optimizer"]["state"].values():
        assert all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                   for value in state.values())
    loader = DataLoader(Sequences(data, "validation", config["seq_len"]),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = build_model(config["model"], stats["num_q"], config["model_config"]).cpu()
    model.load_state_dict(best["model"], strict=True)
    _, target, prediction = evaluate(model, config["model"], loader, "cpu")
    replay = extended_metrics(target, prediction)
    differences = {key: abs(replay[key] - result["validation"][key])
                   for key in ("auc", "f1", "accuracy", "bce", "precision", "recall")}
    assert differences["auc"] <= 2e-6 and differences["bce"] <= 2e-6
    assert all(differences[key] <= 1e-4 for key in ("f1", "accuracy", "precision", "recall"))
    assert replay["n_targets"] == result["validation"]["n_targets"] == 27456
    preservation = read_json(PROJECT / report_dir / "preflight_20261027.json")
    for path, expected in preservation["existing_file_sha256"].items():
        assert digest(path) == expected, f"Pre-existing artifact changed: {path}"
    record = {"status": "PASS", "device": "cpu", "optimizer_updates_performed": 0,
              "test_loader_constructed": False, "test_predictions_accessed": False,
              "split": "validation", "validation_replay": replay, "absolute_differences": differences,
              "best_checkpoint_sha256": digest(result["checkpoint"]),
              "latest_checkpoint_sha256": digest(latest_path), "selected_epoch": best["epoch"],
              "completed_epochs": latest["epoch"], "complete_resume_state_present": True,
              "training_code_sha256": code_hash, "executed_grant_sha256": digest(grant),
              "pre_existing_file_hashes_unchanged": len(preservation["existing_file_sha256"])}
    if arm == "one_layer_recency_softmax":
        initial_raw = torch.full_like(best["model"]["attn.raw_decay"],
                                      math.log(math.expm1(1 / 32)))
        assert not torch.equal(best["model"]["attn.raw_decay"], initial_raw)
        slopes = torch.nn.functional.softplus(best["model"]["attn.raw_decay"])
        assert torch.isfinite(slopes).all() and (slopes > 0).all()
        record["initial_decay_per_head"] = 1 / 32
        record["selected_decay_per_head"] = slopes.tolist()
        record["selected_decay_changed_from_initial"] = True
    write_json(PROJECT / report_dir / "checkpoint_replay_verification_20261027.json", record)
    print(f"PASS: {arm} validation replay, resume-state checks and historical artifact preservation.")


if __name__ == "__main__":
    verify()

"""Independent validation-only CPU replay, including explicit interrupted state."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.architecture_pilot import QUESTION_PILOT, PROJECT, SOURCE, CONTROL, write_json
from scripts.architecture_study import read_json, digest
from scripts.question_difficulty_inputs import QuestionDifficultySequences
from scripts.pilot_metric_extensions import extended_metrics
from scripts.verify_softmax_pilot import verify
from src.models.factory import build_model
from src.training.engine import evaluate


def verify_partial():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    work = PROJECT / "pilot-workspace"; out = PROJECT / QUESTION_PILOT
    config = read_json(PROJECT / CONTROL / "configs/one_layer_question_difficulty.json")
    directory = work / "results/raw" / config["run_id"]
    status = read_json(directory / "status.json")
    assert status["status"] == "interrupted" and not (directory / "metrics.json").exists()
    assert read_json(directory / "config.json") == config and config["evaluate_test"] is False
    checkpoint_dir = work / "checkpoints" / config["run_id"]
    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    history = read_json(directory / "history.json")
    assert best["config"] == latest["config"] == config
    assert latest["epoch"] == status["last_completed_epoch"] == len(history)
    assert best["epoch"] == latest["best_epoch"] == max(history, key=lambda r: r["validation"]["auc"])["epoch"]
    stats = read_json(SOURCE / "reports/ASSIST2009_statistics.json")
    for state in (best, latest):
        assert all(torch.isfinite(x).all() for x in state["model"].values())
        for key in ("processed_sha256", "split_sha256"):
            assert state[key] == config["pilot"][key] == stats[key]
    assert digest(SOURCE / "data/processed/ASSIST2009/interactions.csv") == stats["processed_sha256"]
    assert digest(SOURCE / "data/splits/ASSIST2009.json") == stats["split_sha256"]
    assert all(k in latest for k in ["optimizer", "rng", "loader_rng", "bad_epochs", "history"])
    for state in latest["optimizer"]["state"].values():
        assert all(not isinstance(x, torch.Tensor) or torch.isfinite(x).all() for x in state.values())
    code_hash = hashlib.sha256(b"".join(str(p).encode() + p.read_bytes()
        for p in sorted(Path("src").rglob("*.py")) if "visualization" not in p.parts)).hexdigest()
    assert latest["environment"]["training_code_sha256"] == code_hash
    grant = PROJECT / CONTROL / "gpu_authorization_question_20261028.executed.json"
    assert latest["environment"]["observer"]["gpu_authorization_sha256"] == digest(grant)
    feature_path = work / "data/auxiliary/ASSIST2009-question-difficulty.json"
    assert digest(feature_path) == config["question_difficulty"]["feature_sha256"]
    model = build_model(config["model"], stats["num_q"], config["model_config"]).cpu()
    model.load_state_dict(best["model"], strict=True)
    dataset = QuestionDifficultySequences(SOURCE / "data/processed/ASSIST2009/interactions.csv",
                                          "validation", 100, feature_path)
    _, y, prediction = evaluate(model, config["model"], DataLoader(dataset, batch_size=64), "cpu")
    replay = extended_metrics(y, prediction)
    selected = next(r for r in history if r["epoch"] == best["epoch"])["validation"]
    differences = {k: abs(replay[k] - selected[k]) for k in ["auc", "f1", "accuracy", "bce", "precision", "recall"]}
    assert differences["auc"] < 2e-6 and differences["bce"] < 2e-6
    assert all(differences[k] < 1e-4 for k in ["f1", "accuracy", "precision", "recall"])
    assert replay["n_targets"] == 27456
    protected = read_json(out / "preflight_20261028.json")["existing_file_sha256"]
    for path, expected in protected.items():
        assert digest(path) == expected, path
    record = {"status": "PASS_PARTIAL_CHECKPOINT_VERIFICATION", "training_status": "interrupted",
              "interruption_cause": status["failure_reason"], "epochs_completed": latest["epoch"],
              "selected_epoch": best["epoch"], "selected_partial_validation": selected,
              "validation_replay": replay, "absolute_differences": differences,
              "elapsed_seconds_at_last_saved_epoch": latest["elapsed_seconds"],
              "bad_epochs": latest["bad_epochs"], "early_stopping_complete": False,
              "full_resume_state_present": True, "best_checkpoint_sha256": digest(checkpoint_dir / "best.pt"),
              "latest_checkpoint_sha256": digest(checkpoint_dir / "latest.pt"),
              "learned_difficulty_scale": best["model"]["difficulty_scale"].item(),
              "training_code_sha256": code_hash, "question_feature_sha256": digest(feature_path),
              "protected_file_hashes_unchanged": len(protected), "test_evaluation": False,
              "optimizer_updates_during_verification": 0, "no_metrics_json_completion_claim": True}
    write_json(out / "partial_checkpoint_verification_20261028.json", record)
    np.savez_compressed(PROJECT / ".tools/diagnostics_20261028/question_validation.npz", target=y, prediction=prediction)
    print(f"PASS: interrupted checkpoint epoch {latest['epoch']}; replay AUC {replay['auc']:.9f}; no training resumed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.partial:
        verify_partial()
    else:
        verify("one_layer_question_difficulty", QUESTION_PILOT,
               "gpu_authorization_question_20261028.executed.json")

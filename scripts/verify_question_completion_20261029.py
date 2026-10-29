"""Independent CPU replay and provenance audit after the exact continuation."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL, write_json
from scripts.architecture_study import read_json, digest
from scripts.question_difficulty_inputs import QuestionDifficultySequences
from scripts.pilot_metric_extensions import extended_metrics
from src.models.factory import build_model
from src.training.engine import evaluate

REPORT = Path("reports/architecture_study/question_completion_20261029")


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = PROJECT / REPORT; work = PROJECT / "pilot-workspace"
    config = read_json(PROJECT / CONTROL / "configs/one_layer_question_difficulty.json")
    directory = work / "results/raw" / config["run_id"]
    result = read_json(directory / "metrics.json"); history = read_json(directory / "history.json")
    assert read_json(directory / "status.json")["status"] == "completed"
    assert read_json(directory / "config.json") == config and config["evaluate_test"] is False and result["test"] is None
    before = read_json(out / "before_resume_records/history.json")
    assert len(before) == 18 and history[:18] == before
    assert [r["epoch"] for r in history] == list(range(1, len(history) + 1))
    checkpoint_dir = work / "checkpoints" / config["run_id"]
    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert best["config"] == latest["config"] == config
    assert latest["epoch"] == result["epochs_trained"] == len(history) > 18
    assert best["epoch"] == latest["best_epoch"] == result["best_epoch"] == max(history, key=lambda r: r["validation"]["auc"])["epoch"]
    assert latest["bad_epochs"] >= config["patience"] or latest["epoch"] == config["max_epochs"]
    assert all(k in latest for k in ["optimizer", "rng", "loader_rng", "history", "bad_epochs"])
    for state in [best, latest]:
        assert all(torch.isfinite(x).all() for x in state["model"].values())
        for key in ["processed_sha256", "split_sha256"]:
            assert state[key] == config["pilot"][key] == result[key]
    for state in latest["optimizer"]["state"].values():
        assert all(not isinstance(x, torch.Tensor) or torch.isfinite(x).all() for x in state.values())
    source_hash = hashlib.sha256(b"".join(str(p).encode() + p.read_bytes()
        for p in sorted(Path("src").rglob("*.py")) if "visualization" not in p.parts)).hexdigest()
    assert source_hash == result["training_code_sha256"] == latest["environment"]["training_code_sha256"]
    audit = read_json(out / "leakage_and_resume_audit.json")
    assert latest["environment"]["observer"]["resume_identity"] == audit["observer_resume_identity"]
    grant = PROJECT / CONTROL / "gpu_authorization_question_20261029.executed.json"
    assert latest["environment"]["observer"]["gpu_authorization_sha256"] == digest(grant)
    assert digest(checkpoint_dir / "best.pt") == result["checkpoint_sha256"]
    feature_path = work / "data/auxiliary/ASSIST2009-question-difficulty.json"
    assert digest(feature_path) == config["question_difficulty"]["feature_sha256"]
    dataset = QuestionDifficultySequences(SOURCE / "data/processed/ASSIST2009/interactions.csv",
                                          "validation", 100, feature_path)
    model = build_model(config["model"], 137, config["model_config"]).cpu()
    model.load_state_dict(best["model"], strict=True)
    _, y, prediction = evaluate(model, config["model"], DataLoader(dataset, batch_size=64), "cpu")
    replay = extended_metrics(y, prediction)
    differences = {k: abs(replay[k] - result["validation"][k]) for k in ["auc", "f1", "accuracy", "bce", "precision", "recall"]}
    assert differences["auc"] < 2e-6 and differences["bce"] < 2e-6
    assert all(differences[k] < 1e-4 for k in ["f1", "accuracy", "precision", "recall"])
    assert replay["n_targets"] == 27456
    protected = read_json(out / "preflight_20261029.json")["existing_file_sha256"]
    for path, expected in protected.items():
        assert digest(path) == expected, path
    record = {"status": "PASS_COMPLETED_CONTINUATION", "epochs_completed": len(history),
              "resumed_at_epoch": 19, "preserved_history_prefix_epochs": 18,
              "first_18_rows_exactly_preserved": True, "selected_epoch": best["epoch"],
              "stopping_state": {"bad_epochs": latest["bad_epochs"], "max_epochs": config["max_epochs"], "patience": config["patience"]},
              "validation_replay": replay, "absolute_differences": differences,
              "best_checkpoint_sha256": digest(checkpoint_dir / "best.pt"),
              "latest_checkpoint_sha256": digest(checkpoint_dir / "latest.pt"),
              "training_code_sha256": source_hash, "learned_difficulty_scale": best["model"]["difficulty_scale"].item(),
              "resume_observer_identity_unchanged": True, "protected_file_hashes_unchanged": len(protected),
              "test_evaluation": False, "optimizer_updates_during_verification": 0,
              "scope": "Offline student-disjoint evaluation with fitted full-training statistics; not forward-time deployment proof."}
    write_json(out / "completion_verification.json", record)
    local = PROJECT / ".tools/diagnostics_20261029"; local.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(local / "completed_question_validation.npz", target=y, prediction=prediction)
    print(f"PASS: completed {len(history)} epochs, best {best['epoch']}, replay AUC {replay['auc']:.9f}; exact 18-epoch prefix preserved.")


if __name__ == "__main__":
    main()

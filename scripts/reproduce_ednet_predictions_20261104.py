"""Fresh CPU restore of seed42 in each family, without altering saved predictions."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.ednet_replacement_data import WORK, OUT, DS, write
from scripts.architecture_study import digest
from scripts.professor_data_20261030 import FeatureSequences
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.engine import evaluate


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    checks = []
    stats = json.loads((OUT / "dataset_statistics.json").read_text())
    spec = json.loads((OUT / "feature_specification.json").read_text())
    for name in ["SAKT", "LightSAKT", "LightSAKTQuestion"]:
        record = json.loads((OUT / "replays" / f"{DS}-{name}-42.json").read_text())
        assert digest(record["checkpoint"]) == record["checkpoint_sha256"]
        state = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        cfg = record["config"]
        model = build_model(name, stats["num_q"], cfg["model_config"]).cpu()
        model.load_state_dict(state["model"], strict=True)
        for split in ["validation", "test"]:
            path = WORK / f"data/processed/{DS}/interactions.csv"
            if name == "LightSAKTQuestion":
                key = "train_validation" if split == "validation" else "test"
                sidecar = spec[key + "_feature_path"]
                assert digest(sidecar) == spec[key + "_feature_sha256"]
                data = FeatureSequences(path, split, cfg["seq_len"], sidecar)
            else: data = Sequences(path, split, cfg["seq_len"])
            _, y, p = evaluate(model, name, DataLoader(data, batch_size=8, shuffle=False), "cpu")
            saved = record["prediction_artifacts"][split]
            assert digest(saved["path"]) == saved["sha256"]
            with np.load(saved["path"]) as arrays:
                assert np.array_equal(y, arrays["target"])
                assert np.array_equal(p, arrays["prediction"])
            checks.append({"model": name, "seed": 42, "split": split, "targets": len(y),
                "fresh_restore_predictions_bitwise_equal": True, "saved_file_unchanged": True})
    write(OUT / "fresh_prediction_reproduction.json", {"status": "PASS", "unique_checkpoints": 3,
        "split_reproductions": checks, "purpose": "Explicit independent repeat CPU inference to check saved-prediction reproducibility; no extra training or GPU inference"})
    print("PASS six fresh CPU split reproductions, bitwise equal, no artifact overwritten")


if __name__ == "__main__": main()

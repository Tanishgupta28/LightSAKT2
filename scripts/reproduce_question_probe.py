"""Reproduce the selected train-fitted diagnostic using frozen local features."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL
from scripts.architecture_study import read_json
from scripts.question_difficulty_inputs import QuestionDifficultySequences
from scripts.diagnose_lightsakt_20261028 import infer, logit
from src.models.factory import build_model
from src.datasets.sequences import Sequences
from src.evaluation.metrics import metrics


def main():
    torch.set_num_threads(1)
    work = PROJECT / "pilot-workspace"
    config = read_json(PROJECT / CONTROL / "configs/baseline.json")
    local = PROJECT / ".tools/diagnostics_20261028/baseline_train_validation.npz"
    if local.exists():
        cache = dict(np.load(local))
    else:
        model = build_model("LightSAKT", 137, config["model_config"])
        state = torch.load(work / "checkpoints" / config["run_id"] / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        cache = {}
        for split in ("train", "validation"):
            data = Sequences(SOURCE / "data/processed/ASSIST2009/interactions.csv", split, 100)
            y, p, _ = infer(model, DataLoader(data, batch_size=64))
            cache[split + "_y"], cache[split + "_p"] = y, p
    x = {}
    for split in ("train", "validation"):
        ds = QuestionDifficultySequences(SOURCE / "data/processed/ASSIST2009/interactions.csv", split, 100,
                                         work / "data/auxiliary/ASSIST2009-question-difficulty.json")
        y = np.concatenate([item[3][item[-1]].numpy() for item in ds])
        assert np.array_equal(y, cache[split + "_y"])
        feature = np.concatenate([item[-2][item[-1]].numpy() for item in ds])
        x[split] = np.column_stack([logit(cache[split + "_p"]), feature])
    probe = LogisticRegression(C=1e4, max_iter=1000).fit(x["train"], cache["train_y"])
    prediction = probe.predict_proba(x["validation"])[:, 1]
    result = {"validation": metrics(cache["validation_y"], prediction),
              "coefficients": probe.coef_.tolist(), "intercept": probe.intercept_.tolist(),
              "role": "Exploratory frozen-checkpoint diagnostic, not a completed neural training run."}
    assert abs(result["validation"]["auc"] - .7802025397064125) < 2e-6
    (PROJECT / "reports/diagnostics_20261028/probe_reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: reproduced question probe AUC {result['validation']['auc']:.9f}; training labels only for fit.")


if __name__ == "__main__":
    main()

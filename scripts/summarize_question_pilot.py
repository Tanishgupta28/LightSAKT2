"""Post-replay paired validation diagnostics and exportable research figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from scripts.architecture_pilot import PROJECT, SOURCE, QUESTION_PILOT
from scripts.diagnose_lightsakt_20261028 import metadata, grouped


def main():
    out = PROJECT / QUESTION_PILOT
    baseline = np.load(PROJECT / ".tools/diagnostics_20261028/baseline_train_validation.npz")
    new = np.load(PROJECT / ".tools/diagnostics_20261028/question_validation.npz")
    assert np.array_equal(baseline["validation_y"], new["target"])
    frame = pd.read_csv(SOURCE / "data/processed/ASSIST2009/interactions.csv", dtype={"student": str})
    frame = frame[frame.split.isin(["train", "validation"])]
    counts = frame[frame.split == "train"].groupby("q").label.agg(["size", "sum"])
    meta = metadata(frame, "validation", counts["size"], (counts["sum"] + 1) / (counts["size"] + 2))
    meta["context_bin"] = pd.cut(meta.context_length, [0, 1, 5, 10, 25, 50, 100])
    meta["skill_frequency_bin"] = pd.cut(meta.train_skill_frequency, [-1, 100, 500, 2000, 10000, 1000000])
    for column in ["context_bin", "skill_frequency_bin"]:
        table = pd.DataFrame(grouped(meta, new["prediction"], column))
        table.to_csv(out / f"new_validation_{column}.csv", index=False)
    indices = [np.asarray(x) for x in meta.groupby("student").groups.values()]
    rng = np.random.default_rng(42); deltas = []
    for _ in range(500):
        idx = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        deltas.append(roc_auc_score(new["target"][idx], new["prediction"][idx]) -
                      roc_auc_score(new["target"][idx], baseline["validation_p"][idx]))
    (out / "paired_validation_diagnostic.json").write_text(json.dumps({
        "student_bootstrap_replicates": 500,
        "auc_delta_interval_2_5_97_5": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "limitation": "Conditional on selected models and reused validation split; not independent confirmation."}, indent=2) + "\n")
    curves = pd.read_csv(PROJECT / "reports/diagnostics_20261028/architecture_curves.csv")
    new_curve = pd.read_csv(out / "one_layer_question_difficulty_epochs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, group in curves.groupby("arm", sort=False):
        axes[0].plot(group.epoch, group.validation_auc, label=name.replace("one_layer_", ""), alpha=.65)
    axes[0].plot(new_curve.epoch, new_curve.validation_auc, label="question difficulty (interrupted)", linewidth=2.5)
    axes[0].axhline(.7508035371114405, color="black", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Epoch", ylabel="Validation AUC", title="ASSIST2009, seed 42 (test untouched)")
    axes[0].legend(fontsize=7)
    original = curves[curves.arm == "baseline"]
    axes[1].plot(original.epoch, original.train_bce_online, label="original train, online", linestyle="--")
    axes[1].plot(original.epoch, original.validation_bce, label="original validation")
    axes[1].plot(new_curve.epoch, new_curve.validation_bce, label="question validation")
    axes[1].set(xlabel="Epoch", ylabel="BCE", title="Frozen optimizer and stopping protocol")
    axes[1].legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"comparison_curves.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

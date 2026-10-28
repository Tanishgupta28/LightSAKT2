"""CPU-only train/validation diagnostics. Never constructs a test dataset."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from scripts.architecture_pilot import PROJECT, SOURCE, prepare_gated, summarize
from scripts.architecture_study import digest, read_json
from src.datasets.sequences import Sequences
from src.evaluation.metrics import metrics
from src.models.factory import build_model


OUT = PROJECT / "reports/diagnostics_20261028"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metadata(frame, split, frequency, prior):
    rows = []
    for _, group in frame[frame.split == split].groupby("student", sort=True):
        for start in range(0, len(group) - 1, 100):
            chunk = group.iloc[start:start + 101]
            q = chunk.q.to_numpy(); r = chunk.label.to_numpy()
            n = len(chunk) - 1
            match = (q[:-1][None, :] == q[1:][:, None]) & np.tri(n, dtype=bool)
            success = match @ r[:-1]
            count = match.sum(1)
            part = chunk.iloc[1:].copy()
            part["context_length"] = np.arange(n) + 1
            part["student_history_length"] = start + np.arange(n) + 1
            part["same_skill_count"] = count
            part["same_skill_success"] = success
            part["same_skill_failure"] = count - success
            part["last_response"] = r[:-1]
            part["same_last_skill"] = q[:-1] == q[1:]
            part["train_skill_frequency"] = part.q.map(frequency).fillna(0)
            part["train_skill_prior"] = part.q.map(prior).fillna(prior.mean())
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


@torch.no_grad()
def infer(model, loader, intervention=None, attention=False):
    predictions = []; targets = []; attention_rows = []
    hook = None
    if intervention == "zero_attention":
        hook = model.attn.register_forward_hook(lambda m, args, out: (torch.zeros_like(out[0]), out[1]))
    rng = torch.Generator().manual_seed(991)
    model.eval()
    try:
        for q, r, qry, y, valid in loader:
            if intervention == "flip_responses":
                r = 1 - r
            if intervention == "shuffle_responses":
                r = r.clone()
                for b in range(len(q)):
                    n = int(valid[b].sum())
                    r[b, :n] = r[b, torch.randperm(n, generator=rng)]
            p, weights = model(q, r, qry, valid, attention)
            predictions.append(p[valid].numpy()); targets.append(y[valid].numpy())
            if attention:
                length = q.shape[1]
                positions = torch.arange(length)
                distance = positions[:, None] - positions[None, :]
                allowed = (distance >= 0)[None, None] & valid[:, None, :, None] & valid[:, None, None, :]
                same = (qry[:, :, None] == q[:, None, :])[:, None] & allowed
                recent = ((distance >= 0) & (distance < 5))[None, None]
                w = weights.clamp_min(1e-12)
                values = {
                    "same_skill_mass": (weights * same).sum(-1).mean(1),
                    "uniform_same_skill_mass": same.sum(-1).float().mean(1) / allowed.sum(-1).float().mean(1).clamp_min(1),
                    "recent5_mass": (weights * recent).sum(-1).mean(1),
                    "effective_keys": (1 / weights.square().sum(-1).clamp_min(1e-12)).mean(1),
                    "entropy": -(weights * w.log()).sum(-1).mean(1),
                }
                attention_rows.append(pd.DataFrame({k: x[valid].numpy() for k, x in values.items()}))
    finally:
        if hook is not None:
            hook.remove()
    return np.concatenate(targets), np.concatenate(predictions), (pd.concat(attention_rows, ignore_index=True) if attention else None)


def grouped(meta, prediction, column):
    rows = []
    for value, indexes in meta.groupby(column, observed=True).groups.items():
        idx = np.asarray(indexes)
        result = metrics(meta.label.to_numpy()[idx], prediction[idx])
        result.update(group=str(value), positive_rate=float(meta.label.iloc[idx].mean()),
                      error_rate=float(np.mean((prediction[idx] >= .5) != meta.label.to_numpy()[idx])))
        rows.append(result)
    return rows


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    record = prepare_gated(SOURCE, PROJECT, persist=False)
    comparison = summarize(PROJECT / "pilot-workspace", record, PROJECT, Path("reports/diagnostics_20261028"))
    assert all(r["status"] == "completed" for r in comparison["rows"])
    config = record["configurations"]["baseline"]
    checkpoint = PROJECT / "pilot-workspace/checkpoints" / config["run_id"] / "best.pt"
    expected_hash = comparison["rows"][0]["checkpoint_sha256"]
    assert digest(checkpoint) == expected_hash
    stats = read_json(SOURCE / "reports/ASSIST2009_statistics.json")
    model = build_model("LightSAKT", stats["num_q"], config["model_config"]).cpu()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    path = SOURCE / "data/processed/ASSIST2009/interactions.csv"
    frame = pd.read_csv(path, dtype={"student": str})
    frame = frame[frame.split.isin(["train", "validation"])].copy()
    train = frame[frame.split == "train"]
    counts = train.groupby("q").label.agg(["size", "sum"])
    frequency = counts["size"]; prior = (counts["sum"] + 1) / (counts["size"] + 2)
    # Recover problem IDs for existing train/validation attempts only; never use raw labels.
    raw = pd.read_csv(SOURCE / "data/raw/ASSIST2009.csv", encoding="latin1",
                      usecols=["user_id", "order_id", "problem_id"], dtype={"user_id": str})
    raw = raw.rename(columns={"user_id": "student", "order_id": "order"})
    raw = raw.merge(frame[["student", "order"]], on=["student", "order"], how="inner", validate="many_to_one")
    assert raw.groupby(["student", "order"]).problem_id.nunique().max() == 1
    raw = raw.drop_duplicates(["student", "order"])
    frame = frame.merge(raw, on=["student", "order"], how="left", validate="one_to_one")
    assert frame.problem_id.notna().all()
    problem_frequency = frame[frame.split == "train"].problem_id.value_counts()
    metas = {}; outputs = {}; result = {"checkpoint_sha256": expected_hash, "test_evaluation": False}
    for split in ("train", "validation"):
        meta = metadata(frame, split, frequency, prior)
        loader = DataLoader(Sequences(path, split, 100), batch_size=64, shuffle=False)
        y, p, attn = infer(model, loader, attention=split == "validation")
        assert np.array_equal(y, meta.label.to_numpy())
        meta["train_problem_frequency"] = meta.problem_id.map(problem_frequency).fillna(0)
        meta["context_bin"] = pd.cut(meta.context_length, [0, 1, 5, 10, 25, 50, 100])
        meta["history_bin"] = pd.cut(meta.student_history_length, [0, 5, 25, 50, 100, 200, 100000])
        meta["skill_frequency_bin"] = pd.cut(meta.train_skill_frequency, [-1, 0, 100, 500, 2000, 10000, 1000000])
        meta["problem_frequency_bin"] = pd.cut(meta.train_problem_frequency, [-1, 0, 5, 20, 100, 1000000])
        meta["same_skill_count_bin"] = pd.cut(meta.same_skill_count, [-1, 0, 2, 5, 10, 100])
        metas[split] = meta; outputs[split] = p
        result[split] = {"metrics": metrics(y, p), "positive_rate": float(y.mean()),
                         "uncertain_fraction_04_06": float(np.mean((p >= .4) & (p <= .6))),
                         "confident_wrong_fraction_09": float(np.mean(((p >= .9) & (y == 0)) | ((p <= .1) & (y == 1)))),
                         "negative_recall": float(np.mean(p[y == 0] < .5)),
                         "positive_recall": float(np.mean(p[y == 1] >= .5))}
        for column in ("context_bin", "history_bin", "skill_frequency_bin", "problem_frequency_bin", "same_skill_count_bin", "same_last_skill"):
            pd.DataFrame(grouped(meta, p, column)).to_csv(OUT / f"{split}_{column}.csv", index=False)
        if attn is not None:
            attn["context_bin"] = meta.context_bin
            attn.groupby("context_bin", observed=True).mean().to_csv(OUT / "attention_by_context.csv")
            result["attention_means"] = {k: float(attn[k].mean()) for k in attn if k != "context_bin"}
            skill_rows = []
            for skill, idx in meta.groupby("q").groups.items():
                idx = np.asarray(idx)
                if len(idx) >= 50:
                    m = metrics(y[idx], p[idx]); m.update(q=int(skill), skill=meta.skill.iloc[idx[0]],
                                                        positive_rate=float(y[idx].mean()), train_frequency=int(frequency.get(skill, 0)))
                    skill_rows.append(m)
            pd.DataFrame(skill_rows).sort_values("bce", ascending=False).to_csv(OUT / "validation_skill_errors.csv", index=False)
            for intervention in ("zero_attention", "flip_responses", "shuffle_responses"):
                _, changed, _ = infer(model, loader, intervention)
                result.setdefault("checkpoint_interventions_OOD_not_retrained", {})[intervention] = {
                    **metrics(y, changed), "mean_absolute_prediction_change": float(np.mean(abs(changed - p)))}
        print(f"{split}: {result[split]}", flush=True)
    # Training-fitted low-dimensional probes; validation is evaluation only.
    probes = {}
    for name in ("skill_prior", "pfa_counts", "baseline_calibration", "baseline_plus_counts"):
        features = {}
        for split, meta in metas.items():
            first = logit(meta.train_skill_prior.to_numpy()) if name in {"skill_prior", "pfa_counts"} else logit(outputs[split])
            columns = [first]
            if name in {"pfa_counts", "baseline_plus_counts"}:
                columns += [np.log1p(meta.same_skill_success.to_numpy()), np.log1p(meta.same_skill_failure.to_numpy())]
            features[split] = np.stack(columns, axis=1)
        probe = LogisticRegression(C=1e4, max_iter=1000, random_state=42).fit(features["train"], metas["train"].label)
        pred = probe.predict_proba(features["validation"])[:, 1]
        probes[name] = {"validation": metrics(metas["validation"].label, pred),
                        "coefficients": probe.coef_.tolist(), "intercept": probe.intercept_.tolist(),
                        "limitation": "Frozen-checkpoint train logits are in-sample; exploratory probe is not independent confirmation."}
    result["training_fitted_probes"] = probes
    candidates = read_json(PROJECT / "experiments/selections/ASSIST2009-LightSAKT.json")["all_candidates"]
    grid = []
    for run_id in candidates:
        cfg = read_json(SOURCE / "results/raw" / run_id / "config.json")
        m = read_json(SOURCE / "results/raw" / run_id / "metrics.json")
        grid.append({"run_id": run_id, **cfg["model_config"], "learning_rate": cfg["learning_rate"],
                     "auc": m["validation"]["auc"], "epochs": m["epochs_trained"], "best_epoch": m["best_epoch"]})
    grid = pd.DataFrame(grid); grid.sort_values("auc", ascending=False).to_csv(OUT / "existing_grid.csv", index=False)
    result["grid_marginals_descriptive_confounded"] = {key: grid.groupby(key).auc.agg(["count", "mean", "max"]).reset_index().to_dict("records")
                                                     for key in ("embed_dim", "num_heads", "dropout", "learning_rate")}
    curve_rows = []
    for arm, cfg in record["configurations"].items():
        directory = PROJECT / "pilot-workspace/results/raw" / cfg["run_id"]
        h = read_json(directory / "history.json")
        for row in h:
            curve_rows.append({"arm": arm, "epoch": row["epoch"], "train_bce_online": row["train_objective"],
                               "validation_auc": row["validation"]["auc"], "validation_bce": row["validation"]["bce"]})
    pd.DataFrame(curve_rows).to_csv(OUT / "architecture_curves.csv", index=False)
    # Joint interaction embeddings: are response directions shared across skills?
    weights = model.M.weight.detach().numpy(); k = stats["num_q"]
    differences = weights[k:] - weights[:k]; common = differences[1:].mean(0)
    cosine = differences[1:] @ common / (np.linalg.norm(differences[1:], axis=1) * np.linalg.norm(common) + 1e-12)
    result["joint_response_embedding_geometry"] = {"mean_cosine_to_mean_difference": float(cosine.mean()),
                                                   "median_cosine": float(np.median(cosine)),
                                                   "limitation": "Embedding geometry is not a performance endpoint or proof of identifiability."}
    result["data_limitations"] = "Model q is composite skill, not problem_id. Problem-frequency metadata reconstructed from permitted train/validation keys only, without raw correctness. Context resets every100 targets. No test dataset constructed or predictions used."
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"probes": probes, "attention": result["attention_means"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

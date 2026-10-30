"""Read-only scientific and recovery-state audit; never launches training."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from scripts.architecture_study import read_json, digest
from scripts.architecture_pilot import PROJECT, SOURCE, write_json
from scripts.professor_data_20261030 import question_frame, heldout_difficulty
from scripts.question_difficulty_inputs import difficulty_features
from src.datasets.sequences import Sequences

OUT = PROJECT / "reports/professor_results_20261030"


def leakage_audit(plan):
    findings = []
    for dataset, spec in plan["feature_specs"].items():
        frame, semantics = question_frame(SOURCE, dataset)
        groups = {s: set(frame.loc[frame.split == s, "student"]) for s in ["train", "validation", "test"]}
        assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
        tv = frame[frame.split.isin(["train", "validation"])].copy()
        original = difficulty_features(tv)
        changed = tv.copy()
        validation_mask = changed.split == "validation"
        changed.loc[validation_mask, "label"] = 1 - changed.loc[validation_mask, "label"]
        np.testing.assert_allclose(difficulty_features(changed), original, rtol=0, atol=1e-12)
        student = tv.loc[tv.split == "train"].student.value_counts().index[0]
        own = (tv.student == student).to_numpy()
        changed = tv.copy()
        changed.loc[own, "label"] = 1 - changed.loc[own, "label"]
        np.testing.assert_allclose(difficulty_features(changed)[own], original[own], rtol=0, atol=1e-12)
        changed = tv.copy()
        changed.loc[(changed.split == "train") & ~own, "label"] = 1 - changed.loc[(changed.split == "train") & ~own, "label"]
        control_change = float(np.max(np.abs(difficulty_features(changed)[own] - original[own])))
        assert control_change > 0  # Perturbation test has a responsive positive control.
        training = frame[frame.split == "train"]
        for split, key in [("validation", "train_validation"), ("test", "test")]:
            rows = frame[frame.split == split]
            values = heldout_difficulty(training, rows[["q", "problem_id"]])
            frozen = read_json(spec[key + "_feature_path"])["students"]
            for identity, g in rows.assign(feature=values).groupby("student", sort=True):
                np.testing.assert_allclose(g.feature, frozen[identity]["values"], rtol=0, atol=1e-12)
        findings.append({"dataset": dataset, "question_identity": semantics,
                         "training_question_aggregates": int(training.problem_id.nunique()),
                         "training_skill_aggregates": int(training.q.nunique()),
                         "unseen_question_fraction": {split: float((~frame.loc[frame.split == split, "problem_id"].isin(set(training.problem_id))).mean())
                             for split in ["validation", "test"]},
                         "student_disjoint": True, "validation_label_flip_invariant": True,
                         "entire_own_student_label_flip_invariant": True, "own_student_test_rows": int(own.sum()),
                         "other_training_students_positive_control_maximum_change": control_change,
                         "validation_and_test_sidecars_reproduced_from_train_labels_only": True,
                         "scope": "Offline full-training-corpus fitting; no chronological deployment claim"})
    write_json(OUT / "real_data_leakage_audit.json", {"status": "PASS", "datasets": findings})


def checkpoint_audit(plan):
    results = []
    epoch_rows = []
    for entry in plan["entries"]:
        work = Path(entry["work"]); run = entry["run_id"]
        raw = work / "results/raw" / run
        if not (raw / "metrics.json").exists() or read_json(raw / "status.json")["status"] != "completed":
            continue
        config = read_json(raw / "config.json"); metrics = read_json(raw / "metrics.json")
        history = read_json(raw / "history.json")
        best_path = work / "checkpoints" / run / "best.pt"
        latest_path = work / "checkpoints" / run / "latest.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert config == latest["config"] == best["config"] == read_json(entry["config"])
        assert latest["history"] == history
        assert [h["epoch"] for h in history] == list(range(1, latest["epoch"] + 1))
        selected = max(history, key=lambda h: h["validation"]["auc"])
        assert selected["epoch"] == metrics["best_epoch"] == best["epoch"] == latest["best_epoch"]
        assert selected["validation"] == metrics["validation"]
        assert latest["epoch"] == metrics["epochs_trained"] <= config["max_epochs"]
        assert latest["epoch"] == config["max_epochs"] or latest["bad_epochs"] == config["patience"]
        assert metrics["test"] is None and config["evaluate_test"] is False
        assert digest(best_path) == metrics["checkpoint_sha256"]
        for state in [best, latest]:
            assert all(torch.isfinite(v).all() for v in state["model"].values())
        for state in latest["optimizer"]["state"].values():
            assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
        rng = latest["rng"]
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator().set_state(rng["torch"].cpu())
        torch.Generator().set_state(latest["loader_rng"].cpu())
        assert len(rng["cuda"]) == 1 and rng["cuda"][0].dtype == torch.uint8
        dataset = Sequences(SOURCE / f"data/processed/{entry['dataset']}/interactions.csv", "train", config["seq_len"])
        generator = torch.Generator().manual_seed(config["seed"])
        loader = DataLoader(range(len(dataset)), batch_size=config["batch_size"], shuffle=True, num_workers=0, generator=generator)
        for _ in history:
            for _ in loader:
                pass
        assert torch.equal(generator.get_state(), latest["loader_rng"].cpu())
        expected_steps = len(history) * math.ceil(len(dataset) / config["batch_size"])
        steps = sorted(set(int(v["step"]) for v in latest["optimizer"]["state"].values()))
        assert steps == [expected_steps]
        assert metrics["peak_allocated_bytes"] <= 4 * 1024**3
        spec = plan["feature_specs"][entry["dataset"]]
        assert digest(spec["train_validation_feature_path"]) == spec["train_validation_feature_sha256"]
        for h in history:
            row = {"dataset": entry["dataset"], "seed": entry["seed"], "run_id": run, "epoch": h["epoch"],
                   "epoch_seconds": h["epoch_seconds"], "online_training_bce": h["train_objective"]}
            row.update({"validation_" + k: v for k, v in h["validation"].items() if k != "auc_reason"})
            resources = h.get("resources", {})
            row.update({k: v for k, v in resources.items() if isinstance(v, (int, float))})
            row.update({"online_training_" + k: v for k, v in resources.get("online_training_metrics", {}).items() if k != "auc_reason"})
            epoch_rows.append(row)
        results.append({"run_id": run, "status": "PASS", "epochs": len(history), "best_epoch": best["epoch"],
                        "best_sha256": digest(best_path), "latest_sha256": digest(latest_path),
                        "finite_model_and_optimizer": True, "saved_rng_states_valid": True,
                        "data_order_rng_matches_exact_seeded_epoch_stream": True, "adam_steps": expected_steps,
                        "training_test_not_constructed": True,
                        "difficulty_scale": float(best["model"]["difficulty_scale"]),
                        "protected_training_code_sha256": metrics["training_code_sha256"]})
    pd.DataFrame(epoch_rows).to_csv(OUT / "difficulty_epoch_results.csv", index=False)
    write_json(OUT / "checkpoint_state_audit.json", {"status": "PASS", "completed_runs": results})


def metrics_audit():
    expected = {}
    for dataset in ["ASSIST2009", "ASSIST2015", "Algebra2005"]:
        frame = pd.read_csv(SOURCE / f"data/processed/{dataset}/interactions.csv", dtype={"student": str})
        for split in ["validation", "test"]:
            expected[dataset, split] = np.concatenate([g.label.to_numpy()[1:]
                for _, g in frame[frame.split == split].groupby("student", sort=True)])
    results = []
    for path in sorted((OUT / "replays").glob("*.json")):
        record = read_json(path)
        for split in ["validation", "test"]:
            saved = record["prediction_artifacts"][split]
            assert digest(Path(saved["path"])) == saved["sha256"]
            with np.load(saved["path"]) as arrays:
                y = arrays["target"].astype(np.float64)
                p = arrays["prediction"].astype(np.float64)
            assert np.array_equal(y, expected[record["dataset"], split])
            assert y.ndim == p.ndim == 1 and y.shape == p.shape
            assert np.isfinite(p).all() and np.isin(y, [0, 1]).all() and ((p >= 0) & (p <= 1)).all()
            truth, predicted = y == 1, p >= .5
            tp = np.count_nonzero(truth & predicted); fp = np.count_nonzero(~truth & predicted)
            fn = np.count_nonzero(truth & ~predicted); tn = np.count_nonzero(~truth & ~predicted)
            positive = int(truth.sum()); negative = len(y) - positive
            # Independent Mann–Whitney rank-sum AUC, including average tied ranks.
            auc = (rankdata(p)[truth].sum() - positive * (positive + 1) / 2) / (positive * negative)
            clipped = np.clip(p, 1e-7, 1 - 1e-7)
            actual = {"auc": float(auc), "accuracy": (tp + tn) / len(y),
                      "precision": tp / (tp + fp) if tp + fp else 0.,
                      "recall": tp / (tp + fn) if tp + fn else 0.,
                      "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                      "bce": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log(1 - clipped)))}
            reference = record["cpu_replay"][split]
            differences = {key: abs(value - reference[key]) for key, value in actual.items()}
            assert max(differences.values()) < 1e-12
            results.append({"dataset": record["dataset"], "model": record["model"], "seed": record["seed"],
                            "split": split, "targets": len(y), "positive_targets": positive,
                            "every_expected_n_minus_one_target_in_order": True,
                            "independent_metric_maximum_difference": max(differences.values())})
    write_json(OUT / "masking_and_metrics_audit.json", {"status": "PASS", "split_replays": results,
        "auc_method": "Independent Mann-Whitney rank-sum with tied ranks", "classification": "Manual confusion counts, threshold0.5",
        "mask_check": "Exact array equality to per-student original labels[1:], no padding, duplication or dropped targets"})


def resource_summary(plan):
    samples = [json.loads(line) for line in (OUT / "campaign_resource_samples.jsonl").read_text().splitlines()]
    campaign = read_json(OUT / "campaign_status.json")
    completed = []
    for entry in plan["entries"]:
        if entry["action"] == "reuse_completed":
            continue
        path = Path(entry["work"]) / "results/raw" / entry["run_id"] / "metrics.json"
        if path.exists():
            m = read_json(path)
            assert m["training_seconds"] < 900 and m["peak_allocated_bytes"] <= 4 * 1024**3
            completed.append({"run_id": entry["run_id"], "training_seconds": m["training_seconds"],
                              "peak_allocated_bytes": m["peak_allocated_bytes"], "peak_reserved_bytes": m["peak_reserved_bytes"]})
    assert all(row["gpu_uuid"] == plan["gpu_uuid"] for row in samples)
    consumed = campaign.get("gpu_window_seconds_consumed", 0)
    assert consumed <= plan["maximum_total_gpu_seconds"]
    breaches = [i for i, r in enumerate(samples) if r["free_bytes"] < 6 * 1024**3 or
                r["host_headroom_bytes"] < 2 * 1024**3 or r["process_peak_rss_bytes"] > 4 * 1024**3]
    if campaign["status"] == "completed":
        assert not breaches
    write_json(OUT / "resource_summary.json", {"campaign_status": campaign["status"], "completed_new_runs": completed,
        "resource_samples": len(samples), "gpu_uuid": plan["gpu_uuid"], "gpu_window_wall_seconds": consumed,
        "new_training_seconds_sum": sum(r["training_seconds"] for r in completed),
        "minimum_shared_gpu_free_bytes": min(r["free_bytes"] for r in samples),
        "minimum_host_headroom_bytes": min(r["host_headroom_bytes"] for r in samples),
        "maximum_worker_rss_bytes": max(r["process_peak_rss_bytes"] for r in samples),
        "maximum_allocator_peak_bytes": max(r["peak_allocated_bytes"] for r in completed),
        "runtime_floor_or_rss_breach_sample_indices": breaches,
        "timing_scope": "Admitted GPU-window wall time including pacing; not kernel-active profiling or exclusive GPU time"})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--leakage", action="store_true"); args = parser.parse_args()
    torch.set_num_threads(1)
    plan = read_json(PROJECT / "experiments/professor_campaign_20261030/plan.json")
    for path, expected in plan["protected_sha256"].items():
        assert digest(Path(path)) == expected, path
    assert digest(PROJECT / "scripts/professor_data_20261030.py") == plan["training_adapter_sha256"]
    assert digest(PROJECT / "scripts/run_professor_campaign_20261030.py") == plan["training_launcher_sha256"]
    write_json(OUT / "protected_artifacts_audit.json", {"status": "PASS", "unchanged_files": len(plan["protected_sha256"]),
                "training_adapter_and_launcher_unchanged": True})
    if args.leakage:
        leakage_audit(plan)
    checkpoint_audit(plan)
    metrics_audit()
    resource_summary(plan)
    print("PASS protected artifacts, complete checkpoints, optimizer/RNG/data order, independent masks/metrics" + (", real-data label perturbations" if args.leakage else ""))


if __name__ == "__main__":
    main()

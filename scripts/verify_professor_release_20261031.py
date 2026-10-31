"""CPU-only re-verification of frozen results; original artifacts are read-only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from src.models.factory import build_model
from scripts import audit_professor_campaign_20261030 as original_audit

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "LightSAKT2"
OLD = PROJECT / "reports/professor_results_20261030"
OUT = PROJECT / "reports/professor_release_20261031"
WORK = PROJECT / ".tools/professor_release_20261031/audit"
SEEDS = [42, 123, 2026, 3407, 7777]
DATASETS = ["ASSIST2009", "ASSIST2015", "Algebra2005"]
MODELS = ["SAKT", "LightSAKT", "LightSAKT+Difficulty"]
METRICS = ["auc", "accuracy", "f1", "precision", "recall", "bce"]
SCIENTIFIC_KEYS = ["dataset", "seed", "seq_len", "batch_size", "learning_rate", "betas", "weight_decay",
                   "epsilon", "max_epochs", "patience", "model_config"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    return {str(p.relative_to(OLD)): sha(p) for p in OLD.rglob("*") if p.is_file()}


def close(a, b, tolerance=1e-12):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    before = read(OUT / "original_report_fingerprints.json")
    assert before == inventory(), "Original reports changed since recovery"
    parsed = {"json": 0, "csv": 0, "jsonl": 0, "xml": 0, "text": 0, "binary_figures": 0}
    for path in OLD.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            read(path); parsed["json"] += 1
        elif path.suffix == ".csv":
            pd.read_csv(path); parsed["csv"] += 1
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                json.loads(line)
            parsed["jsonl"] += 1
        elif path.suffix == ".xml":
            ET.parse(path); parsed["xml"] += 1
        elif path.suffix in {".md", ".log"}:
            path.read_text(); parsed["text"] += 1
        else:
            parsed["binary_figures"] += 1

    plan = read(PROJECT / "experiments/professor_campaign_20261030/plan.json")
    for path, expected in plan["protected_sha256"].items():
        assert sha(path) == expected, path
    assert sha(PROJECT / "scripts/professor_data_20261030.py") == plan["training_adapter_sha256"]
    assert sha(PROJECT / "scripts/run_professor_campaign_20261030.py") == plan["training_launcher_sha256"]

    # Existing audit functions write only into an ignored, new temporary root.
    # Their original frozen replay/resource inputs are exposed read-only by links.
    WORK.mkdir(parents=True, exist_ok=True)
    for name in ["replays", "campaign_status.json", "campaign_resource_samples.jsonl"]:
        target = WORK / name
        if target.is_symlink():
            assert target.resolve() == OLD / name
        else:
            assert not target.exists(); target.symlink_to(OLD / name)
    original_audit.OUT = WORK
    original_audit.leakage_audit(plan)
    original_audit.checkpoint_audit(plan)
    original_audit.metrics_audit()
    original_audit.resource_summary(plan)
    for name in ["real_data_leakage_audit.json", "checkpoint_state_audit.json", "masking_and_metrics_audit.json",
                 "resource_summary.json", "difficulty_epoch_results.csv"]:
        shutil.copy2(WORK / name, OUT / name)

    records = [read(path) for path in sorted((OLD / "replays").glob("*.json"))]
    assert len(records) == 40
    lookup = {(r["dataset"], r["model"], r["seed"]): r for r in records}
    assert len(lookup) == 40
    old_individual = {(r["dataset"], r["model"], r["seed"]): r for r in read(OLD / "individual_seed_results.json")}
    references = []; verification = []; actual = []
    configurations = OUT / "configurations"
    for dataset in DATASETS:
        stats = read(SOURCE / f"reports/{dataset}_statistics.json")
        data_path = SOURCE / f"data/processed/{dataset}/interactions.csv"
        split_path = SOURCE / f"data/splits/{dataset}.json"
        assert sha(data_path) == stats["processed_sha256"] and sha(split_path) == stats["split_sha256"]
        frame = pd.read_csv(data_path, dtype={"student": str})
        students = {s: set(frame.loc[frame.split == s, "student"]) for s in ["train", "validation", "test"]}
        assert not (students["train"] & students["validation"] or students["train"] & students["test"] or students["validation"] & students["test"])
        for model in MODELS:
            for seed in SEEDS:
                key = dataset, model, seed
                if key not in lookup:
                    assert dataset == "ASSIST2015" and model == "LightSAKT+Difficulty"
                    actual.append({"dataset": dataset, "model": model, "seed": seed, "status": "UNAVAILABLE_CURRENT_PROTOCOL",
                                   "reason": "No independent question identifier or authentic log-to-question crosswalk in inspected sources"})
                    continue
                r = lookup[key]
                assert r["status"] == "VERIFIED_CPU_REPLAY"
                checkpoint = Path(r["checkpoint"])
                assert sha(checkpoint) == r["checkpoint_sha256"]
                assert sha(r["recorded_metrics_path"]) == r["recorded_metrics_sha256"]
                assert sha(r["config_file"]) == r["config_sha256"]
                config = read(r["config_file"])
                assert config == r["config"]
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                for name in SCIENTIFIC_KEYS + ["model"]:
                    assert state["config"][name] == config[name], (key, name)
                assert state["processed_sha256"] == r["processed_sha256"] == stats["processed_sha256"]
                assert state["split_sha256"] == r["split_sha256"] == stats["split_sha256"]
                assert all(torch.isfinite(v).all() for v in state["model"].values())
                native = read(r["recorded_metrics_path"])
                assert native["validation"] == r["validation"] and native["best_epoch"] == state["epoch"] == r["best_epoch"]
                # Seed42 baseline aliases refer to their selected grid checkpoint.
                history = read(checkpoint.parent.parent.parent / "results/raw" / checkpoint.parent.name / "history.json")
                assert max(history, key=lambda x: x["validation"]["auc"])["epoch"] == state["epoch"]
                network = build_model(config["model"], stats["num_q"], config["model_config"]).cpu()
                network.load_state_dict(state["model"], strict=True)
                assert sum(p.numel() for p in network.parameters()) == r["parameters"]
                assert config["max_epochs"] == 50 and config["patience"] == 5 and config["batch_size"] == 64
                assert config["betas"] == [.9, .999] and config["epsilon"] == 1e-8 and config["weight_decay"] == 0
                if model == "LightSAKT+Difficulty":
                    base = lookup[dataset, "LightSAKT", seed]
                    for name in SCIENTIFIC_KEYS:
                        assert config[name] == base["config"][name], (key, name)
                    assert config["evaluate_test"] is False and native["test"] is None
                    assert r["parameters"] == base["parameters"] + 1
                for split in ["validation", "test"]:
                    for metric in METRICS:
                        cpu = r["cpu_replay"][split][metric]
                        if split == "test":
                            close(r["test"][metric], cpu)
                        elif metric in r["validation"]:
                            close(r["validation"][metric], cpu, 2e-6 if metric in ["auc", "bce"] else 1e-4)
                item = {"dataset": dataset, "model": model, "seed": seed, "status": "VERIFIED"}
                for split in ["validation", "test"]:
                    for metric in METRICS:
                        value = r[split].get(metric, r["cpu_replay"][split][metric])
                        item[f"{split}_{metric}"] = value
                        close(value, old_individual[key][f"{split}_{metric}"])
                for name, source_name in [("parameters", "parameters"), ("best_epoch", "best_epoch"), ("epochs", "epochs_trained"),
                                          ("training_seconds", "training_seconds"), ("checkpoint_sha256", "checkpoint_sha256"), ("run_id", "run_id")]:
                    item[name] = r[source_name]
                item["checkpoint_reference"] = f"{dataset}/{model}/{seed}/best.pt"
                config_relative = Path("configurations") / dataset / model / f"{seed}.json"
                write(OUT / config_relative, config)
                refs = {"dataset": dataset, "model": model, "seed": seed, "checkpoint_reference": item["checkpoint_reference"],
                    "checkpoint_sha256": r["checkpoint_sha256"], "source_local_path": str(checkpoint),
                    "configuration": str(config_relative), "configuration_sha256": sha(OUT / config_relative),
                    "source_configuration_sha256": r["config_sha256"], "selected_epoch": r["best_epoch"], "parameters": r["parameters"],
                    "processed_sha256": stats["processed_sha256"], "split_sha256": stats["split_sha256"],
                    "training_git_commit": r["selected_git_commit"], "weights_in_archive": False,
                    "selection": "maximum validation AUC; strict improvement; maximum50epochs; patience5"}
                references.append(refs); actual.append(item)
                verification.append({"dataset": dataset, "model": model, "seed": seed,
                    "checkpoint_hash_config_state_shape_and_parameter_count": "PASS", "validation_selected_epoch": r["best_epoch"],
                    "baseline_final_evaluate_test_flag": config.get("evaluate_test"),
                    "test_metrics_source": "2026-10-30 CPU replay, batch8, all valid targets pooled",
                    "current_verification": "fresh checkpoint/record/config hashes and saved prediction-array metrics; no new model inference or GPU training"})
                del network, state

    summaries = []; deltas = []; paired = []
    for dataset in DATASETS:
        for model in MODELS:
            rows = [r for r in actual if r["dataset"] == dataset and r["model"] == model and r["status"] == "VERIFIED"]
            group = {"dataset": dataset, "model": model, "verified_seeds": len(rows), "expected_seeds": 5,
                     "status": "COMPLETE" if len(rows) == 5 else "UNAVAILABLE_CURRENT_PROTOCOL"}
            for split in ["validation", "test"]:
                for metric in METRICS:
                    values = [r[f"{split}_{metric}"] for r in rows]
                    group[f"{split}_{metric}_mean"] = mean(values) if values else None
                    group[f"{split}_{metric}_sd"] = stdev(values) if len(values) > 1 else None
            summaries.append(group)
        for comparator in ["LightSAKT", "SAKT"]:
            rows = []
            for seed in SEEDS:
                if (dataset, "LightSAKT+Difficulty", seed) not in lookup:
                    continue
                a, b = lookup[dataset, "LightSAKT+Difficulty", seed], lookup[dataset, comparator, seed]
                row = {"dataset": dataset, "comparator": comparator, "seed": seed}
                for split in ["validation", "test"]:
                    for metric in METRICS:
                        row[f"{split}_{metric}_delta"] = a[split].get(metric, a["cpu_replay"][split][metric]) - b[split].get(metric, b["cpu_replay"][split][metric])
                rows.append(row); paired.append(row)
            group = {"dataset": dataset, "comparator": comparator, "paired_seeds": len(rows),
                     "status": "COMPLETE" if rows else "UNAVAILABLE_CURRENT_PROTOCOL"}
            for split in ["validation", "test"]:
                values = [r[f"{split}_auc_delta"] for r in rows]
                group[f"{split}_auc_delta_mean"] = mean(values) if values else None
                group[f"{split}_auc_delta_sd"] = stdev(values) if len(values) > 1 else None
                group[f"{split}_positive_seed_count"] = sum(v > 0 for v in values)
            deltas.append(group)
    old_summary = {(r["dataset"], r["model"]): r for r in read(OLD / "summary_results.json")}
    for row in summaries:
        ref = old_summary[row["dataset"], row["model"]]
        assert row["verified_seeds"] == ref["verified_seeds"]
        for split in ["validation", "test"]:
            for metric in METRICS:
                for suffix in ["mean", "sd"]:
                    key = f"{split}_{metric}_{suffix}"
                    if row[key] is not None:
                        close(row[key], ref[key])
    for name, value in [("individual_seed_results", actual), ("summary_results", summaries),
                        ("absolute_auc_improvements", deltas), ("paired_seed_deltas", paired), ("checkpoint_references", references)]:
        write(OUT / f"{name}.json", value)
        pd.DataFrame(value).to_csv(OUT / f"{name}.csv", index=False)
    write(OUT / "per_seed_verification.json", verification)
    write(OUT / "checkpoint_provenance.json", {"original_replays": records, "portable_references": references,
        "source_result_commit": "c286bcbcee46a5317dbb3071c1f6c3a66335d04a", "source_handoff_commit": "49982fa7b20c33dc344ef3501b29a03fc35b8bfc"})
    bundles = []
    paths = [SOURCE / "backups/question_completion_20261029/research_results.bundle",
             SOURCE / "backups/professor_results_20261030/research_results-c286bcbc.bundle",
             SOURCE / "backups/professor_results_20261030/research_handoff-49982fa7.bundle"]
    for path in paths:
        verified = subprocess.run(["git", "bundle", "verify", str(path)], cwd=PROJECT, text=True, capture_output=True, check=True)
        bundles.append({"file": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                        "verification": (verified.stdout + verified.stderr).strip(), "status": "PASS"})
    write(OUT / "git_bundle_verification.json", {"status": "PASS", "bundles": bundles,
        "scope": "Incremental chain; preserved baseline a6bdc6c0 prerequisite is required. Not a self-contained full-history backup.",
        "remote_status": "No remote push attempted or claimed; previously recorded CLI authentication blocker remains."})
    assert inventory() == before
    report = {"status": "PASS", "source_files_read_and_unchanged": len(before), "parsed_report_types": parsed,
        "per_seed_results_verified": 40, "unavailable_model_seed_slots": 5, "independent_validation_test_metric_arrays": 80,
        "best_checkpoint_hash_config_parameter_checks": 40, "difficulty_optimizer_rng_data_order_checks": 10,
        "protected_original_files_unchanged": len(plan["protected_sha256"]), "gpu_experiments_executed_this_task": 0,
        "new_model_inference_this_task": 0, "metrics_recalculated_from_saved_predictions": True,
        "sample_standard_deviation_ddof": 1,
        "evaluation_flag_clarification": "Baseline final configs enable post-training test scoring; difficulty training configs disable it and use separate CPU scoring. Best epoch selection is validation-only for all. Published test table uses CPU replay consistently.",
        "current_scope": "Verification, authentic metadata search and packaging; no new training or feature-matched controls."}
    write(OUT / "verification_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

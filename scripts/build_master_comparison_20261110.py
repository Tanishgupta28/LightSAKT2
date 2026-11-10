"""Evidence-backed, CPU-only master comparison. Never launches a trainer.

Completed experiment directories and historical reports are read-only inputs.
Single-seed primary/tuning evidence never populates five-seed aggregate cells.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import pandas as pd
import torch
from src.models.factory import build_model
from scripts.replay_ednet_campaign_20261104 import independent_metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "LightSAKT2"
CORE = ROOT / "reports/professor_three_datasets_20261104"
EDNET = ROOT / ".tools/ednet_replacement_20261104/workspace"
OUT = ROOT / "reports/master_comparison_20261110"
PREP = ROOT / "experiments/master_comparison_20261110"
DS = "EdNetKT1_5k_first"
DATASETS = ["ASSIST2009", "Algebra2005", DS]
MODELS = ["DKT", "DKT+", "DKVMN", "GKT", "GKT-MHA (implemented variant)",
          "SAKT", "LightSAKT", "LightSAKT + Question Difficulty", "SAKT + Question Difficulty"]
SEEDS = [42, 123, 2026, 3407, 7777]
FIELDS = ["validation_auc", "test_auc", "test_accuracy", "test_f1"]
SCIENCE = ["dataset", "model", "seed", "seq_len", "batch_size", "max_epochs", "patience",
           "learning_rate", "betas", "epsilon", "weight_decay", "model_config"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n"); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def model_label(name):
    return {"LightSAKT+Difficulty": "LightSAKT + Question Difficulty",
            "LightSAKTQuestion": "LightSAKT + Question Difficulty",
            "GKT_MHA": "GKT-MHA (implemented variant)", "SAKTQuestion": "SAKT + Question Difficulty"}.get(name, name)


def aggregate(rows):
    for r in rows:
        if r["dataset"] not in DATASETS or r["model"] not in MODELS:
            raise ValueError("Unknown dataset version or model label")
        if any(not np.isfinite(r[k]) or not 0 <= r[k] <= 1 for k in FIELDS):
            raise ValueError("Invalid or unavailable measurement cannot be aggregated")
    identities = [(r["dataset"], r["model"], r["seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate model/dataset/seed; aliases or repeats are not independent seeds")
    result = []
    for dataset in DATASETS:
        for model in MODELS:
            group = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            seeds = {r["seed"] for r in group}
            if not seeds.issubset(SEEDS): raise ValueError("Unexpected training seed")
            complete = seeds == set(SEEDS)
            if len({(r["processed_sha256"], r["split_sha256"]) for r in group}) > 1:
                raise ValueError("Mixed dataset versions or student splits")
            row = {"dataset": dataset, "model": model, "seed_count": len(group), "required_seed_count": 5,
                   "experiment_status": "VERIFIED_5_SEEDS" if complete else "PENDING"}
            for field in FIELDS:
                values = [r[field] for r in group]
                row[field + "_mean"] = mean(values) if complete else None
                row[field + "_std"] = stdev(values) if complete else None
                row[field] = f"{mean(values):.9f} ± {stdev(values):.9f}" if complete else "PENDING"
            row["primary_metric"] = "test_auc"
            row["mean"] = row["test_auc_mean"]
            row["standard_deviation"] = row["test_auc_std"]
            row["missing_verified_seeds"] = [s for s in SEEDS if s not in seeds]
            row["source_references"] = [r["source_record"] for r in group]
            row["checkpoint_references"] = [{"seed": r["seed"], "path": r["checkpoint"], "sha256": r["checkpoint_sha256"]} for r in group]
            row["configuration_references"] = [r["configuration_reference"] for r in group]
            result.append(row)
    return result


def expected_data():
    metadata = {d["dataset"]: d for d in read(CORE / "dataset_metadata.json")}
    expected = {}
    for dataset, stats in metadata.items():
        work = EDNET if dataset == DS else SOURCE
        data = work / f"data/processed/{dataset}/interactions.csv"
        assert sha(data) == stats["processed_sha256"]
        assert sha(work / f"data/splits/{dataset}.json") == stats["split_sha256"]
        frame = pd.read_csv(data, usecols=["student", "q", "label", "split"], dtype={"student": str})
        sets = [set(frame.loc[frame.split == s, "student"]) for s in ["train", "validation", "test"]]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
        for split in ["validation", "test"]:
            expected[dataset, split] = np.concatenate([g.label.to_numpy()[1:] for _, g in frame[frame.split == split].groupby("student", sort=True)])
    return metadata, expected


def checkpoint_check(record, metadata):
    path = Path(record["checkpoint"]); assert sha(path) == record["checkpoint_sha256"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = record["config"]
    for key in SCIENCE: assert state["config"][key] == cfg[key], (path, key)
    assert state["epoch"] == record["best_epoch"]
    stats = metadata[record["dataset"]]
    assert state["processed_sha256"] == record["processed_sha256"] == stats["processed_sha256"]
    assert state["split_sha256"] == record["split_sha256"] == stats["split_sha256"]
    assert all(torch.isfinite(t).all() for t in state["model"].values())
    raw = Path(record["recorded_metrics_path"]).parent
    paths = [raw / "history.json", path.parents[2] / "results/raw" / path.parent.name / "history.json"]
    history_path = next((p for p in paths if p.exists()), None); assert history_path is not None
    history = read(history_path)
    selected = max(history, key=lambda h: h["validation"]["auc"])
    assert selected["epoch"] == state["epoch"] and abs(state["validation"]["auc"] - record["validation"]["auc"]) < 2e-6
    model = build_model(cfg["model"], stats["num_q"], cfg["model_config"])
    model.load_state_dict(state["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == record["parameters"]
    return {"status": "PASS", "history": str(history_path), "strict_load": True,
            "validation_maximum_epoch": state["epoch"], "checkpoint_sha256": sha(path)}


def array_check(path, digest, expected, reference):
    assert sha(path) == digest
    with np.load(path) as arrays:
        y = arrays["target"]
        p = arrays["prediction"] if "prediction" in arrays else arrays["probability"]
    assert np.array_equal(y, expected)
    measured = independent_metrics(y, p)
    differences = {k: abs(measured[k] - reference[k]) for k in ["auc", "accuracy", "f1", "bce"]}
    assert max(differences.values()) < 1e-12, (path, differences)
    return {"status": "PASS", "path": str(path), "sha256": digest, "targets": len(y),
            "independent_metric_max_difference": max(differences.values())}


def load_existing(metadata, expected):
    records = read(CORE / "experiment_provenance.json")["records"]
    sources = {r["run_id"]: str(CORE / "experiment_provenance.json") for r in records}
    cache_path = SOURCE / "reports/selected_final_inference_verification.json"
    cache = read(cache_path)["runs"]
    for dataset in DATASETS[:2]:
        for model in ["DKT", "DKT+", "DKVMN"]:
            for seed in SEEDS:
                raw = SOURCE / "results/raw" / f"{dataset}-{model}-final-{seed}"
                cfg, metrics = read(raw / "config.json"), read(raw / "metrics.json")
                assert read(raw / "status.json")["status"] == "completed"
                matches = [r for r in cache if r["run_id"] == metrics["run_id"] and r.get("verified")
                    and r.get("checkpoint_sha256") == metrics["checkpoint_sha256"]
                    and r.get("prediction_sha256") == metrics["prediction_sha256"]
                    and r.get("metrics_sha256") == sha(raw / "metrics.json")
                    and r.get("config_sha256") == sha(raw / "config.json")]
                assert matches, (dataset, model, seed, "No matching independent CPU replay proof")
                replay = max(matches, key=lambda r: r["verified_utc"])
                assert replay["device"] == "cpu" and replay["locked_test_identity_verified"] and replay["locked_test_labels_verified"]
                checkpoint = Path(metrics["checkpoint"])
                if not checkpoint.is_absolute(): checkpoint = SOURCE / checkpoint
                prediction = Path(metrics["prediction_path"])
                if not prediction.is_absolute(): prediction = SOURCE / prediction
                record = {**metrics, "config": cfg, "checkpoint": str(checkpoint), "recorded_metrics_path": str(raw / "metrics.json"),
                    "config_file": str(raw / "config.json"), "config_sha256": sha(raw / "config.json"),
                    "recorded_metrics_sha256": sha(raw / "metrics.json"), "native_recorded_test": metrics["test"],
                    "test": replay["regenerated_metrics"]["test"], "cpu_replay": replay["regenerated_metrics"],
                    "prediction_artifacts": {"test": {"path": str(prediction), "sha256": metrics["prediction_sha256"]}},
                    "selected_git_commit": metrics["git_commit"], "prior_cpu_replay": replay,
                    "source_record": str(cache_path), "cpu_replay_timestamp": replay["verified_utc"]}
                records.append(record); sources[record["run_id"]] = str(cache_path)
    assert len(records) == 75
    rows, audits = [], []
    for record in records:
        assert sha(record["config_file"]) == record["config_sha256"]
        assert sha(record["recorded_metrics_path"]) == record["recorded_metrics_sha256"]
        state = checkpoint_check(record, metadata)
        arrays = []
        for split, saved in record["prediction_artifacts"].items():
            reference = record["native_recorded_test"] if "native_recorded_test" in record else record["cpu_replay"][split]
            arrays.append(array_check(saved["path"], saved["sha256"], expected[record["dataset"], split], reference))
        for split in ["validation", "test"]:
            for key in ["auc", "accuracy", "f1", "bce"]:
                native = record["native_recorded_test"] if split == "test" and "native_recorded_test" in record else record[split]
                assert abs(record["cpu_replay"][split][key] - native[key]) < 2e-6
        label = model_label(record["model"])
        native_test = record.get("native_recorded_test", read(record["recorded_metrics_path"]).get("test"))
        config_copy = OUT / "configurations" / f"{record['dataset']}-{record['model']}-{record['seed']}.json"
        write(config_copy, record["config"])
        row = {"dataset": record["dataset"], "model": label, "implementation_model_key": record["config"]["model"],
            "seed": record["seed"], "experiment_status": "VERIFIED", "validation_auc": record["validation"]["auc"],
            "test_auc": record["test"]["auc"], "test_accuracy": record["test"]["accuracy"], "test_f1": record["test"]["f1"],
            "native_recorded_test_auc": native_test["auc"] if native_test is not None else None,
            "parameters": record["parameters"], "best_epoch": record["best_epoch"], "epochs_trained": record["epochs_trained"],
            "training_seconds": record["training_seconds"], "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"], "source_record": sources[record["run_id"]],
            "run_id": record["run_id"], "configuration_reference": str(config_copy.relative_to(ROOT)),
            "configuration_sha256": sha(config_copy),
            "processed_sha256": record["processed_sha256"], "split_sha256": record["split_sha256"],
            "test_metric_origin": "existing independently verified CPU replay", "validation_metric_origin": "native validation-selected epoch",
            "training_commit": record.get("selected_git_commit", record.get("git_commit"))}
        rows.append(row); audits.append({"dataset": row["dataset"], "model": label, "seed": row["seed"], **state, "stored_prediction_checks": arrays})
    return rows, records, audits


def selected_gkt_training():
    selected, primary = {}, []
    for dataset in DATASETS[:2]:
        for name in ["GKT", "GKT_MHA"]:
            candidates = []
            for p in sorted((SOURCE / "results/raw").glob(f"{dataset}-{name}-baseline_tuning-*/metrics.json")):
                m = read(p); cfg = read(p.parent / "config.json")
                if read(p.parent / "status.json")["status"] != "completed": continue
                assert cfg["seed"] == 42 and cfg["evaluate_test"] is False and m["test"] is None
                candidates.append((p, m, cfg))
            if candidates:
                assert dataset == "ASSIST2009" and len(candidates) == 36
                p, m, cfg = max(candidates, key=lambda v: v[1]["validation"]["auc"])
                checkpoint = SOURCE / m["checkpoint"]
                assert sha(checkpoint) == m["checkpoint_sha256"]
                best = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert best["config"] == cfg and best["epoch"] == m["best_epoch"]
                assert all(torch.isfinite(t).all() for t in best["model"].values())
                model = build_model(name, read(SOURCE / f"reports/{dataset}_statistics.json")["num_q"], cfg["model_config"])
                model.load_state_dict(best["model"], strict=True)
                history = read(p.parent / "history.json")
                assert max(history, key=lambda h: h["validation"]["auc"])["epoch"] == best["epoch"]
                selected[dataset, model_label(name)] = {"config": cfg, "source": str(p), "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha(checkpoint), "best_validation_auc": m["validation"]["auc"],
                    "test_evaluated": False, "completed_grid_candidates": len(candidates),
                    "candidates": [{"run_id": x[1]["run_id"], "validation_auc": x[1]["validation"]["auc"], "metrics_sha256": sha(x[0])} for x in candidates]}
            for p in sorted((SOURCE / "results/raw").glob(f"{dataset}-{name}-primary-42-*/metrics.json")):
                m = read(p); assert read(p.parent / "status.json")["status"] == "completed"
                checkpoint = SOURCE / m["checkpoint"]; assert sha(checkpoint) == m["checkpoint_sha256"]
                primary.append({"dataset": dataset, "model": model_label(name), "seed": 42,
                    "status": "EXCLUDED_SINGLE_SEED_PRIMARY", "reason": "Different fixed primary config; not selected five-seed final protocol",
                    "validation_auc": m["validation"]["auc"], "test_auc": m["test"]["auc"],
                    "test_accuracy": m["test"]["accuracy"], "test_f1": m["test"]["f1"],
                    "source": str(p), "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint)})
    return selected, primary


def prepare_missing(master, selected):
    specs = read(ROOT / "experiments/professor_campaign_20261030/plan.json")["feature_specs"]
    specs[DS] = read(ROOT / "experiments/ednet_replacement_20261104/plan.json")["feature_spec"]
    entries = []
    for row in master:
        if row["experiment_status"] != "PENDING": continue
        dataset, label = row["dataset"], row["model"]
        data_source = EDNET if dataset == DS else SOURCE
        work = ROOT / ".tools/master_comparison_20261110/workspaces" / dataset
        for relative in [f"data/processed/{dataset}/interactions.csv", f"data/splits/{dataset}.json", f"reports/{dataset}_statistics.json"]:
            source, link = data_source / relative, work / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink(): assert link.is_symlink() and link.resolve() == source.resolve()
            else: link.symlink_to(source)
        stats = read(data_source / f"reports/{dataset}_statistics.json")
        for seed in SEEDS:
            entry = {"dataset": dataset, "model": label, "seed": seed, "status": "PENDING",
                     "gpu_execution_authorized": False, "automatically_launch": False,
                     "work_root": str(work), "source_dataset_root": str(data_source),
                     "processed_sha256": stats["processed_sha256"], "split_sha256": stats["split_sha256"]}
            if (dataset, label) in selected:
                chosen = selected[dataset, label]; cfg = copy.deepcopy(chosen["config"])
                reason = "Reuse highest-validation checkpoint from complete original 36-point grid for seed42; freeze same settings for other four seeds"
                if seed == 42:
                    entry.update(action="REUSE_TRAINED_CHECKPOINT_PENDING_CPU_REPLAY", reusable_checkpoint=chosen["checkpoint"],
                                 checkpoint_sha256=chosen["checkpoint_sha256"])
            elif label == "SAKT + Question Difficulty":
                if dataset == DS:
                    cfg = read(ROOT / f"experiments/ednet_replacement_20261104/configs/SAKT-{seed}.json")
                else: cfg = read(SOURCE / f"results/raw/{dataset}-SAKT-final-{seed}/config.json")
                cfg.update(model="SAKTQuestion", feature_path=specs[dataset]["train_validation_feature_path"])
                cfg["question_difficulty"] = {"algorithm": specs[dataset]["algorithm"], "smoothing": 10.,
                    "feature_sha256": specs[dataset]["train_validation_feature_sha256"]}
                cfg["experimental_implementation_sha256"] = sha(ROOT / "scripts/feature_matched_sakt.py")
                entry["adapter"] = "scripts.ednet_feature_matched_control.prepared_control (generic frozen-sidecar adapter)"
                reason = "Exact native SAKT selected/frozen configuration plus identical existing one-scalar question feature"
            elif dataset == DS:
                rawname = {"GKT-MHA (implemented variant)": "GKT_MHA"}.get(label, label)
                if rawname in ["GKT", "GKT_MHA"]:
                    cfg = read(next((SOURCE / "results/raw").glob(f"ASSIST2009-{rawname}-primary-42-*/config.json")))
                    reason = "Untuned original primary template; graph memory/runtime profiling required on 1452 skills"
                    entry["feasibility"] = "UNPROFILED under 4GiB/900s bounds; no automatic launch"
                else:
                    cfg = read(SOURCE / f"results/raw/ASSIST2009-{rawname}-final-42/config.json")
                    reason = "Prespecified transfer of existing ASSIST2009 selected model settings, without EdNet test-guided tuning"
            else:
                entry.update(action="AWAIT_VALIDATION_ONLY_CONFIGURATION_SELECTION", config=None,
                    reason="Algebra2005 GKT tuning grid not complete; primary seed42 cannot substitute for a selected repeated-seed protocol")
                entries.append(entry); continue
            cfg.update(dataset=dataset, seed=seed, evaluate_test=False, device="cuda", group="prepared_master_comparison_20261110",
                       run_id=f"{dataset}-{cfg['model']}-master-{seed}-20261110-UNLAUNCHED")
            cfg.pop("selection_path", None)
            path = PREP / "prepared_configs" / f"{dataset}-{cfg['model']}-{seed}.json"
            write(path, cfg)
            entry.setdefault("action", "MISSING_TRAINING")
            entry.update(config=str(path.relative_to(ROOT)), config_sha256=sha(path), reason=reason,
                         max_epochs=cfg["max_epochs"], patience=cfg["patience"], selection="strict maximum validation AUC only",
                         source_checkpoint_selection="No test metric used", candidate_window_seconds=900,
                         candidate_allocator_cap_bytes=4 * 1024**3)
            entries.append(entry)
    assert len(entries) == 60
    write(PREP / "plan.json", {"status": "PREPARED_PENDING_CURRENT_ALLOCATION_AND_EXECUTION_BUDGET",
        "not_a_training_authorization": True, "automatic_execution": False, "seeds": SEEDS,
        "entries": entries, "verified_result_slots_missing": len(entries),
        "reusable_training_checkpoints": sum(e["action"].startswith("REUSE_") for e in entries),
        "minimum_missing_selected_configuration_training_runs": sum(not e["action"].startswith("REUSE_") for e in entries),
        "additional_tuning_cost": "Algebra GKT selection unresolved; preserving original protocol would additionally require its candidate trials",
        "resource_scope": "Prior15-run EdNet budget completed/consumed. No additional invocation or allocation assumed from idle hardware.",
        "HiTSKT": "ON HOLD, untouched"})
    return entries


def finish(master, rows, records, audits, metadata, selected, primary, entries):
    for row in master:
        stats = metadata[row["dataset"]]
        row.update(dataset_version=stats.get("display_name", row["dataset"]), processed_sha256=stats["processed_sha256"],
                   split_sha256=stats["split_sha256"], required_seeds=SEEDS,
                   implementation_note="Existing simplified PAM; canonical GKT reproduction not established" if row["model"] == "GKT" else
                   "Existing GKT_MHA implementation; erase-add gates/batchnorm omitted, not canonical published GKT" if row["model"].startswith("GKT-MHA") else
                   "Existing native model; difficulty models add one scalar and fitted item information" )
        group = [r for r in records if r["dataset"] == row["dataset"] and model_label(r["model"]) == row["model"]]
        if group:
            row["training_budget"] = {"max_epochs": sorted({r["config"]["max_epochs"] for r in group}),
                "patience": sorted({r["config"]["patience"] for r in group}), "checkpoint_selection": "strict maximum validation AUC",
                "actual_epochs": {str(r["seed"]): r["epochs_trained"] for r in group},
                "actual_training_seconds": {str(r["seed"]): r["training_seconds"] for r in group},
                "model_configs": {str(r["seed"]): r["config"] for r in group}}
        else:
            missing = [e for e in entries if e["dataset"] == row["dataset"] and e["model"] == row["model"]]
            row["configuration_references"] = [e["config"] for e in missing if e.get("config")]
            row["training_budget"] = "PENDING current allocation/budget; prepared settings max50epochs,patience5; GKT configuration/feasibility gaps documented"
            row["pending_reason"] = "; ".join(sorted({e["reason"] for e in missing}))
            row["source_references"] = [str(PREP.relative_to(ROOT) / "plan.json"),
                                         str(OUT.relative_to(ROOT) / "excluded_single_seed_primary_evidence.csv")]
            row["selected_training_seed_count_available"] = sum(e["action"].startswith("REUSE_") for e in missing)
            if (row["dataset"], row["model"]) in selected:
                v = selected[row["dataset"], row["model"]]
                row["checkpoint_references"] = [{"seed": 42, "path": v["checkpoint"], "sha256": v["checkpoint_sha256"], "status": "TRAINED_NOT_TEST_REPLAYED"}]
                row["source_references"] = [v["source"]]
    csv_file(OUT / "master_comparison.csv", master); write(OUT / "master_comparison.json", master)
    per_seed = []
    look = {(r["dataset"], r["model"], r["seed"]): r for r in rows}
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                per_seed.append(look.get((dataset, model, seed), {"dataset": dataset, "model": model, "seed": seed,
                    "experiment_status": "PENDING", **{k: None for k in FIELDS}}))
    csv_file(OUT / "individual_seed_results.csv", per_seed)
    csv_file(OUT / "excluded_single_seed_primary_evidence.csv", primary)
    write(OUT / "selected_reusable_gkt_checkpoints.json", list(selected.values()))
    write(OUT / "dataset_provenance.json", list(metadata.values()))
    write(OUT / "experiment_provenance.json", {"completed_records": records, "verification": audits,
        "no_training_or_checkpoint_prediction_inference": True, "prior_core_report": str(CORE),
        "prior_core_report_sha256": sha(CORE / "PROFESSOR_REPORT.md")})
    write(OUT / "verification_summary.json", {"status": "PASS_AVAILABLE_WITH_PENDING", "expected_groups": 27,
        "verified_five_seed_groups": sum(r["experiment_status"] == "VERIFIED_5_SEEDS" for r in master),
        "pending_groups": sum(r["experiment_status"] == "PENDING" for r in master), "verified_seed_records": len(rows),
        "missing_verified_seed_slots": 135 - len(rows), "checkpoints_strictly_loaded": len(audits),
        "saved_prediction_arrays_checked": sum(len(a["stored_prediction_checks"]) for a in audits),
        "fresh_checkpoint_prediction_passes": 0, "new_GPU_training_runs": 0, "sample_standard_deviation_ddof": 1,
        "single_seed_primary_results_excluded_from_aggregates": len(primary), "trained_GKT_seed42_reuse_candidates": len(selected),
        "native_vs_prior_CPU_test_metric_difference": "within2e-6; CPU replay consistently used; native values retained per seed",
        "test_selection": "validation-only; no new test inference or tuning performed"})


def main():
    OUT.mkdir(parents=True, exist_ok=True); PREP.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    metadata, expected = expected_data()
    rows, records, audits = load_existing(metadata, expected)
    master = aggregate(rows)
    selected, primary = selected_gkt_training()
    entries = prepare_missing(master, selected)
    finish(master, rows, records, audits, metadata, selected, primary, entries)
    print(f"VERIFIED {len(rows)} existing seed records; {sum(r['experiment_status']=='VERIFIED_5_SEEDS' for r in master)}/27 five-seed groups complete; no training or prediction inference.")


if __name__ == "__main__": main()

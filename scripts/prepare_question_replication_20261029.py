"""Freeze configurations and a proposed budget. Never launches a campaign."""
import copy
from pathlib import Path
from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL, write_json
from scripts.architecture_study import read_json, digest


def main():
    root = PROJECT; destination = root / "experiments/question_replication_20261029"
    destination.mkdir(exist_ok=False)
    seeds = read_json(SOURCE / "configs/protocol.json")["training_seeds"]
    light = read_json(root / CONTROL / "configs/one_layer_question_difficulty.json")
    result_path = root / "pilot-workspace/results/raw" / light["run_id"] / "metrics.json"
    result = read_json(result_path)
    assert read_json(result_path.parent / "status.json")["status"] == "completed"
    selected = read_json(SOURCE / "experiments/selections/ASSIST2009-SAKT.json")["selected_run_id"]
    sakt = read_json(SOURCE / "configs/runs" / f"{selected}.json")
    candidates = []; references = []
    for seed in seeds:
        config = copy.deepcopy(light)
        if seed != 42:
            config.update(seed=seed, group="question_replication",
                          run_id=f"ASSIST2009-LightSAKTQuestion-replication-{seed}-v1")
            config["pilot"]["baseline_reference"] = f"ASSIST2009-LightSAKT-final-{seed}"
        relative = Path("configs") / f"LightSAKTQuestion-{seed}.json"
        write_json(destination / relative, config)
        candidates.append({"seed": seed, "model": config["model"], "run_id": config["run_id"],
                           "configuration": str(relative), "configuration_sha256": digest(destination / relative),
                           "action": "reuse_verified_completed_seed42" if seed == 42 else "pending_budget"})
        control = copy.deepcopy(sakt)
        control.update(model="SAKTQuestion", seed=seed, group="question_feature_control", device="cuda",
                       evaluate_test=False, run_id=f"ASSIST2009-SAKTQuestion-featurematched-{seed}-v1")
        control["question_difficulty"] = copy.deepcopy(light["question_difficulty"])
        control["experimental_implementation_sha256"] = digest(root / "scripts/feature_matched_sakt.py")
        relative = Path("configs") / f"SAKTQuestion-{seed}.json"
        write_json(destination / relative, control)
        candidates.append({"seed": seed, "model": control["model"], "run_id": control["run_id"],
                           "configuration": str(relative), "configuration_sha256": digest(destination / relative),
                           "action": "pending_budget"})
        for model in ["LightSAKT", "SAKT"]:
            path = SOURCE / "results/raw" / f"ASSIST2009-{model}-final-{seed}" / "metrics.json"
            reference = read_json(path)
            references.append({"seed": seed, "model": model, "metrics": str(path),
                               "metrics_sha256": digest(path), "validation_auc": reference["validation"]["auc"]})
    plan = {"status": "PREPARED_NOT_AUTHORIZED_FOR_GPU", "launches_jobs": False,
            "frozen_training_source_sha256": result["training_code_sha256"],
            "dataset_adapter_sha256": digest(root / "scripts/question_difficulty_inputs.py"),
            "control_implementation_sha256": digest(root / "scripts/feature_matched_sakt.py"),
            "dataset": "ASSIST2009", "seeds": seeds, "comparison": "Paired validation AUC deltas against original model references, feature-matched LightSAKT versus SAKT, mean/sample SD across all five; never best seed.",
            "protocol": "Offline student-disjoint full-training-corpus fitting; not forward-time performance.",
            "feature_spec": light["question_difficulty"], "feature_controls": "Exact same frozen feature values and one zero-initialized neural scale in both model families.",
            "model_settings": "Each model retains its own previously selected hyperparameters. SAKT control lr0.0005/4heads/dropout0.2; LightSAKT lr0.001/2heads/dropout0.1. No new tuning; feature matched, not hyperparameter identical.",
            "completed_seed42": {"run_id": light["run_id"], "validation_auc": result["validation"]["auc"], "checkpoint_sha256": result["checkpoint_sha256"]},
            "configurations": candidates, "baseline_references": references,
            "new_runs": 9, "new_light_runs": 4, "new_sakt_controls": 5,
            "proposed_resource_budget_not_approved": {"concurrency": 1, "gpu_allocator_cap_bytes": 4 * 1024**3,
                "maximum_seconds_per_run_invocation": 900, "maximum_new_invocations": 9,
                "maximum_total_gpu_seconds": 8100, "maximum_total_gpu_minutes": 135,
                "stop_on_resource_breach": True, "automatic_extra_windows_or_retries": False},
            "execution_conditions": "Requires separate explicit campaign resource authorization and a budget-enforcing seed-general launcher. Current pilot launcher is deliberately seed42-only. No launch entrypoint in this preparation script.",
            "control_implementation": "scripts/feature_matched_sakt.py (CPU-validated only; use feature_matched_control context inside record_metrics). Bound implementation digest in each control config.",
            "test_policy": "All evaluate_test=false; no test features, loader, metrics or predictions permitted in campaign.",
            "other_datasets": "Algebra2005 needs verified question/step semantics. ASSIST2015 current release lacks separate problem IDs; no silent transplant or dataset change."}
    write_json(destination / "plan.json", plan)
    print("Prepared 10 configurations: reuse completed LightSAKT seed42; 9 new runs remain unauthorized.")


if __name__ == "__main__":
    main()

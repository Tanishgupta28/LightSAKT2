"""Prepare a separate, validation-only architecture study. Never launches jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("ASSIST2009", "ASSIST2015", "Algebra2005")
ARMS = {
    "two_layer_linear": {"num_layers": 2, "attention_type": "linear"},
    "one_layer_softmax": {"num_layers": 1, "attention_type": "softmax"},
    "one_layer_recency_softmax": {"num_layers": 1, "attention_type": "recency_softmax"},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(root):
    """Read existing configurations/validation records, never test predictions."""
    root = Path(root)
    protocol = read_json(root / "configs/protocol.json")
    plan = {
        "schema_version": 1,
        "status": "prepared_not_submitted",
        "launches_jobs": False,
        "selection_metric": "validation_auc",
        "test_evaluation": False,
        "training_seeds": protocol["training_seeds"],
        "split_seed": protocol["split_seed"],
        "split_unit": protocol["split_unit"],
        "split_ratios": protocol["split_ratios"],
        "protocol_sha256": digest(root / "configs/protocol.json"),
        "selection_rule": "mean paired validation-AUC delta over all five seeds; never choose a best seed",
        "screening": "seed 42 is a diagnostic; complete all prespecified seeds before ranking arms",
        "tie_break": "higher mean validation AUC, then fewer parameters, then lexical arm name",
        "test_policy": "Existing baseline test reports are historical context only. Freeze architecture and settings before any later authorized final test evaluation.",
        "datasets": {},
        "baseline_references": [],
        "proposed_runs": [],
    }
    for dataset in DATASETS:
        selection_path = Path("experiments/selections") / f"{dataset}-LightSAKT.json"
        selection = read_json(root / selection_path)
        if selection["selection_metric"] != "validation_auc" or selection["test_access_for_selection"]:
            raise ValueError("Baseline selection must be validation only")
        selected = selection["selected_run_id"]
        config = read_json(root / "configs/runs" / f"{selected}.json")
        stats_path = Path("reports") / f"{dataset}_statistics.json"
        stats = read_json(root / stats_path)
        split_path = Path("data/splits") / f"{dataset}.json"
        if digest(root / split_path) != stats["split_sha256"]:
            raise ValueError(f"Changed split: {dataset}")
        checks = {"seq_len": protocol["seq_len"][dataset], "batch_size": protocol["batch_size"],
                  "max_epochs": protocol["max_epochs"], "patience": protocol["patience"],
                  "betas": protocol["betas"], "epsilon": protocol["epsilon"],
                  "weight_decay": protocol["weight_decay"]}
        if any(config[k] != v for k, v in checks.items()):
            raise ValueError(f"Selected baseline disagrees with protocol: {dataset}")
        plan["datasets"][dataset] = {
            "selected_grid_run": selected,
            "selected_config_sha256": digest(root / "configs/runs" / f"{selected}.json"),
            "statistics_sha256": digest(root / stats_path),
            "selection_sha256": digest(root / selection_path),
            "num_q": stats["num_q"],
            "processed_sha256": stats["processed_sha256"],
            "split_sha256": stats["split_sha256"],
            "identifier_semantics": stats["identifier_semantics"],
            "split_students": stats["split_students"],
            "split_interactions": stats["split_interactions"],
            "model_config": config["model_config"],
            "learning_rate": config["learning_rate"],
        }
        aucs = []
        for seed in protocol["training_seeds"]:
            baseline_id = f"{dataset}-LightSAKT-final-{seed}"
            path = Path("results/raw") / baseline_id / "metrics.json"
            result = read_json(root / path)
            actual_config = read_json(root / path.parent / "config.json")
            if result["split_sha256"] != stats["split_sha256"] or result["processed_sha256"] != stats["processed_sha256"]:
                raise ValueError(f"Baseline data binding mismatch: {baseline_id}")
            if (actual_config["model"] != "LightSAKT" or actual_config["seed"] != seed
                    or actual_config["model_config"] != config["model_config"]
                    or any(actual_config[k] != config[k] for k in (*checks, "learning_rate"))):
                raise ValueError(f"Baseline configuration mismatch: {baseline_id}")
            aucs.append(result["validation"]["auc"])
            plan["baseline_references"].append({
                "dataset": dataset, "seed": seed, "arm": "original_one_layer_linear",
                "status": "existing_completed_reference_no_new_training",
                "run_id": baseline_id, "metrics_path": str(path), "metrics_sha256": digest(root / path),
                "validation_auc": result["validation"]["auc"],
                "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"],
                "training_code_sha256": result["training_code_sha256"],
            })
            for arm, architecture in ARMS.items():
                candidate = copy.deepcopy(config)
                candidate.update(model="LightSAKTResearch", group="architecture_study", seed=seed,
                                 device="cuda", evaluate_test=False,
                                 run_id=f"{dataset}-LightSAKTResearch-{arm}-{seed}-v1")
                candidate["model_config"].update(architecture)
                candidate["architecture_study"] = {
                    "arm": arm, "baseline_reference": baseline_id,
                    "processed_sha256": stats["processed_sha256"], "split_sha256": stats["split_sha256"],
                }
                plan["proposed_runs"].append({
                    "arm": arm, "status": "not_submitted",
                    "config_path": f"configs/architecture_study/{candidate['run_id']}.json",
                    "config": candidate,
                })
        plan["datasets"][dataset].update(baseline_validation_auc_mean=mean(aucs),
                                          baseline_validation_auc_std=stdev(aucs))
    return plan


def write_plan(plan, output):
    """Only create new study files; refuse to overwrite different content."""
    output = Path(output)
    files = {"experiments/architecture_study/plan.json": plan}
    files.update({run["config_path"]: run["config"] for run in plan["proposed_runs"]})
    for relative, value in files.items():
        path = output / relative
        encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"Refusing to replace a different study file: {path}")
    for relative, value in files.items():
        path = output / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    plan = make_plan(args.source_root)
    write_plan(plan, args.output_root)
    print(f"Prepared {len(plan['proposed_runs'])} validation-only configurations; no jobs submitted.")

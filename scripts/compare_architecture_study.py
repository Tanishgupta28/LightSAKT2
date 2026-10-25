"""Compare recorded validation AUC only; missing experiments stay pending."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

from scripts.architecture_study import ARMS, DATASETS, read_json


def comparison(root, plan):
    root = Path(root)
    rows = []
    for dataset in DATASETS:
        references = {r["seed"]: r["validation_auc"] for r in plan["baseline_references"]
                      if r["dataset"] == dataset}
        for arm in ("original_one_layer_linear", *ARMS):
            if arm == "original_one_layer_linear":
                values = references.copy()
            else:
                values = {}
                for entry in plan["proposed_runs"]:
                    config = entry["config"]
                    if config["dataset"] != dataset or entry["arm"] != arm:
                        continue
                    directory = root / "results/raw" / config["run_id"]
                    path = directory / "metrics.json"
                    if not path.exists():
                        continue
                    status = read_json(directory / "status.json")
                    if status["status"] != "completed":
                        continue
                    if read_json(directory / "config.json") != config:
                        raise ValueError("Study configuration changed")
                    result = read_json(path)
                    metadata = plan["datasets"][dataset]
                    if (result.get("test") is not None or config["evaluate_test"]
                            or result["processed_sha256"] != metadata["processed_sha256"]
                            or result["split_sha256"] != metadata["split_sha256"]
                            or result["run_id"] != config["run_id"]
                            or result["seed"] != config["seed"] or result["model"] != config["model"]):
                        raise ValueError("Invalid validation-only evidence binding")
                    auc = result["validation"]["auc"]
                    if auc is None or not math.isfinite(auc) or not 0 <= auc <= 1:
                        raise ValueError("Invalid validation AUC")
                    values[config["seed"]] = auc
            complete = set(values) == set(references)
            deltas = [values[s] - references[s] for s in references] if complete else []
            rows.append({"dataset": dataset, "arm": arm,
                         "status": "complete_recorded_validation" if complete else "pending",
                         "completed_seeds": len(values), "required_seeds": len(references),
                         "validation_auc_mean": mean(values.values()) if complete else None,
                         "validation_auc_std": stdev(values.values()) if complete else None,
                         "paired_validation_auc_delta_mean": mean(deltas) if complete else None,
                         "paired_validation_auc_delta_std": stdev(deltas) if complete else None,
                         "checkpoint_replay": "required before performance claim for new variants"})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/architecture_study/validation_comparison.csv"))
    args = parser.parse_args()
    rows = comparison(args.root, read_json(args.root / "experiments/architecture_study/plan.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("Validation comparison written; no test evaluation performed.")

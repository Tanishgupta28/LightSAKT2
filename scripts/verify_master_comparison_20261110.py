"""Standard-library verification of exported tables, not model execution."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def verify(root):
    root = Path(root)
    with (root / "master_comparison.csv").open(newline="") as f: master = list(csv.DictReader(f))
    with (root / "individual_seed_results.csv").open(newline="") as f: individual = list(csv.DictReader(f))
    assert len(master) == 27 and len(individual) == 135
    assert len({(r["dataset"], r["model"]) for r in master}) == 27
    assert len({(r["dataset"], r["model"], r["seed"]) for r in individual}) == 135
    assert {r["dataset"] for r in master} == {"ASSIST2009", "Algebra2005", "EdNetKT1_5k_first"}
    assert len({r["model"] for r in master}) == 9
    fields = ["validation_auc", "test_auc", "test_accuracy", "test_f1"]
    measured = [r for r in individual if r["experiment_status"] == "VERIFIED"]
    configs = 0
    for r in measured:
        path = root / "configurations" / Path(r["configuration_reference"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["configuration_sha256"]
        cfg = json.loads(path.read_text())
        assert cfg["dataset"] == r["dataset"] and cfg["seed"] == int(r["seed"])
        configs += 1
    complete = pending = 0
    for r in master:
        group = [s for s in measured if (s["dataset"], s["model"]) == (r["dataset"], r["model"])]
        assert len(group) == int(r["seed_count"])
        assert all((s["processed_sha256"], s["split_sha256"]) == (r["processed_sha256"], r["split_sha256"]) for s in group)
        if r["experiment_status"] == "VERIFIED_5_SEEDS":
            complete += 1
            assert {int(s["seed"]) for s in group} == {42, 123, 2026, 3407, 7777}
            for field in fields:
                values = [float(s[field]) for s in group]
                mu = math.fsum(values) / 5
                sd = math.sqrt(math.fsum((v - mu) ** 2 for v in values) / 4)
                assert math.isclose(float(r[field + "_mean"]), mu, abs_tol=1e-12, rel_tol=0)
                assert math.isclose(float(r[field + "_std"]), sd, abs_tol=1e-12, rel_tol=0)
            assert r["primary_metric"] == "test_auc"
            assert float(r["mean"]) == float(r["test_auc_mean"])
            assert float(r["standard_deviation"]) == float(r["test_auc_std"])
            assert len(json.loads(r["checkpoint_references"])) == 5
        else:
            pending += 1; assert r["experiment_status"] == "PENDING" and len(group) < 5
            assert r["mean"] == r["standard_deviation"] == ""
            for field in fields:
                assert r[field] == "PENDING" and r[field + "_mean"] == r[field + "_std"] == ""
    for r in individual:
        if r["experiment_status"] == "PENDING": assert all(r[k] == "" for k in fields)
    js = json.loads((root / "master_comparison.json").read_text())
    assert len(js) == 27
    for csvrow, j in zip(master, js):
        assert (csvrow["dataset"], csvrow["model"], csvrow["experiment_status"]) == (j["dataset"], j["model"], j["experiment_status"])
        for field in fields:
            for suffix in ["_mean", "_std"]:
                key = field + suffix
                assert (j[key] is None and csvrow[key] == "") or float(csvrow[key]) == j[key]
    return {"status": "PASS", "complete_five_seed_groups": complete, "pending_groups": pending,
        "verified_seed_records": len(measured), "missing_seed_records": 135 - len(measured),
        "configuration_snapshots_verified": configs, "sample_sd_denominator": "n-1",
        "pending_numeric_cells_empty": True, "CSV_JSON_consistent": True,
        "scope": "Export integrity, five-seed grouping, dataset bindings, copied configs and independent mean/sample-SD arithmetic; no training or inference"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('reports/master_comparison_20261110')), indent=2))

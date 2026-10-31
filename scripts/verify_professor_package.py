"""Verify an extracted professor package using Python's standard library only.

This checks file integrity, result aggregation and paired differences. It does
not execute a model, retrieve datasets, connect to GitHub, or launch training.
"""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean, stdev
import sys


def check(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = manifest["files"]
    for name, value in expected.items():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and ".." not in relative.parts, name
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == value["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"], name
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual == set(expected) | {"MANIFEST.json"}, "Missing or extra package file"
    records = json.loads((root / "individual_seed_results.json").read_text())
    summary = json.loads((root / "summary_results.json").read_text())
    differences = json.loads((root / "absolute_auc_improvements.json").read_text())
    assert len(records) == 45 and len(summary) == 9
    assert len({(r["dataset"], r["model"], r["seed"]) for r in records}) == 45
    measured = [r for r in records if r["status"] == "VERIFIED"]
    missing = [r for r in records if r["status"] == "UNAVAILABLE_CURRENT_PROTOCOL"]
    assert len(measured) == 40 and len(missing) == 5
    assert all(r["dataset"] == "ASSIST2015" and r["model"] == "LightSAKT+Difficulty" for r in missing)
    lookup = {(r["dataset"], r["model"], r["seed"]): r for r in measured}

    def equal(a, b):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)

    for row in summary:
        group = [r for r in measured if (r["dataset"], r["model"]) == (row["dataset"], row["model"])]
        assert len(group) == row["verified_seeds"]
        for split in ["validation", "test"]:
            for metric in ["auc", "accuracy", "f1", "precision", "recall", "bce"]:
                values = [r[f"{split}_{metric}"] for r in group]
                if values:
                    equal(mean(values), row[f"{split}_{metric}_mean"])
                    equal(stdev(values), row[f"{split}_{metric}_sd"])
                else:
                    assert row[f"{split}_{metric}_mean"] is None and row[f"{split}_{metric}_sd"] is None
    for row in differences:
        dataset, comparator = row["dataset"], row["comparator"]
        group = []
        for seed in [42, 123, 2026, 3407, 7777]:
            key = dataset, "LightSAKT+Difficulty", seed
            if key in lookup:
                group.append((lookup[key], lookup[dataset, comparator, seed]))
        assert len(group) == row["paired_seeds"]
        for split in ["validation", "test"]:
            delta = [a[f"{split}_auc"] - b[f"{split}_auc"] for a, b in group]
            if delta:
                equal(mean(delta), row[f"{split}_auc_delta_mean"])
                equal(stdev(delta), row[f"{split}_auc_delta_sd"])
                assert sum(v > 0 for v in delta) == row[f"{split}_positive_seed_count"]
    with (root / "individual_seed_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 45
    for row in rows:
        if row["status"] == "VERIFIED":
            source = lookup[row["dataset"], row["model"], int(row["seed"])]
            for split in ["validation", "test"]:
                for metric in ["auc", "accuracy", "f1", "precision", "recall", "bce"]:
                    equal(float(row[f"{split}_{metric}"]), source[f"{split}_{metric}"])
    references = json.loads((root / "checkpoint_references.json").read_text())
    assert len(references) == 40
    for row in references:
        config = root / row["configuration"]
        assert hashlib.sha256(config.read_bytes()).hexdigest() == row["configuration_sha256"]
        identity = row["dataset"], row["model"], row["seed"]
        assert row["checkpoint_sha256"] == lookup[identity]["checkpoint_sha256"]
        assert row["weights_in_archive"] is False
    return {"status": "PASS", "verified_payload_files": len(expected), "measured_model_seed_results": 40,
            "unavailable_model_seed_slots": 5, "sample_standard_deviation_ddof": 1,
            "scope": "File integrity, metric aggregation, CSV/JSON consistency, paired differences and configuration/checkpoint references. External weights/data not included or executed."}


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    print(json.dumps(check(directory), indent=2))

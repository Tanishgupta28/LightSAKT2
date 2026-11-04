"""Standalone standard-library verifier for the qualified three-dataset release."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys


def check(root):
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = manifest["files"]
    for name, entry in expected.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected) | {"MANIFEST.json"}
    individuals = json.loads((root / "individual_seed_results.json").read_text())
    summary = json.loads((root / "summary_results.json").read_text())
    gains = json.loads((root / "absolute_auc_improvements.json").read_text())
    assert len(individuals) == 45 and len(summary) == 9 and len(gains) == 6
    assert {r["dataset"] for r in individuals} == {"ASSIST2009", "Algebra2005", "EdNetKT1_5k_first"}
    lookup = {(r["dataset"], r["model"], r["seed"]): r for r in individuals}
    assert len(lookup) == 45
    assert all(r["status"] in {"VERIFIED", "PENDING"} for r in individuals)
    measured = [r for r in individuals if r["status"] == "VERIFIED"]
    def equal(a, b):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)
    def aggregate(values, row, prefix):
        if values:
            equal(mean(values), row[prefix + "_mean"])
            if len(values) > 1: equal(stdev(values), row[prefix + "_sd"])
            else: assert row[prefix + "_sd"] is None
        else:
            assert row[prefix + "_mean"] is None and row[prefix + "_sd"] is None
    for row in summary:
        group = [r for r in measured if r["dataset"] == row["dataset"] and r["model"] == row["model"]]
        assert len(group) == row["verified_seeds"]
        assert row["status"] == ("COMPLETE" if len(group) == 5 else "PENDING")
        for split in ["validation", "test"]:
            for metric in ["auc", "accuracy", "f1", "precision", "recall", "bce"]:
                key = f"{split}_{metric}"
                aggregate([r[key] for r in group], row, key)
    for row in gains:
        group = []
        for seed in [42, 123, 2026, 3407, 7777]:
            a = lookup[row["dataset"], "LightSAKT+Difficulty", seed]
            b = lookup[row["dataset"], row["comparator"], seed]
            if a["status"] == b["status"] == "VERIFIED": group.append((a, b))
        assert row["paired_seeds"] == len(group)
        for split in ["validation", "test"]:
            v = [a[f"{split}_auc"] - b[f"{split}_auc"] for a, b in group]
            aggregate(v, row, f"{split}_auc_delta")
            assert row[f"{split}_positive_seed_count"] == sum(x > 0 for x in v)
    for name, identity in [("individual_seed_results", ["dataset", "model", "seed"]),
                           ("summary_results", ["dataset", "model"]),
                           ("absolute_auc_improvements", ["dataset", "comparator"])]:
        source = json.loads((root / f"{name}.json").read_text())
        with (root / f"{name}.csv").open(newline="") as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == len(source)
        for actual, reference in zip(rows, source):
            for key in identity: assert str(actual[key]) == str(reference[key])
            for key, value in reference.items():
                if isinstance(value, (float, int)): equal(float(actual[key]), value)
                elif value is None: assert actual[key] == ""
    references = json.loads((root / "checkpoint_references.json").read_text())
    assert len(references) == len(measured)
    for row in references:
        assert hashlib.sha256((root / row["configuration"]).read_bytes()).hexdigest() == row["configuration_sha256"]
        assert row["checkpoint_sha256"] == lookup[row["dataset"], row["model"], row["seed"]]["checkpoint_sha256"]
        assert row["weights_in_archive"] is False
    return {"status": "PASS", "verified_payload_files": len(expected), "measured_model_seed_results": len(measured),
        "pending_model_seed_slots": 45 - len(measured), "sample_standard_deviation_ddof": 1,
        "scope": "Portable payload hashes, tables/CSV-JSON equality, aggregation, paired deltas and checkpoint config references; external data/weights not executed."}


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent), indent=2))

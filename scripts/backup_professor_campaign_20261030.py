"""Archive terminal research artifacts without modifying original files."""
import hashlib
import tarfile
from pathlib import Path
from datetime import datetime, timezone
from scripts.architecture_study import digest, read_json
from scripts.architecture_pilot import PROJECT, SOURCE, write_json

OUT = PROJECT / "reports/professor_results_20261030"
DESTINATION = SOURCE / "backups/professor_results_20261030"


def archive(name, files):
    destination = DESTINATION / (name + ".tar.gz")
    manifest = {relative: digest(path) for path, relative in files}
    if not destination.exists():
        temporary = destination.with_suffix(".incomplete")
        with tarfile.open(temporary, "w:gz") as output:
            for path, relative in files:
                output.add(path, arcname=relative, recursive=False)
        temporary.rename(destination)
    with tarfile.open(destination) as saved:
        assert {m.name for m in saved.getmembers()} == set(manifest)
        for member in saved.getmembers():
            checksum = hashlib.sha256()
            with saved.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            assert checksum.hexdigest() == manifest[member.name]
    return {"path": str(destination), "sha256": digest(destination), "size_bytes": destination.stat().st_size,
            "member_sha256": manifest, "archive_readback_verified": True}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plan = read_json(PROJECT / "experiments/professor_campaign_20261030/plan.json")
    records = []
    for dataset, features in plan["feature_specs"].items():
        files = [(Path(features[key + "_feature_path"]), dataset + "/features/" + Path(features[key + "_feature_path"]).name)
                 for key in ["train_validation", "test"]]
        records.append({"component": dataset + "-features", **archive(dataset + "-frozen_features", files)})
    campaign = read_json(OUT / "campaign_status.json")
    for entry in plan["entries"]:
        work = Path(entry["work"]); run = entry["run_id"]
        raw = work / "results/raw" / run
        if not (raw / "status.json").exists():
            continue
        status = read_json(raw / "status.json")["status"]
        if status != "completed" and campaign["status"] == "running":
            continue
        files = [(p, run + "/raw/" + p.name) for p in sorted(raw.glob("*.json"))]
        files.extend((p, run + "/checkpoints/" + p.name) for p in sorted((work / "checkpoints" / run).glob("*.pt")))
        if not any(p.suffix == ".pt" for p, _ in files):
            continue
        records.append({"component": run, "training_status": status, **archive(run, files)})
    if campaign["status"] == "completed":
        predictions = []
        for replay in sorted((OUT / "replays").glob("*.json")):
            for split, artifact in read_json(replay)["prediction_artifacts"].items():
                path = Path(artifact["path"])
                assert digest(path) == artifact["sha256"]
                predictions.append((path, "predictions/" + path.name))
        records.append({"component": "verified_validation_test_predictions",
                        **archive("verified_validation_test_predictions", predictions)})
    write_json(OUT / "checkpoint_backups.json", {"status": "READBACK_VERIFIED", "updated_utc": datetime.now(timezone.utc).isoformat(),
                                               "records": records})
    print(f"Verified {len(records)} local archives under {DESTINATION}")


if __name__ == "__main__":
    main()

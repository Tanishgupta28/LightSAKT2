"""Preserve new checkpoints and create a readback-verified reporting archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
from scripts.ednet_replacement_data import ROOT, WORK, OUT as EDNET_OUT, BASE, DS, write
from scripts.architecture_study import digest
from scripts.verify_three_dataset_package import check

OUT = ROOT / "reports/professor_three_datasets_20261104"
BACKUP = ROOT.parent / "LightSAKT2/backups/ednet_replacement_20261104"
DEST = ROOT.parent / "LightSAKT2/backups/professor_three_datasets_20261104"


def verified_tar(path, members):
    hashes = {str(name): digest(file) for name, file in members.items()}
    if not path.exists():
        with tarfile.open(path, "x:gz") as archive:
            for name, file in members.items(): archive.add(file, arcname=str(name), recursive=False)
    with tarfile.open(path, "r:gz") as archive:
        assert set(archive.getnames()) == set(hashes)
        for name, expected in hashes.items():
            stream = archive.extractfile(name)
            assert stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
            "verified_members": hashes, "status": "PASS_READBACK"}


def backup():
    BACKUP.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "experiments/ednet_replacement_20261104/plan.json").read_text())
    results = []
    for entry in plan["entries"]:
        raw = WORK / "results/raw" / entry["run_id"]
        if not (raw / "status.json").exists() or json.loads((raw / "status.json").read_text())["status"] != "completed": continue
        files = {f"checkpoints/{name}.pt": WORK / "checkpoints" / entry["run_id"] / f"{name}.pt" for name in ["best", "latest"]}
        files.update({f"results/{p.name}": p for p in raw.glob("*.json")})
        for p in (WORK / "results/predictions").glob(entry["run_id"] + "-*.npz"):
            files[f"predictions/{p.name}"] = p
        # Do not freeze an incomplete prediction backup: require both CPU splits.
        if sum(name.startswith("predictions/") for name in files) != 2: continue
        r = verified_tar(BACKUP / f"{entry['run_id']}.tar.gz", files)
        r.update(run_id=entry["run_id"], seed=entry["seed"], model=entry["model"]); results.append(r)
    dataset_files = {str(p.relative_to(WORK)): p for p in (WORK / "data").rglob("*") if p.is_file()}
    dataset_files[f"reports/{DS}_statistics.json"] = WORK / f"reports/{DS}_statistics.json"
    dataset_backup = verified_tar(BACKUP / "ednet_frozen_population_splits_features.tar.gz", dataset_files)
    write(OUT / "checkpoint_backups.json", {"status": "PASS_COMPLETED_REPLAYED_RUNS", "completed_backups": len(results),
        "runs": results, "dataset_processed_population_splits_features": dataset_backup,
        "raw_archives_preserved_at": str(BASE / "raw"), "original_artifacts_unchanged": True})
    print(f"BACKED_UP {len(results)} completed replayed runs and frozen dataset; all members hash-verified")


def package():
    DEST.mkdir(parents=True, exist_ok=True)
    stage = BASE / "portable/LightSAKT_question_difficulty_three_datasets_20261104"
    archive = DEST / "LightSAKT_question_difficulty_three_datasets_20261104.zip"
    assert not archive.exists() and not stage.exists(), "Never overwrite a previous release archive/stage"
    stage.mkdir(parents=True)
    for p in OUT.rglob("*"):
        if p.is_file() and p.name not in {"archive_verification.json", "packaging.log"}:
            q = stage / p.relative_to(OUT); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    for p in EDNET_OUT.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".xml", ".md", ".csv", ".jsonl"}:
            q = stage / "ednet_evidence" / p.relative_to(EDNET_OUT); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    records = json.loads((OUT / "experiment_provenance.json").read_text())["records"]
    for r in records:
        raw = Path(r["config_file"]).parent
        for name in ["environment.json", "history.json", "metrics.json", "status.json"]:
            p = raw / name
            if p.exists():
                q = stage / "run_metadata" / r["run_id"] / name; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    for dataset in ["ASSIST2009", "Algebra2005", DS]:
        data_root = WORK if dataset == DS else ROOT.parent / "LightSAKT2"
        p = data_root / f"data/splits/{dataset}.json"
        q = stage / "split_identifiers" / f"{dataset}.json"; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    for name in ["selected_students.json", "source_members.json"]:
        q = stage / "ednet_population" / name; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(WORK / "data/selection" / name, q)
    shutil.copy2(WORK / f"data/processed/{DS}/vocabulary.json", stage / "ednet_population/vocabulary.json")
    historical = ROOT / "reports/professor_release_20261031"
    for name in ["ASSIST2015_LIMITATION.md", "verification_summary.json", "per_seed_verification.json",
                 "real_data_leakage_audit.json", "archive_verification.json", "git_bundle_verification.json"]:
        p = historical / name; q = stage / "historical_evidence" / name; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    for p in (ROOT / "src").rglob("*.py"):
        q = stage / "source" / p.relative_to(ROOT); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    helper_names = ["ednet_replacement_data", "prepare_ednet_campaign_20261104", "run_ednet_campaign_20261104",
        "replay_ednet_campaign_20261104", "report_three_datasets_20261104", "archive_three_datasets_20261104",
        "audit_ednet_completion_20261104", "audit_ednet_source_rows", "reproduce_ednet_predictions_20261104", "verify_three_dataset_package",
        "professor_data_20261030", "question_difficulty_inputs", "pilot_metric_extensions", "pilot_shared_resources",
        "architecture_study", "architecture_pilot", "feature_matched_sakt", "ednet_feature_matched_control"]
    for name in helper_names:
        q = stage / "source/scripts" / f"{name}.py"; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / f"scripts/{name}.py", q)
    for p in (ROOT / "experiments/ednet_replacement_20261104").rglob("*.json"):
        q = stage / "source" / p.relative_to(ROOT); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    for p in [ROOT / "requirements.txt", ROOT / "tests/test_ednet_replacement.py"]:
        if p.exists():
            q = stage / "source" / p.relative_to(ROOT); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    shutil.copy2(ROOT / "scripts/verify_three_dataset_package.py", stage / "verify_three_dataset_package.py")
    (stage / "README.md").write_text("# LightSAKT three-dataset reporting package\n\nRead PROFESSOR_SUMMARY.md and PROFESSOR_REPORT.md. EdNet is the deterministic 5k-student first-question/session subset.\n\nRun `python verify_three_dataset_package.py .` from this directory. This checks package integrity and measured aggregation, not external models/data. Raw data, student-level predictions and model weights are excluded; checkpoint references and local backup locators are explicit.\n\nASSIST2015 is replaced, not a completed difficulty experiment. No attention-superiority or chronological-deployment claim. Remote publication is not established by this archive.\n")
    files = {p.relative_to(stage).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(stage.rglob("*")) if p.is_file()}
    write(stage / "MANIFEST.json", {"format": "lightsakt-three-datasets-reporting-v1", "files": files,
        "scope": "Reports/source/config/provenance/checkpoint references; no raw data, weights or student arrays"})
    result = check(stage)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file(): z.write(p, f"{stage.name}/{p.relative_to(stage).as_posix()}")
    verify_root = BASE / "portable_readback"
    assert not verify_root.exists(); verify_root.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
        z.extractall(verify_root)
    readback = check(verify_root / stage.name)
    write(OUT / "archive_verification.json", {"status": "PASS", "archive": str(archive), "sha256": digest(archive),
        "bytes": archive.stat().st_size, "zip_crc_all_members": True, "stage_check": result, "extracted_readback_check": readback,
        "model_weights_in_archive": False, "raw_data_or_student_arrays_in_archive": False})
    print(json.dumps({"archive": str(archive), "status": "PASS", **readback}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--package", action="store_true"); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    backup()
    if args.package: package()

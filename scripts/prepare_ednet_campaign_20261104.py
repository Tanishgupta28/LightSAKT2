"""Audit actual EdNet features and freeze the fifteen-run core comparison."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.ednet_replacement_data import ROOT, WORK, OUT, RAW, DS, write, canonical_tags
from scripts.question_difficulty_inputs import difficulty_features
from scripts.professor_data_20261030 import heldout_difficulty
from scripts.architecture_study import digest
from scripts.pilot_shared_resources import observe, assess, POLICY

SOURCE = ROOT.parent / "LightSAKT2"
PLAN_DIR = ROOT / "experiments/ednet_replacement_20261104"
SEEDS = [42, 123, 2026, 3407, 7777]


def audit_data():
    stats = json.loads((WORK / f"reports/{DS}_statistics.json").read_text())
    specs = json.loads((WORK / "data/auxiliary/feature_spec.json").read_text())
    path = WORK / f"data/processed/{DS}/interactions.csv"
    assert digest(path) == stats["processed_sha256"]
    frame = pd.read_csv(path, dtype={"student": str, "problem_id": str})
    assert not frame.duplicated(["student", "solving_id"]).any()
    meta = pd.read_csv(RAW / "questions.csv").set_index("question_id")
    assert frame.problem_id.isin(meta.index).all()
    assert (frame.skill == frame.problem_id.map(meta.tags.map(canonical_tags))).all()
    sets = {s: set(frame.loc[frame.split == s, "student"]) for s in ["train", "validation", "test"]}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    training = frame[frame.split == "train"]
    vocab = json.loads((WORK / f"data/processed/{DS}/vocabulary.json").read_text())
    assert set(vocab) == set(training.skill)
    np.testing.assert_array_equal(frame.q, frame.skill.map(vocab).fillna(0).astype(int))
    tv = frame[frame.split.isin(["train", "validation"])].copy()
    original = difficulty_features(tv)
    changed = tv.copy(); changed.loc[changed.split == "validation", "label"] = 1 - changed.loc[changed.split == "validation", "label"]
    np.testing.assert_allclose(original, difficulty_features(changed), atol=1e-12, rtol=0)
    student = training.student.value_counts().index[0]
    own = (tv.student == student).to_numpy()
    changed = tv.copy(); changed.loc[own, "label"] = 1 - changed.loc[own, "label"]
    np.testing.assert_allclose(original[own], difficulty_features(changed)[own], atol=1e-12, rtol=0)
    changed = tv.copy(); mask = (changed.split == "train") & ~own
    changed.loc[mask, "label"] = 1 - changed.loc[mask, "label"]
    responsive = float(np.max(np.abs(original[own] - difficulty_features(changed)[own])))
    assert responsive > 0
    for name, rows, values in [("train_validation", tv, original),
                               ("test", frame[frame.split == "test"], heldout_difficulty(training, frame[frame.split == "test"][["q", "problem_id"]]))]:
        sidecar_path = Path(specs[name + "_feature_path"])
        assert digest(sidecar_path) == specs[name + "_feature_sha256"]
        stored = json.loads(sidecar_path.read_text())["students"]
        for user, group in rows.assign(value=values).groupby("student", sort=True):
            np.testing.assert_allclose(group.value, stored[user]["values"], atol=1e-12, rtol=0)
    lengths = training.groupby("student").size()
    chunks = int(sum(math.ceil(max(0, n - 1) / 100) for n in lengths))
    audit = {"status": "PASS", "student_disjoint": True, "authentic_metadata_item_and_tag_alignment": True,
        "vocabulary_train_only": True, "own_student_label_flip_invariant": True,
        "all_validation_label_flip_invariant": True, "own_student_records_tested": int(own.sum()),
        "other_students_positive_control_maximum_change": responsive,
        "frozen_features_independently_reproduced": True, "training_chunks": chunks,
        "training_batches": math.ceil(chunks / 64), "training_question_aggregates": int(training.problem_id.nunique()),
        "unseen_question_fraction": {s: float((~frame.loc[frame.split == s, "problem_id"].isin(set(training.problem_id))).mean()) for s in ["validation", "test"]}}
    write(OUT / "real_data_feature_audit.json", audit)
    return stats, specs, audit


def main():
    target = PLAN_DIR / "plan.json"
    if target.exists():
        raise RuntimeError("Frozen replacement plan already exists; never regenerate or silently change it")
    stats, specs, audit = audit_data()
    protocol = json.loads((PLAN_DIR / "protocol.json").read_text())
    entries = []
    for seed in SEEDS:
        for model in ["SAKT", "LightSAKT", "LightSAKTQuestion"]:
            model_config = ({"n": 100, "d": 64, "num_attn_heads": 2, "dropout": .1} if model == "SAKT" else
                            {"max_seq_len": 100, "embed_dim": 64, "num_heads": 2, "dropout": .1})
            config = {"run_id": f"{DS}-{model}-seed{seed}-20261104-v1", "dataset": DS, "model": model,
                "group": "replacement_three_model_20261104", "seed": seed, "device": "cuda",
                "seq_len": 100, "batch_size": 64, "learning_rate": .001, "betas": [.9, .999],
                "epsilon": 1e-8, "weight_decay": 0., "max_epochs": 50, "patience": 5,
                "evaluate_test": False, "model_config": model_config}
            if model == "LightSAKTQuestion":
                config["feature_path"] = specs["train_validation_feature_path"]
                config["question_difficulty"] = {k: specs[k] for k in ["algorithm", "smoothing", "processed_sha256", "split_sha256", "raw_sha256", "metadata_sha256"]}
                config["question_difficulty"]["feature_sha256"] = specs["train_validation_feature_sha256"]
            path = PLAN_DIR / "configs" / f"{model}-{seed}.json"; write(path, config)
            entries.append({"seed": seed, "model": model, "run_id": config["run_id"], "config": str(path),
                            "config_sha256": digest(path), "work": str(WORK), "action": "new_core_run"})
    # Separate, prepared optional control uses the already implemented SAKTQuestion class.
    controls = []
    for seed in SEEDS:
        config = json.loads((PLAN_DIR / "configs" / f"SAKT-{seed}.json").read_text())
        config.update(model="SAKTQuestion", group="prepared_feature_matched_control", run_id=f"{DS}-SAKTQuestion-seed{seed}-UNLAUNCHED")
        config["feature_path"] = specs["train_validation_feature_path"]
        config["question_difficulty"] = {"algorithm": specs["algorithm"], "smoothing": 10., "feature_sha256": specs["train_validation_feature_sha256"]}
        config["experimental_implementation_sha256"] = digest(ROOT / "scripts/feature_matched_sakt.py")
        path = PLAN_DIR / "prepared_controls" / f"SAKTQuestion-{seed}.json"; write(path, config)
        controls.append({"seed": seed, "config": str(path), "config_sha256": digest(path), "status": "PREPARED_NOT_LAUNCHED"})
    hold = {}
    for root in [SOURCE, ROOT]:
        for name in ["USER_TRAINING_HOLD.json", "PAUSED_BY_USER.json"]:
            path = root / "experiments" / name
            if path.exists():
                hold[str(path)] = digest(path)
    protected = dict(json.loads((ROOT / "experiments/professor_campaign_20261030/plan.json").read_text())["protected_sha256"])
    for path, value in protected.items():
        assert digest(path) == value, path
    for path, value in json.loads((OUT / "preserved_report_fingerprints.json").read_text()).items():
        assert digest(ROOT / path) == value
        protected[str(ROOT / path)] = value
    sample = observe(protocol["gpu_uuid"]); assessment = assess(sample, admission=True)
    scientific_files = [ROOT / "scripts/ednet_replacement_data.py", ROOT / "scripts/professor_data_20261030.py",
                        ROOT / "scripts/question_difficulty_inputs.py", ROOT / "scripts/pilot_metric_extensions.py",
                        ROOT / "scripts/pilot_shared_resources.py", ROOT / "scripts/run_ednet_campaign_20261104.py"]
    data_files = [WORK / f"data/processed/{DS}/interactions.csv", WORK / f"data/splits/{DS}.json",
                  WORK / f"reports/{DS}_statistics.json", Path(specs["train_validation_feature_path"])]
    plan = {"status": "FROZEN_REPLACEMENT_CAMPAIGN_20261104", "authorization": "Current user explicitly authorizes missing fifteen replacement-dataset core experiments after authentic data and safe shared allocation verification.",
        "entries": entries, "prepared_controls": controls, "dataset": DS, "data_stats": stats,
        "feature_spec": specs, "protocol_sha256": digest(PLAN_DIR / "protocol.json"),
        "scientific_code_sha256": {str(p): digest(p) for p in scientific_files},
        "data_binding_sha256": {str(p): digest(p) for p in data_files}, "hold_sha256": hold,
        "protected_sha256": protected, "gpu_uuid": protocol["gpu_uuid"], "shared_policy": POLICY,
        "maximum_new_core_runs": 15, "maximum_total_window_seconds": 13500, "window_seconds": 900,
        "automatic_retry": False, "concurrency": 1, "allocation": {"type": "existing authorized shared MIG plus current user replacement-run scope",
        "sample": sample, "assessment": assessment, "exclusive_claim": False, "HiTSKT": "ON HOLD; untouched"},
        "source_manifest_sha256": digest(OUT / "preserved_report_fingerprints.json"), "feature_audit": audit}
    write(target, plan)
    print("FROZEN fifteen core configs, five unlaunched controls; real feature audit PASS; shared admission PASS.")


if __name__ == "__main__":
    main()

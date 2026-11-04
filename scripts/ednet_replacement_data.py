"""Deterministic, genuine-item EdNet-KT1 preprocessing; no training or tuning."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from src.datasets.preprocess import split_students, encode_records, sha256
from scripts.question_difficulty_inputs import difficulty_features
from scripts.professor_data_20261030 import heldout_difficulty

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".tools/ednet_replacement_20261104"
RAW = BASE / "raw"
WORK = BASE / "workspace"
OUT = ROOT / "reports/ednet_replacement_20261104"
DS = "EdNetKT1_5k_first"
SUBSET_SEED = 20261104
SPLIT_SEED = 42
POPULATION = 5000


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_tags(value):
    if pd.isna(value) or str(value).strip() in {"", "-1"}:
        return "<MISSING_TAGS>"
    tags = [int(x) for x in str(value).split(";")]
    if any(x < 0 for x in tags):
        raise ValueError("Invalid mixed/negative metadata tags")
    return ";".join(map(str, sorted(set(tags))))


def clean_student(raw, student, metadata):
    required = ["timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"]
    assert list(raw.columns) == required, list(raw.columns)
    raw = raw.copy(); raw["source_row"] = np.arange(len(raw))
    counts = {"raw_rows": len(raw), "exact_duplicates_removed": int(raw.duplicated(subset=required).sum())}
    raw = raw.drop_duplicates(subset=required, keep="first").copy()
    for key in ["timestamp", "solving_id"]:
        raw[key] = pd.to_numeric(raw[key], errors="raise")
        if raw[key].isna().any() or (raw[key] < 0).any() or ((raw[key] % 1) != 0).any():
            raise ValueError(f"Unusable {key} for chronological session ordering")
        raw[key] = raw[key].astype("int64")
    raw = raw.sort_values(["timestamp", "source_row"], kind="stable")
    sessions = raw.groupby("solving_id", sort=False).timestamp.agg(["min", "max"])
    # A session's question presentations must not interleave another session.
    if len(sessions) > 1:
        if (sessions["min"].to_numpy()[1:] <= sessions["max"].to_numpy()[:-1]).any():
            raise ValueError("Overlapping/tied solving sessions; chronology cannot be established")
    # Pick using presentation metadata only, BEFORE inspecting response validity.
    selected = raw.drop_duplicates("solving_id", keep="first").copy()
    counts["sessions"] = len(selected)
    counts["other_questions_in_session_excluded"] = len(raw) - len(selected)
    selected = selected.merge(metadata, on="question_id", how="left", validate="many_to_one", indicator=True)
    missing = selected["_merge"] != "both"
    counts["missing_question_metadata_excluded"] = int(missing.sum())
    selected = selected[~missing].copy()
    valid = selected.user_answer.isin(["a", "b", "c", "d"])
    valid &= ~((selected.part == 2) & (selected.user_answer == "d"))
    counts["invalid_answer_excluded"] = int((~valid).sum())
    selected = selected[valid].copy()
    selected["label"] = (selected.user_answer == selected.correct_answer).astype("int8")
    selected["student"] = student
    selected["order"] = selected.timestamp
    selected["skill"] = selected.tags.map(canonical_tags)
    selected["problem_id"] = selected.question_id
    assert selected.solving_id.is_unique
    assert selected.order.is_monotonic_increasing
    counts["retained_rows"] = len(selected)
    return selected[["student", "problem_id", "skill", "label", "order", "source_row", "solving_id"]], counts


def prepare():
    OUT.mkdir(exist_ok=True)
    final = WORK / f"reports/{DS}_statistics.json"
    if final.exists():
        result = json.loads(final.read_text())
        assert sha256(WORK / f"data/processed/{DS}/interactions.csv") == result["processed_sha256"]
        print("Reusing completed preprocessing; no population/split change", flush=True)
        return result
    meta = pd.read_csv(RAW / "questions.csv", dtype={"question_id": str})
    assert meta.question_id.is_unique and meta.question_id.str.fullmatch(r"q\d+").all()
    assert meta.correct_answer.isin(["a", "b", "c", "d"]).all()
    assert meta.part.isin(range(1, 8)).all()
    assert not ((meta.part == 2) & (meta.correct_answer == "d")).any()
    meta = meta[["question_id", "bundle_id", "correct_answer", "part", "tags"]]
    archive = RAW / "EdNet-KT1.zip"
    with zipfile.ZipFile(archive) as z:
        info = [m for m in z.infolist() if re.fullmatch(r"(?:KT1/)?u\d+\.csv", m.filename)]
        members = {Path(m.filename).stem: m for m in info}
        assert len(members) == len(info) == 784309
        students = sorted(members)
        selected = np.random.default_rng(SUBSET_SEED).choice(students, size=POPULATION, replace=False).tolist()
        selected = sorted(selected)
        population = {"algorithm": "numpy.default_rng.choice without replacement over sorted authentic ZIP user filenames",
                      "subset_seed": SUBSET_SEED, "selected_students": selected, "pool_students": len(students),
                      "criterion": "No label, length, performance or test-score criterion; first question per solving session chosen by timestamp/source-row"}
        write(WORK / "data/selection/selected_students.json", population)
        # ZIP member reads check CRC for every selected student's complete file.
        frames = []; totals = {}; identities = []
        for i, student in enumerate(selected):
            m = members[student]
            raw = pd.read_csv(io.BytesIO(z.read(m)), dtype={"question_id": str, "user_answer": str})
            frame, counts = clean_student(raw, student, meta)
            frames.append(frame)
            identities.append({"student": student, "zip_member": m.filename, "crc32": m.CRC,
                               "uncompressed_bytes": m.file_size, **counts})
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            if (i + 1) % 1000 == 0:
                print("PREPROCESS students", i + 1, "retained", totals["retained_rows"], flush=True)
        structural = {"zip_members_total": len(z.infolist()), "authentic_student_files": len(members),
                      "selected_student_file_crc_checks": len(selected), "compressed_bytes": archive.stat().st_size,
                      "all_student_uncompressed_bytes": sum(m.file_size for m in info), "central_directory_duplicate_students": 0}
    frame = pd.concat(frames, ignore_index=True).sort_values(["student", "order", "source_row"], kind="stable")
    assert frame.problem_id.isin(set(meta.question_id)).all()
    assert frame.label.isin([0, 1]).all()
    assignments = split_students(frame.student, SPLIT_SEED)
    encoded, vocabulary = encode_records(frame, assignments)
    for a, first in enumerate(assignments.values()):
        for second in list(assignments.values())[:a]:
            assert not (set(first) & set(second))
    directory = WORK / f"data/processed/{DS}"; directory.mkdir(parents=True, exist_ok=True)
    path = directory / "interactions.csv"
    encoded.to_csv(path, index=False)
    write(directory / "vocabulary.json", vocabulary)
    split = WORK / f"data/splits/{DS}.json"; write(split, assignments)
    write(WORK / "data/selection/source_members.json", identities)
    stats = {"dataset": DS, "display_name": "EdNet-KT1: deterministic 5k students, first question per solving session",
        "source_url": "https://github.com/riiid/ednet", "raw_sha256": sha256(archive),
        "metadata_sha256": sha256(RAW / "questions.csv"), "metadata_archive_sha256": sha256(RAW / "EdNet-Contents.zip"),
        "selection_sha256": sha256(WORK / "data/selection/selected_students.json"),
        "source_members_sha256": sha256(WORK / "data/selection/source_members.json"),
        "subset_seed": SUBSET_SEED, "split_seed": SPLIT_SEED, "selected_student_files": POPULATION,
        "active_students": int(encoded.student.nunique()), "students_with_at_least_two_retained_rows": int((encoded.groupby("student").size() >= 2).sum()),
        "clean_interactions": len(encoded), "unique_authentic_questions": int(encoded.problem_id.nunique()),
        "question_metadata_rows": len(meta), "skill_semantics": "sorted unique full tagset; no multiskill row explosion",
        "unknown_skill_semantics": "training-only vocabulary,UNK0; missing tags explicit category; authentic item IDs retained separately",
        "num_q": len(vocabulary) + 1, "unique_skill_identifiers": int(encoded.skill.nunique()),
        "split_students": {k: len(v) for k, v in assignments.items()},
        "split_interactions": {k: int((encoded.split == k).sum()) for k in assignments},
        "split_targets": {k: int(sum(max(0, len(g) - 1) for _, g in encoded[encoded.split == k].groupby("student"))) for k in assignments},
        "split_unknown_skills": {k: int(((encoded.split == k) & (encoded.q == 0)).sum()) for k in assignments},
        "split_sha256": sha256(split), "processed_sha256": sha256(path), "student_overlap": 0,
        "filter_counts": totals, "archive_integrity": structural,
        "causal_session_policy": "Keep first presented question per solving_id before inspecting answers; other same-session responses never enter model history; preserve earlier-session order",
        "label_rule": "recorded user_answer equals official correct_answer; key and raw answers are never predictor inputs",
        "repeated_interactions": "Authentic question repeated in different sessions retained; only exact duplicate exported rows removed",
        "temporal_scope": "Offline student-disjoint KT on session-ordered retained events; no global chronological-deployment claim",
        "license": "EdNet CC BY-NC4.0 research; raw data/model binaries not committed"}
    tv = encoded[encoded.split.isin(["train", "validation"])].copy()
    tv["difficulty"] = difficulty_features(tv, smoothing=10.)
    training = encoded[encoded.split == "train"]
    test = encoded[encoded.split == "test"].copy()
    test["difficulty"] = heldout_difficulty(training, test[["q", "problem_id"]], smoothing=10.)
    specs = {"algorithm": "question-vs-skill-logit-loso-v1", "smoothing": 10., "dataset": DS,
             "processed_sha256": stats["processed_sha256"], "split_sha256": stats["split_sha256"],
             "raw_sha256": stats["raw_sha256"], "metadata_sha256": stats["metadata_sha256"],
             "question_identity": "authentic question_id from official EdNet Questions metadata"}
    for name, rows in [("train_validation", tv), ("test", test)]:
        sidecar = {**specs, "students": {s: {"split": g.split.iloc[0], "values": g.difficulty.tolist()}
                                            for s, g in rows.groupby("student", sort=True)}}
        dest = WORK / f"data/auxiliary/{DS}-{name}.json"; write(dest, sidecar)
        specs[name + "_feature_path"] = str(dest)
        specs[name + "_feature_sha256"] = sha256(dest)
    write(WORK / "data/auxiliary/feature_spec.json", specs)
    write(final, stats)
    write(OUT / "dataset_statistics.json", stats)
    write(OUT / "feature_specification.json", specs)
    print(json.dumps(stats, indent=2), flush=True)
    return stats


if __name__ == "__main__":
    prepare()

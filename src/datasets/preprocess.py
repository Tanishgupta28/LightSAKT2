"""Fresh raw-data preprocessing; no historical pickle caches are consumed."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "student-disjoint-v1"
SOURCES = {
    "ASSIST2009": "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
    "ASSIST2015": "https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data",
    "Algebra2005": "https://www.kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_students(students, seed=42):
    students = sorted(set(map(str, students)))
    if len(students) < 10:
        raise ValueError("At least ten students required for 70/10/20 splitting")
    order = np.random.default_rng(seed).permutation(students).tolist()
    ntrain, nval = int(.7*len(order)), int(.1*len(order))
    return {"train": order[:ntrain], "validation": order[ntrain:ntrain+nval], "test": order[ntrain+nval:]}


def encode_records(frame, assignments):
    # UNK=0; vocabulary uses identifiers in training students only, never heldout labels.
    training = frame[frame.student.isin(assignments["train"])]
    vocab = {skill: i+1 for i, skill in enumerate(sorted(training.skill.unique()))}
    encoded = frame.copy()
    encoded["q"] = encoded.skill.map(vocab).fillna(0).astype("int32")
    split_of = {s: split for split, students in assignments.items() for s in students}
    encoded["split"] = encoded.student.map(split_of)
    assert encoded.split.notna().all()
    return encoded, vocab


def prepare(dataset, path, root=".", seed=42):
    root, path = Path(root), Path(path)
    sep = "\t" if dataset == "Algebra2005" else ","
    raw = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
    stats = {"dataset": dataset, "raw_rows": len(raw), "columns": raw.columns.tolist(),
             "raw_sha256": sha256(path), "source_url": SOURCES[dataset], "raw_path": str(path),
             "preprocessing_version": VERSION, "split_seed": seed,
             "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
             "missing_by_column": raw.isna().sum().astype(int).to_dict()}
    raw = raw.drop_duplicates().copy()
    if dataset == "ASSIST2009":
        columns = ["user_id", "order_id", "skill_name", "correct"]
        raw = raw.dropna(subset=columns)
        # Multiple tags attached to one answered problem are one interaction, not repeated labels.
        groups = raw.groupby(["user_id", "order_id"], sort=False)
        if (groups.correct.nunique() > 1).any():
            raise ValueError("Conflicting correctness for identical attempt")
        frame = groups.agg(skill=("skill_name", lambda s: "~~".join(sorted(set(map(str, s))))),
                           label=("correct", "first")).reset_index()
        frame = frame.rename(columns={"user_id": "student", "order_id": "order"})
        stats["collapsed_multitag_rows"] = len(raw)-len(frame)
        stats["identifier_semantics"] = "skill_name; simultaneous tags collapsed to sorted composite"
    elif dataset == "ASSIST2015":
        frame = raw.rename(columns={"user_id": "student", "log_id": "order", "sequence_id": "skill", "correct": "label"})
        frame = frame.dropna(subset=["student", "order", "skill", "label"]).copy()
        stats["identifier_semantics"] = "sequence_id (skill-builder); no exercise ID in four-column release"
    elif dataset == "Algebra2005":
        frame = raw.rename(columns={"Anon Student Id": "student", "First Transaction Time": "order", "KC(Default)": "skill", "Correct First Attempt": "label"})
        frame = frame.dropna(subset=["student", "order", "skill", "label"]).copy()
        frame["order"] = pd.to_datetime(frame.order, errors="raise")
        stats["identifier_semantics"] = "KC(Default) full composite string; original training subset only"
        stats["ordering_semantics"] = "First Transaction Time: first-attempt response availability, not step start; stable source-row ties"
    else:
        raise ValueError(dataset)
    stats["nonbinary_rows_excluded"] = int((~frame.label.isin([0, 1])).sum())
    frame = frame[frame.label.isin([0, 1])].copy()
    frame["label"] = frame.label.astype("int8")
    frame["student"] = frame.student.astype(str)
    frame["skill"] = frame.skill.astype(str)
    frame["source_row"] = np.arange(len(frame))
    frame = frame.sort_values(["student", "order", "source_row"], kind="stable")
    assignments = split_students(frame.student, seed)
    encoded, vocab = encode_records(frame, assignments)
    directory = root/"data/processed"/dataset
    directory.mkdir(parents=True, exist_ok=True)
    split_path = root/"data/splits"/f"{dataset}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(assignments, indent=2))
    encoded[["student", "q", "label", "order", "source_row", "split", "skill"]].to_csv(directory/"interactions.csv", index=False)
    (directory/"vocabulary.json").write_text(json.dumps(vocab, indent=2))
    stats.update({"clean_interactions": len(frame), "students": int(frame.student.nunique()),
                  "unique_skill_identifiers": int(frame.skill.nunique()), "num_q": len(vocab)+1,
                  "split_students": {k: len(v) for k, v in assignments.items()},
                  "split_interactions": encoded.groupby("split").size().astype(int).to_dict(),
                  "split_unknown_interactions": encoded[encoded.q == 0].groupby("split").size().astype(int).to_dict(),
                  "split_sha256": sha256(split_path), "processed_sha256": sha256(directory/"interactions.csv"),
                  "student_overlap": 0, "order": "stable ascending order, source-row tie break",
                  "normalization": "none; only exact binary labels retained",
                  "license": "Publisher public research availability; no explicit redistribution license established. Raw files excluded from Git."})
    (root/"reports"/f"{dataset}_statistics.json").write_text(json.dumps(stats, indent=2))
    return stats

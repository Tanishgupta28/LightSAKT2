"""Full-builder leakage tests without accessing actual held-out test labels."""
import json
import numpy as np
import pandas as pd
from scripts.question_difficulty_inputs import build_features, difficulty_features


def synthetic_source(root, flip_heldout=False):
    rows = []
    for student, split in [("a", "train"), ("b", "train"), ("v", "validation"), ("t", "test")]:
        for step in range(4):
            label = int((student == "a") or (step % 2))
            if flip_heldout and split != "train":
                label = 1 - label
            rows.append(dict(student=student, q=1, label=label, order=len(rows),
                             source_row=len(rows), split=split, skill="s", problem_id=step % 2))
    frame = pd.DataFrame(rows)
    (root / "data/processed/ASSIST2009").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True); (root / "data/splits").mkdir(parents=True)
    frame.drop(columns="problem_id").to_csv(root / "data/processed/ASSIST2009/interactions.csv", index=False)
    raw = frame.rename(columns={"student": "user_id", "order": "order_id", "label": "correct"})
    raw[["user_id", "order_id", "problem_id", "correct"]].to_csv(root / "data/raw/ASSIST2009.csv", index=False)
    (root / "data/splits/ASSIST2009.json").write_text(json.dumps({"train": ["a", "b"], "validation": ["v"], "test": ["t"]}))


def test_full_builder_ignores_validation_test_and_raw_correctness(tmp_path):
    records = []
    for flipped in [False, True]:
        source = tmp_path / str(flipped); work = source / "work"
        synthetic_source(source, flipped)
        build_features(source, work)
        records.append(json.loads((work / "data/auxiliary/ASSIST2009-question-difficulty.json").read_text()))
    assert records[0]["students"] == records[1]["students"]
    assert "t" not in records[0]["students"]
    # Whole-file provenance hashes correctly change, but numerical inputs do not.
    assert records[0]["processed_sha256"] != records[1]["processed_sha256"]


def test_future_own_student_labels_do_not_change_earlier_features_and_positive_control():
    f = pd.DataFrame({"student": ["a"] * 4 + ["b"] * 4 + ["v"] * 2,
                      "q": [1] * 10, "problem_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                      "label": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
                      "split": ["train"] * 8 + ["validation"] * 2})
    before = difficulty_features(f)
    changed = f.copy(); changed.loc[[2, 3], "label"] = [0, 0]
    np.testing.assert_array_equal(before[:4], difficulty_features(changed)[:4])
    # The test is not vacuous: changing another training student's responses
    # legitimately changes the fitted statistics and held-out predictions' input.
    changed = f.copy(); changed.loc[[4, 6], "label"] = [0, 0]
    assert not np.allclose(before[:4], difficulty_features(changed)[:4])

"""Independently align every retained interaction with the official raw row."""
import io
import json
import zipfile
import numpy as np
import pandas as pd
from scripts.ednet_replacement_data import RAW, WORK, OUT, DS, write


def main():
    frame = pd.read_csv(WORK / f"data/processed/{DS}/interactions.csv", dtype={"student": str})
    groups = dict(tuple(frame.groupby("student", sort=True)))
    members = json.loads((WORK / "data/selection/source_members.json").read_text())
    meta = pd.read_csv(RAW / "questions.csv").set_index("question_id")
    checked = 0
    with zipfile.ZipFile(RAW / "EdNet-KT1.zip") as archive:
        for entry in members:
            raw = pd.read_csv(io.BytesIO(archive.read(entry["zip_member"])))
            rows = groups[entry["student"]]
            aligned = raw.iloc[rows.source_row.to_numpy()]
            np.testing.assert_array_equal(aligned.question_id, rows.problem_id)
            np.testing.assert_array_equal(aligned.timestamp, rows.order)
            np.testing.assert_array_equal(aligned.solving_id, rows.solving_id)
            answer = aligned.question_id.map(meta.correct_answer)
            np.testing.assert_array_equal((aligned.user_answer.to_numpy() == answer.to_numpy()).astype(int), rows.label)
            # Independent reconstruction of the retained source-row population,
            # without calling the preprocessing implementation.
            original_columns = list(raw.columns)
            raw["row_index"] = np.arange(len(raw))
            first = raw.drop_duplicates(original_columns).sort_values(["timestamp", "row_index"], kind="stable").drop_duplicates("solving_id")
            valid = first.question_id.isin(meta.index) & first.user_answer.isin(["a", "b", "c", "d"])
            valid &= ~((first.question_id.map(meta.part) == 2) & (first.user_answer == "d"))
            np.testing.assert_array_equal(first.loc[valid, "row_index"], rows.source_row)
            checked += len(rows)
    assert checked == len(frame) == 563469
    write(OUT / "raw_row_alignment_audit.json", {"status": "PASS", "students": len(members), "retained_rows": checked,
        "authentic_item_and_session_ids_exact": True, "chronological_timestamps_exact": True,
        "official_answer_key_correctness_labels_exact": True, "first_session_selection_before_label_validation_exact": True,
        "all_source_rows_and_exclusions_independently_reconstructed": True,
        "no_clean_student_function_reused": True, "no_training_or_model_inference": True})
    print(f"PASS independent raw-source alignment and correctness on {checked} rows, {len(members)} students")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.ednet_replacement_data import clean_student, canonical_tags
from scripts.question_difficulty_inputs import difficulty_features
from scripts.professor_data_20261030 import heldout_difficulty
from src.models.light_sakt import LightSAKT
from src.models.light_sakt_question import LightSAKTQuestion
from src.models.sakt import SAKT
from scripts.feature_matched_sakt import SAKTQuestion


def metadata():
    return pd.DataFrame({"question_id": ["q1", "q2", "q3"], "bundle_id": ["b1", "b1", "b2"],
                         "correct_answer": ["a", "b", "c"], "part": [3, 3, 2], "tags": ["3;1;3", "1", "2"]})


def raw(rows):
    return pd.DataFrame(rows, columns=["timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"])


def test_session_target_is_chosen_before_answers_and_repeats_keep_true_ids():
    data = raw([[200, 2, "q1", "b", 20], [100, 1, "q1", "a", 20],
                [101, 1, "q2", "b", 20], [100, 1, "q1", "a", 20], [300, 3, "q3", "c", 20]])
    result, counts = clean_student(data, "u123", metadata())
    assert result.problem_id.tolist() == ["q1", "q1", "q3"]
    assert result.label.tolist() == [1, 0, 1]
    assert result.student.unique().tolist() == ["u123"]
    assert result.skill.iloc[0] == "1;3"
    assert counts["exact_duplicates_removed"] == counts["other_questions_in_session_excluded"] == 1
    changed = data.copy(); changed.loc[2, "user_answer"] = "a"
    other, _ = clean_student(changed, "u123", metadata())
    pd.testing.assert_frame_equal(result, other)  # Current bundle's other answer cannot enter retained history.


def test_invalid_first_answer_is_not_replaced_with_a_later_bundle_question():
    data = raw([[100, 1, "q1", "x", 20], [101, 1, "q2", "b", 20],
                [200, 2, "q3", "d", 20], [300, 3, "q3", "c", 20]])
    result, counts = clean_student(data, "u1", metadata())
    assert result.solving_id.tolist() == [3]
    assert counts["invalid_answer_excluded"] == 2


def test_overlapping_sessions_fail_closed_and_tag_order_is_canonical():
    data = raw([[100, 1, "q1", "a", 20], [102, 1, "q2", "b", 20], [101, 2, "q3", "c", 20]])
    with pytest.raises(ValueError, match="Overlapping"):
        clean_student(data, "u1", metadata())
    assert canonical_tags("4;2;4") == "2;4"
    assert canonical_tags(np.nan) == "<MISSING_TAGS>"


def test_difficulty_excludes_all_own_labels_and_all_heldout_labels():
    frame = pd.DataFrame({"student": ["a", "a", "b", "b", "v", "v"],
                          "problem_id": ["q1", "q2", "q1", "q2", "q1", "q-new"],
                          "q": [1] * 6, "label": [0, 1, 1, 0, 1, 0],
                          "split": ["train"] * 4 + ["validation"] * 2})
    baseline = difficulty_features(frame)
    changed = frame.copy(); changed.loc[changed.student == "a", "label"] = [1, 0]
    np.testing.assert_allclose(difficulty_features(changed)[:2], baseline[:2])
    changed = frame.copy(); changed.loc[changed.split == "validation", "label"] = [0, 1]
    np.testing.assert_allclose(difficulty_features(changed), baseline)
    values = heldout_difficulty(frame[frame.split == "train"], frame.iloc[4:][["q", "problem_id"]])
    np.testing.assert_allclose(values, baseline[4:]); assert values[-1] == 0


@pytest.mark.parametrize("family", ["LightSAKT", "SAKT"])
def test_models_zero_initialization_causality_padding_and_gradients(family):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = dict(embed_dim=16, num_heads=2, dropout=0., max_seq_len=6) if family == "LightSAKT" else dict(n=6, d=16, num_attn_heads=2, dropout=0.)
    base_class, modified_class = (LightSAKT, LightSAKTQuestion) if family == "LightSAKT" else (SAKT, SAKTQuestion)
    base = base_class(8, **cfg)
    torch.manual_seed(42); modified = modified_class(8, **cfg)
    q = torch.tensor([[1, 2, 3, 4, 0, 0]]); r = torch.tensor([[0, 1, 0, 1, 0, 0]])
    query = torch.tensor([[2, 3, 4, 5, 0, 0]]); valid = q != 0
    feature = torch.tensor([[.5, -.3, .8, .4, 999., 999.]])
    expected = base(q, r, query, valid)[0]
    actual = modified(q, r, query, valid, difficulty=feature)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    modified.difficulty_scale.data.fill_(.7)
    first = modified(q, r, query, valid, difficulty=feature)[0]
    changed_r = r.clone(); changed_r[:, 2:] = 1 - changed_r[:, 2:]
    changed_f = feature.clone(); changed_f[:, 2:] *= -3
    second = modified(q, changed_r, query, valid, difficulty=changed_f)[0]
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
    assert (first[~valid] == 0).all() and torch.isfinite(first).all()
    loss = torch.nn.functional.binary_cross_entropy(first[valid], torch.tensor([1., 0., 1., 0.]))
    loss.backward()
    assert modified.difficulty_scale.grad is not None and modified.difficulty_scale.grad.abs() > 0
    assert all(torch.isfinite(p.grad).all() for p in modified.parameters() if p.grad is not None)

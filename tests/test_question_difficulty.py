import json
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.question_difficulty_inputs import difficulty_features, QuestionDifficultySequences, question_inputs
from scripts.architecture_study import digest
from scripts.pilot_metric_extensions import MetricsObserver, record_metrics
from src.models.light_sakt import LightSAKT
from src.models.light_sakt_question import LightSAKTQuestion
from src.training import engine
from test_architecture_study import batch
from test_training import fixture


def test_target_encoding_excludes_own_student_and_validation_labels():
    frame = pd.DataFrame({"student": ["a", "a", "b", "b", "c", "v", "v"],
                          "q": [1] * 7, "problem_id": [1, 2, 1, 2, 1, 1, 999],
                          "label": [1, 0, 1, 1, 0, 0, 1],
                          "split": ["train"] * 5 + ["validation"] * 2})
    expected = difficulty_features(frame)
    changed = frame.copy(); changed.loc[changed.student == "a", "label"] = 1 - changed.loc[changed.student == "a", "label"]
    np.testing.assert_array_equal(expected[:2], difficulty_features(changed)[:2])
    changed = frame.copy(); changed.loc[changed.split == "validation", "label"] = 1 - changed.loc[changed.split == "validation", "label"]
    np.testing.assert_array_equal(expected, difficulty_features(changed))
    assert expected[-1] == pytest.approx(0)  # Unseen question falls back to skill.
    invalid = frame.copy(); invalid.loc[6, "split"] = "test"
    with pytest.raises(ValueError):
        difficulty_features(invalid)


def test_identity_initialization_rng_causality_padding_and_gradients():
    torch.manual_seed(42); original = LightSAKT(7, 16, 2, .1, 100); rng = torch.get_rng_state()
    torch.manual_seed(42); model = LightSAKTQuestion(7, 16, 2, .1, 100)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, model.state_dict()[k], atol=0, rtol=0)
    data = batch(); feature = torch.randn(2, 8)
    for training in (False, True):
        original.train(training); model.train(training)
        torch.manual_seed(19); expected = original(*data[:3], data[-1])[0]
        torch.manual_seed(19); actual = model(*data[:3], data[-1], difficulty=feature)[0]
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    actual.sum().backward()
    assert model.difficulty_scale.grad.abs() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        model.difficulty_scale.fill_(.5)
    model.eval(); data[-1][0, 6:] = False; data[-1][1] = False
    before = model(*data[:3], data[-1], difficulty=feature)[0]
    dirty = [x.clone() for x in data]; f = feature.clone()
    for x in dirty[:3]:
        x[~data[-1]] = 999999
    f[~data[-1]] = float("nan")
    torch.testing.assert_close(before, model(*dirty[:3], dirty[-1], difficulty=f)[0], atol=0, rtol=0)
    assert not before[~data[-1]].any()
    for x in dirty[:3]:
        x[:, 4:6] = 0
    f[:, 4:6] += 10
    torch.testing.assert_close(before[:, :4], model(*dirty[:3], dirty[-1], difficulty=f)[0][:, :4])
    with pytest.raises(ValueError):
        model(*data[:3], data[-1], difficulty=torch.full_like(feature, float("nan")))


def setup(root):
    config = fixture(root)
    frame = pd.read_csv(root / "data/processed/toy/interactions.csv", dtype={"student": str})
    students = {s: {"split": g.split.iloc[0], "values": [float(q - 1) for q in g.q]}
                for s, g in frame[frame.split != "test"].groupby("student")}
    path = root / "data/auxiliary/ASSIST2009-question-difficulty.json"
    path.parent.mkdir(parents=True); path.write_text(json.dumps({"students": students}))
    config["model"] = "LightSAKTQuestion"
    config["question_difficulty"] = {"feature_sha256": digest(path)}
    return config, path


def test_question_pipeline_exact_resume_and_epoch_reporting(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ca, _ = setup(a); cb, path = setup(b)
    original_dataset, original_loss = engine.Sequences, engine.training_loss
    for config, root, stop in ((ca, a, None), (cb, b, 1), (cb, b, None)):
        observer = MetricsObserver()
        with record_metrics(engine, observer):
            with question_inputs(engine, root, config):
                engine.run(config, root, stop_after_epoch=stop, observer=observer)
    assert engine.Sequences is original_dataset and engine.training_loss is original_loss
    states = [torch.load(r / "checkpoints/toy/latest.pt", weights_only=False) for r in (a, b)]
    for k, v in states[0]["model"].items():
        torch.testing.assert_close(v, states[1]["model"][k], atol=0, rtol=0)
    for row in states[0]["history"]:
        assert row["resources"]["online_training_metrics"]["n_targets"] == 84
        assert row["validation"]["n_targets"] == 12
        assert {"auc", "f1", "accuracy", "bce", "precision", "recall"} <= row["validation"].keys()
    with pytest.raises(ValueError):
        QuestionDifficultySequences(b / "data/processed/toy/interactions.csv", "test", 5, path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        with question_inputs(engine, b, cb):
            pass

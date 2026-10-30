import numpy as np
import pandas as pd
import pytest
from scripts.professor_data_20261030 import heldout_difficulty, training_features
from scripts.question_difficulty_inputs import difficulty_features
from scripts.pilot_metric_extensions import record_metrics, MetricsObserver
from src.training import engine
from test_question_difficulty import setup


def test_test_transform_never_uses_query_labels_and_matches_validation_formula():
    train = pd.DataFrame(dict(student=["a", "a", "b", "b"], q=[1, 1, 1, 1],
                             problem_id=[1, 2, 1, 2], label=[1, 0, 1, 1], split=["train"] * 4))
    query = pd.DataFrame(dict(student=["v", "v"], q=[1, 1], problem_id=[1, 999], label=[0, 1], split=["validation"] * 2))
    expected = heldout_difficulty(train, query)
    query.label = [1, 0]
    np.testing.assert_array_equal(expected, heldout_difficulty(train, query))
    reference = difficulty_features(pd.concat([train, query], ignore_index=True))[-2:]
    np.testing.assert_allclose(expected, reference, atol=1e-12)
    assert expected[-1] == pytest.approx(0, abs=1e-12)


def test_generic_training_adapter_has_original_mask_and_metrics(tmp_path):
    config, feature_path = setup(tmp_path)
    config["feature_path"] = str(feature_path)
    original_dataset, original_loss = engine.Sequences, engine.training_loss
    observer = MetricsObserver()
    with record_metrics(engine, observer):
        with training_features(engine, tmp_path, config):
            result = engine.run(config, tmp_path, observer=observer)
    assert engine.Sequences is original_dataset and engine.training_loss is original_loss
    assert result["test"] is None and result["validation"]["n_targets"] == 12
    assert {"auc", "f1", "accuracy", "bce", "precision", "recall"} <= result["validation"].keys()

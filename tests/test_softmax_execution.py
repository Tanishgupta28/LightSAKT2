import json
from pathlib import Path

import pytest
import torch

from scripts.architecture_pilot import CONTROL, STAGES, PilotBlocked, check_holds
from scripts.architecture_study import digest
from scripts.pilot_metric_extensions import MetricsObserver, extended_metrics, record_metrics
from src.training import engine
from test_architecture_pilot import grant
from test_training import fixture


def test_softmax_precision_recall_are_fixed_threshold_metrics():
    result = extended_metrics([1, 1, 0, 0], [.9, .3, .6, .1])
    assert result["precision"] == result["recall"] == result["f1"] == .5
    assert result["accuracy"] == .5 and result["auc"] == .75
    result = extended_metrics([0, 0], [.1, .2])
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_softmax_scope_is_single_arm_and_cannot_start_recency(tmp_path):
    assert STAGES["softmax"] == ("one_layer_softmax",)
    source, project = tmp_path / "source", tmp_path / "project"
    markers = []
    for root in (source, project):
        marker = root / "experiments/USER_TRAINING_HOLD.json"
        marker.parent.mkdir(parents=True)
        marker.write_text('{"status":"USER_TRAINING_HOLD_ACTIVE"}')
        markers.append(marker)
    config = project / CONTROL / "configs/one_layer_softmax.json"
    config.parent.mkdir(parents=True); config.write_text('{"attention_type":"softmax"}')
    _, record = grant(tmp_path)
    record["stages"] = ["softmax"]
    record["hold_exception"] = {"stage": "softmax", "dataset": "ASSIST2009", "seed": 42,
                                "user_authorization_evidence": "synthetic test only",
                                "hold_marker_sha256": {str(p): digest(p) for p in markers},
                                "configuration_sha256": {"one_layer_softmax": digest(config)}}
    check_holds(source, project, record, "softmax")
    for forbidden in ("depth", "attention"):
        with pytest.raises(PilotBlocked):
            check_holds(source, project, record, forbidden)
    config.write_text('{"attention_type":"recency_softmax"}')
    with pytest.raises(PilotBlocked):
        check_holds(source, project, record, "softmax")


def test_softmax_extended_reporting_preserves_updates_and_exact_resume(tmp_path):
    roots = [tmp_path / name for name in ("plain", "reported", "resumed")]
    configs = [fixture(root) for root in roots]
    for config in configs:
        config["model"] = "LightSAKTResearch"
        config["model_config"].update(num_layers=1, attention_type="softmax")
    plain = engine.run(configs[0], roots[0])
    original_metrics, original_loss = engine.metrics, engine.training_loss
    observer = MetricsObserver()
    with record_metrics(engine, observer):
        reported = engine.run(configs[1], roots[1], observer=observer)
    observer = MetricsObserver()
    with record_metrics(engine, observer):
        engine.run(configs[2], roots[2], stop_after_epoch=1, observer=observer)
    observer = MetricsObserver()
    with record_metrics(engine, observer):
        resumed = engine.run(configs[2], roots[2], observer=observer)
    assert engine.metrics is original_metrics and engine.training_loss is original_loss
    for key, value in plain["validation"].items():
        assert reported["validation"][key] == resumed["validation"][key] == value
    assert reported["validation"] == resumed["validation"]
    states = [torch.load(root / "checkpoints/toy/latest.pt", weights_only=False) for root in roots]
    for key, value in states[0]["model"].items():
        for state in states[1:]:
            torch.testing.assert_close(value, state["model"][key], atol=0, rtol=0)
    for row in states[1]["history"]:
        train = row["resources"]["online_training_metrics"]
        assert train["n_targets"] == 84
        assert set(("auc", "accuracy", "precision", "recall", "f1", "bce")) <= train.keys()
        assert set(("auc", "accuracy", "precision", "recall", "f1", "bce")) <= row["validation"].keys()
    assert plain["test"] is reported["test"] is resumed["test"] is None

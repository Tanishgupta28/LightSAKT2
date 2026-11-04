"""Prepared control adapter, deliberately without a GPU launch command.

Uses the existing SAKTQuestion class and exactly the core frozen EdNet sidecar.
An independently defined control execution budget is still required.
"""
from contextlib import contextmanager
from scripts.architecture_study import digest
from scripts.feature_matched_sakt import SAKTQuestion
from scripts.professor_data_20261030 import training_features


@contextmanager
def prepared_control(engine, work, config):
    assert config["model"] == "SAKTQuestion" and config["evaluate_test"] is False
    assert digest(__import__("scripts.feature_matched_sakt", fromlist=["__file__"]).__file__) == config["experimental_implementation_sha256"]
    original_build, original_predict = engine.build_model, engine.predict
    def build(name, num_q, model_config):
        return SAKTQuestion(num_q, **model_config) if name == "SAKTQuestion" else original_build(name, num_q, model_config)
    def predict(model, name, batch):
        if name == "SAKTQuestion":
            q, r, qry, _, difficulty, valid = batch
            return model(q, r, qry, valid, difficulty=difficulty)[0], None
        return original_predict(model, name, batch)
    engine.build_model, engine.predict = build, predict
    try:
        with training_features(engine, work, config):
            yield
    finally:
        engine.build_model, engine.predict = original_build, original_predict

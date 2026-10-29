"""CPU-prepared SAKT control. No launch CLI; campaign budget remains unapproved.

Kept outside src so preparing the control cannot alter the running LightSAKT
experiment's source identity. Its exact bytes must be bound in control configs.
"""
from contextlib import contextmanager
import torch
from scripts.architecture_study import digest
from scripts.question_difficulty_inputs import question_inputs
from src.models.sakt import SAKT


class SAKTQuestion(SAKT):
    def __init__(self, num_q, **config):
        super().__init__(num_q=num_q, **config)
        self.difficulty_scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, q, r, qry, valid=None, difficulty=None):
        valid = torch.ones_like(q, dtype=torch.bool) if valid is None else valid
        if difficulty is None or difficulty.shape != q.shape:
            raise ValueError("Known-question difficulty B,L required")
        difficulty = difficulty.masked_fill(~valid, 0)
        if not torch.isfinite(difficulty).all():
            raise ValueError("Nonfinite question difficulty")
        # Hook only the native prediction logits. No SAKT attention/FFN edit.
        handle = self.pred.register_forward_hook(
            lambda module, args, logits: logits + self.difficulty_scale * difficulty[..., None])
        try:
            return super().forward(q, r, qry, valid)
        finally:
            handle.remove()


@contextmanager
def feature_matched_control(engine, work, config):
    expected = config.get("experimental_implementation_sha256")
    if expected is not None and digest(__file__) != expected:
        raise ValueError("Feature-matched SAKT implementation changed")
    original_build, original_predict = engine.build_model, engine.predict
    def build(name, num_q, model_config):
        return SAKTQuestion(num_q, **model_config) if name == "SAKTQuestion" else original_build(name, num_q, model_config)
    def predict(model, name, batch):
        if name == "SAKTQuestion":
            q, r, qry, target, difficulty, valid = batch
            return model(q, r, qry, valid, difficulty=difficulty)[0], None
        return original_predict(model, name, batch)
    engine.build_model, engine.predict = build, predict
    try:
        with question_inputs(engine, work, config):
            yield
    finally:
        engine.build_model, engine.predict = original_build, original_predict

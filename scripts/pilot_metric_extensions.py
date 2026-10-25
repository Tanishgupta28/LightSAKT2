"""Add pilot reporting metrics without changing the original model/trainer.

The context is local to one dedicated training process and restores bindings on
exit. Online training scores use each batch's pre-update/dropout-active outputs;
validation scores use the evaluation-mode model after the complete epoch.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib

import numpy as np

from src.evaluation.metrics import metrics
from src.training.pilot_observer import PilotObserver


def extended_metrics(target, prediction):
    result = metrics(target, prediction)
    truth = np.asarray(target) == 1
    predicted = np.asarray(prediction) >= .5
    tp = int(np.count_nonzero(truth & predicted))
    positives = int(np.count_nonzero(predicted))
    actual = int(np.count_nonzero(truth))
    result.update(precision=tp / positives if positives else 0.,
                  recall=tp / actual if actual else 0.)
    return result


class MetricsObserver(PilotObserver):
    def provenance(self):
        record = super().provenance()
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        record["evaluation_extension_sha256"] = code_hash
        record["resume_identity"]["evaluation_extension_sha256"] = code_hash
        record["training_metrics_scope"] = "online pre-update batch scores, dropout active"
        return record

    def start_epoch(self, epoch, device):
        super().start_epoch(epoch, device)
        self.targets, self.predictions = [], []

    def capture(self, prediction, target, valid):
        self.targets.append(target[valid].detach().cpu().numpy())
        self.predictions.append(prediction[valid].detach().cpu().numpy())

    def end_epoch(self, epoch, device):
        record = super().end_epoch(epoch, device)
        record["online_training_metrics"] = extended_metrics(
            np.concatenate(self.targets), np.concatenate(self.predictions))
        self.targets, self.predictions = [], []
        return record


@contextmanager
def record_metrics(engine, observer):
    original_metrics, original_loss = engine.metrics, engine.training_loss
    def loss(model, name, batch, prediction, all_predictions):
        value = original_loss(model, name, batch, prediction, all_predictions)
        observer.capture(prediction, batch[3], batch[4])
        return value
    engine.metrics, engine.training_loss = extended_metrics, loss
    try:
        yield
    finally:
        engine.metrics, engine.training_loss = original_metrics, original_loss

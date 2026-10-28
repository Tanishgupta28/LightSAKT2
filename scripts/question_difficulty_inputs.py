"""Train-only question target encoding; isolated optional six-field input adapter."""
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.architecture_study import digest
from src.datasets.sequences import Sequences


def difficulty_features(frame, smoothing=10.):
    if not set(frame.split.unique()) <= {"train", "validation"}:
        raise ValueError("Only training and validation rows permitted")
    training = frame[frame.split == "train"]
    pc = training.groupby("problem_id").label.agg(["sum", "count"])
    sc = training.groupby("q").label.agg(["sum", "count"])
    ps = training.groupby(["student", "problem_id"]).label.agg(["sum", "count"])
    ss = training.groupby(["student", "q"]).label.agg(["sum", "count"])
    result = np.zeros(len(frame), dtype=np.float64)
    for split in ("train", "validation"):
        mask = frame.split.to_numpy() == split
        rows = frame.loc[mask]
        qs = rows.q.map(sc["sum"]).fillna(0).to_numpy(dtype=float)
        qn = rows.q.map(sc["count"]).fillna(0).to_numpy(dtype=float)
        ts = rows.problem_id.map(pc["sum"]).fillna(0).to_numpy(dtype=float)
        tn = rows.problem_id.map(pc["count"]).fillna(0).to_numpy(dtype=float)
        if split == "train":
            si = pd.MultiIndex.from_frame(rows[["student", "q"]])
            pi = pd.MultiIndex.from_frame(rows[["student", "problem_id"]])
            qs -= ss.reindex(si)["sum"].to_numpy(); qn -= ss.reindex(si)["count"].to_numpy()
            ts -= ps.reindex(pi)["sum"].to_numpy(); tn -= ps.reindex(pi)["count"].to_numpy()
        skill_rate = (qs + 1) / (qn + 2)
        problem_rate = (ts + smoothing * skill_rate) / (tn + smoothing)
        result[mask] = np.log(problem_rate / (1 - problem_rate)) - np.log(skill_rate / (1 - skill_rate))
    assert np.isfinite(result).all()
    return result


def build_features(source, work):
    source, work = Path(source), Path(work)
    path = source / "data/processed/ASSIST2009/interactions.csv"
    frame = pd.read_csv(path, dtype={"student": str})
    frame = frame[frame.split.isin(["train", "validation"])].copy()
    raw_path = source / "data/raw/ASSIST2009.csv"
    raw = pd.read_csv(raw_path, encoding="latin1", usecols=["user_id", "order_id", "problem_id"],
                      dtype={"user_id": str}).rename(columns={"user_id": "student", "order_id": "order"})
    raw = raw.merge(frame[["student", "order"]], on=["student", "order"], how="inner", validate="many_to_one")
    assert raw.groupby(["student", "order"]).problem_id.nunique().max() == 1
    raw = raw.drop_duplicates(["student", "order"])
    frame = frame.merge(raw, on=["student", "order"], how="left", validate="one_to_one")
    assert frame.problem_id.notna().all()
    frame["difficulty"] = difficulty_features(frame)
    record = {"algorithm": "question-vs-skill-logit-loso-v1", "smoothing": 10.,
              "processed_sha256": digest(path), "raw_sha256": digest(raw_path),
              "split_sha256": digest(source / "data/splits/ASSIST2009.json"),
              "students": {s: {"split": g.split.iloc[0], "values": g.difficulty.tolist()}
                           for s, g in frame.groupby("student", sort=True)}}
    destination = work / "data/auxiliary/ASSIST2009-question-difficulty.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
    if destination.exists() and destination.read_text() != text:
        raise ValueError("Refusing to change frozen question features")
    destination.write_text(text)
    return {k: record[k] for k in ("algorithm", "smoothing", "processed_sha256", "raw_sha256", "split_sha256")} | {"feature_sha256": digest(destination)}


class QuestionDifficultySequences(Sequences):
    def __init__(self, path, split, seq_len, feature_path):
        if split not in {"train", "validation"}:
            raise ValueError("Question pilot refuses test dataset construction")
        super().__init__(path, split, seq_len)
        features = json.loads(Path(feature_path).read_text())["students"]
        for i, identity in enumerate(self.identities):
            item = features[identity["student"]]
            assert item["split"] == split
            start = identity["start"] + 1; n = identity["n_targets"]
            values = item["values"][start:start + n]
            assert len(values) == n
            feature = torch.zeros(seq_len, dtype=torch.float32); feature[:n] = torch.tensor(values)
            q, r, qry, target, valid = self.samples[i]
            self.samples[i] = q, r, qry, target, feature, valid


@contextmanager
def question_inputs(engine, work, config):
    feature_path = Path(work) / "data/auxiliary/ASSIST2009-question-difficulty.json"
    if digest(feature_path) != config["question_difficulty"]["feature_sha256"]:
        raise ValueError("Question feature checksum changed")
    original_dataset, original_loss = engine.Sequences, engine.training_loss
    engine.Sequences = lambda path, split, seq_len: QuestionDifficultySequences(path, split, seq_len, feature_path)
    def loss(model, name, batch, prediction, all_predictions):
        # Pass the original five-field view to the unchanged loss/reporting hook.
        return original_loss(model, name, tuple(batch[:4]) + (batch[-1],), prediction, all_predictions)
    engine.training_loss = loss
    try:
        yield
    finally:
        engine.Sequences, engine.training_loss = original_dataset, original_loss

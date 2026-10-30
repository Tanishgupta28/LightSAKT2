"""Frozen question-feature adapters for the professor comparison; no new model."""
import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.architecture_study import digest, read_json
from scripts.question_difficulty_inputs import difficulty_features
from src.datasets.sequences import Sequences


def question_frame(source, dataset):
    source = Path(source)
    if dataset == "ASSIST2015":
        raise ValueError("ASSIST2015 source has no independent problem ID; no substitute feature authorized")
    frame = pd.read_csv(source / f"data/processed/{dataset}/interactions.csv", dtype={"student": str})
    stats = read_json(source / f"reports/{dataset}_statistics.json")
    if dataset == "ASSIST2009":
        raw = pd.read_csv(source / "data/raw/ASSIST2009.csv", encoding="latin1",
                          usecols=["user_id", "order_id", "problem_id"], dtype={"user_id": str})
        raw = raw.rename(columns={"user_id": "student", "order_id": "order"})
        raw = raw.merge(frame[["student", "order"]], on=["student", "order"], validate="many_to_one")
        assert raw.groupby(["student", "order"]).problem_id.nunique().max() == 1
        frame = frame.merge(raw.drop_duplicates(["student", "order"]), on=["student", "order"], validate="one_to_one")
        semantics = "problem_id"
    elif dataset == "Algebra2005":
        # The frozen original preprocessing had no exact duplicates, missing
        # correctness or nonbinary exclusions. Missing KC rows were removed.
        assert stats["raw_exact_duplicate_rows"] == stats["nonbinary_rows_excluded"] == 0
        assert stats["missing_by_column"]["Correct First Attempt"] == 0
        cols = ["Anon Student Id", "First Transaction Time", "KC(Default)", "Problem Name", "Step Name"]
        raw = pd.read_csv(source / "data/raw/Algebra2005.tsv", sep="\t", encoding="latin1", usecols=cols)
        raw = raw.dropna(subset=["Anon Student Id", "First Transaction Time", "KC(Default)"]).reset_index(drop=True)
        assert len(raw) == len(frame) == stats["clean_interactions"]
        aligned = raw.iloc[frame.source_row.to_numpy()].reset_index(drop=True)
        assert np.array_equal(aligned["Anon Student Id"].astype(str), frame.student)
        assert np.array_equal(pd.to_datetime(aligned["First Transaction Time"]), pd.to_datetime(frame.order))
        assert np.array_equal(aligned["KC(Default)"].astype(str), frame.skill.astype(str))
        assert aligned[["Problem Name", "Step Name"]].notna().all().all()
        frame["problem_id"] = [json.dumps([str(p), str(s)], separators=(",", ":"))
                               for p, s in zip(aligned["Problem Name"], aligned["Step Name"])]
        semantics = "lossless composite (Problem Name, Step Name), target tutor-step identity"
    else:
        raise ValueError(dataset)
    assert frame.problem_id.notna().all()
    return frame, semantics


def heldout_difficulty(training, queries, smoothing=10.):
    """Transform known item/skill metadata using training labels only."""
    pc = training.groupby("problem_id").label.agg(["sum", "count"])
    sc = training.groupby("q").label.agg(["sum", "count"])
    qs = queries.q.map(sc["sum"]).fillna(0).to_numpy(dtype=float)
    qn = queries.q.map(sc["count"]).fillna(0).to_numpy(dtype=float)
    ps = queries.problem_id.map(pc["sum"]).fillna(0).to_numpy(dtype=float)
    pn = queries.problem_id.map(pc["count"]).fillna(0).to_numpy(dtype=float)
    skill_rate = (qs + 1) / (qn + 2)
    problem_rate = (ps + smoothing * skill_rate) / (pn + smoothing)
    return np.log(problem_rate / (1 - problem_rate)) - np.log(skill_rate / (1 - skill_rate))


def prepare_dataset(source, work, dataset, original_assist_feature=None):
    source, work = Path(source), Path(work)
    frame, semantics = question_frame(source, dataset)
    stats = read_json(source / f"reports/{dataset}_statistics.json")
    assert digest(source / f"data/processed/{dataset}/interactions.csv") == stats["processed_sha256"]
    assert digest(source / f"data/splits/{dataset}.json") == stats["split_sha256"]
    tv = frame[frame.split.isin(["train", "validation"])].copy()
    tv["difficulty"] = difficulty_features(tv, smoothing=10.)
    validation = tv[tv.split == "validation"]
    independent = heldout_difficulty(frame[frame.split == "train"], validation[["q", "problem_id"]])
    np.testing.assert_allclose(independent, validation.difficulty, atol=1e-12)
    test = frame[frame.split == "test"].copy()
    # Test outcomes are removed from the predictor transformation interface.
    test["difficulty"] = heldout_difficulty(frame[frame.split == "train"], test[["q", "problem_id"]])
    metadata = {"algorithm": "question-vs-skill-logit-loso-v1", "smoothing": 10.,
                "dataset": dataset, "question_identity": semantics,
                "processed_sha256": stats["processed_sha256"], "split_sha256": stats["split_sha256"],
                "raw_sha256": stats["raw_sha256"]}
    for name, rows in [("train_validation", tv), ("test", test)]:
        record = {**metadata, "students": {s: {"split": g.split.iloc[0], "values": g.difficulty.tolist()}
                                            for s, g in rows.groupby("student", sort=True)}}
        path = work / f"data/auxiliary/{dataset}-{name}.json"; path.parent.mkdir(parents=True, exist_ok=True)
        if name == "train_validation" and original_assist_feature is not None:
            old = read_json(original_assist_feature)
            assert old["students"] == record["students"]
            payload = Path(original_assist_feature).read_bytes()  # Preserve exact original features and SHA.
        else:
            payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if path.exists():
            assert path.read_bytes() == payload
        else:
            path.write_bytes(payload)
        metadata[name + "_feature_path"] = str(path)
        metadata[name + "_feature_sha256"] = digest(path)
    for relative in [f"data/processed/{dataset}/interactions.csv", f"data/splits/{dataset}.json", f"reports/{dataset}_statistics.json"]:
        p = work / relative; p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            assert p.resolve() == source / relative
        else:
            assert not p.exists(); p.symlink_to(source / relative)
    metadata["numerical_feature_variation"] = {split: {"rows": int(len(g)), "std": float(g.difficulty.std()),
         "nonzero_fraction": float((g.difficulty.abs() > 1e-12).mean())} for split, g in pd.concat([tv, test]).groupby("split")}
    return metadata


class FeatureSequences(Sequences):
    def __init__(self, path, split, seq_len, feature_path):
        super().__init__(path, split, seq_len)
        features = read_json(feature_path)["students"]
        for i, identity in enumerate(self.identities):
            entry = features[identity["student"]]; assert entry["split"] == split
            start, n = identity["start"] + 1, identity["n_targets"]
            values = entry["values"][start:start + n]; assert len(values) == n
            value = torch.zeros(seq_len, dtype=torch.float32); value[:n] = torch.tensor(values)
            q, r, qry, y, valid = self.samples[i]
            self.samples[i] = q, r, qry, y, value, valid


@contextmanager
def training_features(engine, work, config):
    path = Path(config["feature_path"])
    assert digest(path) == config["question_difficulty"]["feature_sha256"]
    original_dataset, original_loss = engine.Sequences, engine.training_loss
    def dataset(path_, split, seq_len):
        if split not in {"train", "validation"}:
            raise ValueError("Training adapter refuses test selection")
        return FeatureSequences(path_, split, seq_len, path)
    def loss(model, name, batch, prediction, all_predictions):
        return original_loss(model, name, tuple(batch[:4]) + (batch[-1],), prediction, all_predictions)
    engine.Sequences, engine.training_loss = dataset, loss
    try:
        yield
    finally:
        engine.Sequences, engine.training_loss = original_dataset, original_loss

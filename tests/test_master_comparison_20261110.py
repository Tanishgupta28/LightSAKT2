import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pytest
import torch
from scripts.build_master_comparison_20261110 import aggregate, FIELDS, SEEDS, ROOT, PREP, DS, read
from scripts.ednet_feature_matched_control import prepared_control
from src.training import engine


def row(seed, value=.7):
    return {"dataset": "ASSIST2009", "model": "DKT", "seed": seed,
        **{k: value for k in FIELDS}, "processed_sha256": "fixeddata", "split_sha256": "fixedsplit",
        "source_record": "fixture", "checkpoint": "fixture", "checkpoint_sha256": "fixture",
        "configuration_reference": "fixture"}


def test_partial_seed_sets_never_populate_aggregate_cells():
    for count in [0, 1, 4]:
        out = aggregate([row(s) for s in SEEDS[:count]])[0]
        assert out["experiment_status"] == "PENDING" and out["seed_count"] == count
        assert out["mean"] is None and out["standard_deviation"] is None
        for field in FIELDS:
            assert out[field] == "PENDING"
            assert out[field + "_mean"] is out[field + "_std"] is None


def test_five_seeds_use_sample_standard_deviation():
    group = [row(s, .5 + i * .1) for i, s in enumerate(SEEDS)]
    out = aggregate(group)[0]
    assert out["experiment_status"] == "VERIFIED_5_SEEDS"
    assert out["test_auc_mean"] == pytest.approx(.7)
    assert out["test_auc_std"] == pytest.approx(.15811388300841897)


def test_duplicate_aliases_and_split_drift_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([row(42), row(42)])
    second = row(123); second["split_sha256"] = "different"
    with pytest.raises(ValueError, match="Mixed"):
        aggregate([row(42), second])


def test_unexpected_population_and_invalid_measurements_rejected():
    invalid = row(42); invalid["dataset"] = "EdNet-full-unverified"
    with pytest.raises(ValueError, match="Unknown"): aggregate([invalid])
    invalid = row(42); invalid["test_auc"] = float("nan")
    with pytest.raises(ValueError, match="Invalid"): aggregate([invalid])


CASES = [(DS, n) for n in ["DKT", "DKT+", "DKVMN", "GKT", "GKT_MHA", "SAKTQuestion"]]
CASES += [(d, "SAKTQuestion") for d in ["ASSIST2009", "Algebra2005"]]


@pytest.mark.parametrize("dataset,name", CASES)
def test_prepared_native_forward_backward_causality_and_padding(dataset, name):
    torch.set_num_threads(1); torch.manual_seed(42)
    cfg = read(PREP / "prepared_configs" / f"{dataset}-{name}-42.json")
    stats = next(x for x in read(ROOT / "reports/master_comparison_20261110/dataset_provenance.json") if x["dataset"] == dataset)
    from contextlib import nullcontext
    context = prepared_control(engine, ROOT, cfg) if name == "SAKTQuestion" else nullcontext()
    original = (engine.build_model, engine.predict, engine.Sequences, engine.training_loss)
    with context:
        model = engine.build_model(name, stats["num_q"], cfg["model_config"])
        q = torch.tensor([[1, 2, 3, 0]])
        response = torch.tensor([[0, 1, 0, 0]])
        query = torch.tensor([[2, 3, 4, 0]])
        y = torch.tensor([[1, 0, 1, 0]])
        valid = torch.tensor([[True, True, True, False]])
        batch = [q, response, query, y, valid]
        if name == "SAKTQuestion": batch.insert(-1, torch.tensor([[.4, -.3, .2, 0.]]))
        model.eval()
        with torch.no_grad():
            p, _ = engine.predict(model, name, batch)
            changed = [x.clone() for x in batch]
            for i in [0, 2]: changed[i][:, 2:] = 5
            changed[1][:, 2:] = 1 - changed[1][:, 2:]
            if name == "SAKTQuestion": changed[-2][:, 2:] += 5
            other, _ = engine.predict(model, name, changed)
            torch.testing.assert_close(p[:, :2], other[:, :2], atol=1e-7, rtol=1e-7)
            padded = [x.clone() for x in batch]
            for i in [0, 2]: padded[i][:, -1] = stats["num_q"] - 1
            padded[1][:, -1] = 1
            if name == "SAKTQuestion": padded[-2][:, -1] = float("nan")
            other, _ = engine.predict(model, name, padded)
            torch.testing.assert_close(p[:, :3], other[:, :3], atol=1e-7, rtol=1e-7)
        model.train(); p, all_p = engine.predict(model, name, batch)
        loss = engine.training_loss(model, name, batch, p, all_p)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
        if name == "SAKTQuestion":
            assert model.difficulty_scale.grad is not None and torch.isfinite(model.difficulty_scale.grad)
    assert original == (engine.build_model, engine.predict, engine.Sequences, engine.training_loss)
    assert not torch.cuda.is_initialized()

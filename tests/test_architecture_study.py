import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.architecture_study import ARMS, make_plan, write_plan
from src.attention.softmax import CausalSoftmaxAttention
from src.evaluation.metrics import masked_bce
from src.models.factory import build_model, predict
from src.models.light_sakt import LightSAKT
from src.models.light_sakt_research import LightSAKTResearch


def batch(b=2, length=8):
    g = torch.Generator().manual_seed(31)
    return [torch.randint(0, 7, (b, length), generator=g),
            torch.randint(0, 2, (b, length), generator=g),
            torch.randint(0, 7, (b, length), generator=g),
            torch.randint(0, 2, (b, length), generator=g),
            torch.ones(b, length, dtype=torch.bool)]


def model(layers=2, attention="linear", length=100):
    return LightSAKTResearch(7, 16, 2, .1, length, layers, attention)


def test_one_layer_linear_is_exact_baseline_including_dropout():
    torch.manual_seed(42)
    baseline = LightSAKT(7, 16, 2, .1, 100)
    torch.manual_seed(42)
    research = model(layers=1)
    assert baseline.state_dict().keys() == research.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, research.state_dict()[key], atol=0, rtol=0)
    data = batch()
    for training in (False, True):
        baseline.train(training); research.train(training)
        torch.manual_seed(123); expected = predict(baseline, "LightSAKT", data)[0]
        torch.manual_seed(123); actual = predict(research, "LightSAKTResearch", data)[0]
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)


def test_two_layers_are_independent_sequential_blocks_and_keep_first_initialization():
    torch.manual_seed(42); baseline = LightSAKT(7, 16, 2, .1, 100)
    torch.manual_seed(42); stacked = model()
    assert stacked.num_layers == 2 and len(stacked.extra_layers) == 1
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, stacked.state_dict()[key], atol=0, rtol=0)
    assert stacked.attn.q_proj.weight.data_ptr() != stacked.extra_layers[0].attn.q_proj.weight.data_ptr()
    memory = []
    handles = [stacked.FFN_layer_norm.register_forward_hook(lambda m, args, out: memory.append(out)),
               stacked.extra_layers[0].register_forward_pre_hook(
                   lambda m, args: torch.testing.assert_close(args[1], memory[0]))]
    predict(stacked, "LightSAKTResearch", batch())[0].sum().backward()
    for handle in handles:
        handle.remove()
    assert len(memory) == 1
    for block in (stacked, stacked.extra_layers[0]):
        assert block.attn.q_proj.weight.grad.abs().sum() > 0
        assert block.FFN[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("attention", ("linear", "softmax", "recency_softmax"))
@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("b,length", ((1, 1), (2, 9), (1, 100), (1, 200)))
def test_cpu_forward_backward_optimizer_and_checkpoint(attention, layers, b, length, tmp_path):
    torch.manual_seed(42)
    m = model(layers, attention, length)
    data = batch(b, length)
    if length > 1:
        data[-1][:, -1] = False
    optimizer = torch.optim.Adam(m.parameters(), lr=.001)
    before = m.pred.weight.detach().clone()
    p = predict(m, "LightSAKTResearch", data)[0]
    assert p.shape == (b, length) and torch.isfinite(p).all()
    assert not p[~data[-1]].any()
    masked_bce(p, data[3], data[-1]).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in m.parameters())
    optimizer.step()
    assert not torch.equal(before, m.pred.weight)
    m.eval(); expected = predict(m, "LightSAKTResearch", data)[0]
    path = tmp_path / "state.pt"; torch.save(m.state_dict(), path)
    restored = model(layers, attention, length).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(expected, predict(restored, "LightSAKTResearch", data)[0], atol=0, rtol=0)


@pytest.mark.parametrize("attention", ("linear", "softmax", "recency_softmax"))
@pytest.mark.parametrize("layers", (1, 2))
def test_future_padding_batch_and_visualization_invariance(attention, layers):
    m = model(layers, attention).eval(); data = batch()
    data[-1][0, 6:] = False; data[-1][1] = False; data[-1][0, 1] = False
    expected = predict(m, "LightSAKTResearch", data)[0]
    changed = [x.clone() for x in data]
    for item in changed[:3]:
        item[~data[-1]] = 999999
    torch.testing.assert_close(expected, predict(m, "LightSAKTResearch", changed)[0])
    extended = [torch.cat((x, torch.zeros(2, 4, dtype=x.dtype)), 1) for x in data]
    torch.testing.assert_close(expected, predict(m, "LightSAKTResearch", extended)[0][:, :8])
    torch.testing.assert_close(expected[:1], predict(m, "LightSAKTResearch", [x[:1] for x in data])[0])
    future = batch()
    before = predict(m, "LightSAKTResearch", future)[0]
    for i in range(3):
        future[i][:, 4:] = (future[i][:, 4:] + 1) % (2 if i == 1 else 7)
    torch.testing.assert_close(before[:, :4], predict(m, "LightSAKTResearch", future)[0][:, :4], atol=1e-7, rtol=1e-6)
    p, weights = m(*data[:3], data[-1], True)
    torch.testing.assert_close(expected, p, atol=0, rtol=0)
    allowed = (torch.ones(8, 8, dtype=torch.bool).tril()[None, None]
               & data[-1][:, None, :, None] & data[-1][:, None, None, :])
    assert not weights.masked_select(~allowed).any()
    sums = weights.sum(-1)[data[-1][:, None].expand(-1, 2, -1)]
    torch.testing.assert_close(sums, torch.ones_like(sums))


@pytest.mark.parametrize("recency", (False, True))
def test_softmax_independent_dense_reference_and_gradient(recency):
    torch.manual_seed(42)
    a = CausalSoftmaxAttention(8, 2, 0, recency_bias=recency).double()
    xs = [torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True) for _ in range(3)]
    valid = torch.tensor([[True, False, True, True, False], [False] * 5])
    actual, weights = a(*xs, valid, True)
    projected = [proj(x.masked_fill(~valid[..., None], 0)).reshape(2, 5, 2, 4).transpose(1, 2)
                 for proj, x in zip((a.q_proj, a.k_proj, a.v_proj), xs)]
    rows = []
    for b in range(2):
        heads = []
        for h in range(2):
            outputs = []
            for t in range(5):
                if valid[b, t]:
                    indices = torch.arange(t + 1)[valid[b, :t + 1]]
                    scores = (projected[0][b, h, t] * projected[1][b, h, indices]).sum(-1) / 2
                    if recency:
                        scores = scores - torch.nn.functional.softplus(a.raw_decay[h]) * (t - indices)
                    outputs.append(torch.softmax(scores, 0) @ projected[2][b, h, indices])
                else:
                    outputs.append(torch.zeros(4, dtype=torch.double))
            heads.append(torch.stack(outputs))
        rows.append(torch.stack(heads).transpose(0, 1).reshape(5, 8))
    reference = a.out_proj(torch.stack(rows)).masked_fill(~valid[..., None], 0)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    variables = xs + list(a.parameters())
    for x, y in zip(torch.autograd.grad(actual.square().sum(), variables, retain_graph=True),
                    torch.autograd.grad(reference.square().sum(), variables)):
        torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)
    assert not weights[1].any()


def test_recency_bias_favors_recent_keys_when_content_scores_equal():
    a = CausalSoftmaxAttention(8, 2, 0, recency_bias=True)
    for projection in (a.q_proj, a.k_proj):
        torch.nn.init.zeros_(projection.weight); torch.nn.init.zeros_(projection.bias)
    _, weights = a(*[torch.zeros(1, 8, 8)] * 3, get_attn_weights=True)
    assert (weights[0, :, -1, 1:] > weights[0, :, -1, :-1]).all()


@pytest.mark.parametrize("kwargs", ({"num_layers": 0}, {"num_layers": True}, {"num_layers": 1.5},
                                     {"attention_type": "unknown"}))
def test_invalid_research_configuration(kwargs):
    with pytest.raises(ValueError):
        LightSAKTResearch(7, **kwargs)


def test_plan_preserves_baseline_settings_and_never_enqueues(tmp_path):
    plan = make_plan(Path("."))
    assert len(plan["baseline_references"]) == 15 and len(plan["proposed_runs"]) == 45
    assert plan["launches_jobs"] is False and plan["test_evaluation"] is False
    for run in plan["proposed_runs"]:
        config = run["config"]; dataset = config["dataset"]
        assert config["evaluate_test"] is False and config["model"] == "LightSAKTResearch"
        expected = plan["datasets"][dataset]["model_config"]
        assert all(config["model_config"][k] == v for k, v in expected.items())
        assert config["model_config"]["num_layers"] == ARMS[run["arm"]]["num_layers"]
        assert config["seed"] in (42, 123, 2026, 3407, 7777)
    write_plan(plan, tmp_path); write_plan(plan, tmp_path)
    assert not (tmp_path / "experiments/manifest.json").exists()
    assert len(list((tmp_path / "configs/architecture_study").glob("*.json"))) == 45
    changed = copy.deepcopy(plan); changed["selection_metric"] = "test_auc"
    with pytest.raises(ValueError, match="Refusing"):
        write_plan(changed, tmp_path)


def test_comparison_requires_all_seeds_and_rejects_test_evidence(tmp_path):
    from scripts.compare_architecture_study import comparison
    plan = make_plan(Path("."))
    rows = comparison(tmp_path, plan)
    assert sum(row["status"] == "pending" for row in rows) == 9
    selected = [entry for entry in plan["proposed_runs"]
                if entry["arm"] == "one_layer_softmax" and entry["config"]["dataset"] == "ASSIST2009"]
    for entry in selected:
        config = entry["config"]
        directory = tmp_path / "results/raw" / config["run_id"]
        directory.mkdir(parents=True)
        result = {"run_id": config["run_id"], "model": config["model"], "seed": config["seed"],
                  "test": None, "validation": {"auc": .8},
                  "processed_sha256": config["architecture_study"]["processed_sha256"],
                  "split_sha256": config["architecture_study"]["split_sha256"]}
        for name, value in (("config.json", config), ("status.json", {"status": "completed"}),
                            ("metrics.json", result)):
            (directory / name).write_text(json.dumps(value))
        current = next(row for row in comparison(tmp_path, plan)
                       if row["dataset"] == "ASSIST2009" and row["arm"] == "one_layer_softmax")
        assert (current["validation_auc_mean"] is not None) == (entry is selected[-1])
    assert current["validation_auc_mean"] == pytest.approx(.8)
    result["test"] = {"auc": .99}
    (directory / "metrics.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="validation-only"):
        comparison(tmp_path, plan)


def test_training_estimates_follow_reused_seed42_checkpoint_history():
    from scripts.benchmark_architecture_study import requirements
    estimates = requirements(Path("."))
    assert estimates["gpu_profile_performed"] is False
    assert len(estimates["rows"]) == 9
    for row in estimates["rows"]:
        assert row["recorded_baseline_epoch_seconds_mean"] > 0
        assert row["same_stopping_minutes_interval"][1] > row["same_stopping_minutes_interval"][0] > 0


@pytest.mark.parametrize("attention,layers", (("linear", 2), ("softmax", 1), ("recency_softmax", 1)))
def test_common_trainer_validation_only_and_exact_resume(tmp_path, monkeypatch, attention, layers):
    # Small synthetic fixture only; the real dataset/test records are never loaded.
    from test_training import fixture
    from src.training import engine
    from src.datasets.sequences import Sequences
    loaded = []
    def guarded_sequences(path, split, length):
        assert split != "test", "Architecture training must not construct the test loader"
        loaded.append(split)
        return Sequences(path, split, length)
    monkeypatch.setattr(engine, "Sequences", guarded_sequences)
    a, b = tmp_path / "full", tmp_path / "resumed"
    ca, cb = fixture(a), fixture(b)
    for config in (ca, cb):
        config["model"] = "LightSAKTResearch"
        config["model_config"].update(num_layers=layers, attention_type=attention)
    full = engine.run(ca, a)
    engine.run(cb, b, stop_after_epoch=1)
    resumed = engine.run(cb, b)
    assert full["test"] is None and resumed["test"] is None
    assert set(loaded) == {"train", "validation"}
    assert not (a / "results/predictions").exists()
    states = [torch.load(root / "checkpoints/toy/latest.pt", weights_only=False) for root in (a, b)]
    for key, value in states[0]["model"].items():
        torch.testing.assert_close(value, states[1]["model"][key], atol=0, rtol=0)
    history = json.loads((a / "results/raw/toy/history.json").read_text())
    assert full["validation"]["auc"] == max(row["validation"]["auc"] for row in history)
    assert full["validation"] == resumed["validation"]

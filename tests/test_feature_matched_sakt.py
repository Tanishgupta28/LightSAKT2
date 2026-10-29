import torch
from scripts.feature_matched_sakt import SAKTQuestion, feature_matched_control
from scripts.pilot_metric_extensions import MetricsObserver, record_metrics
from src.models.sakt import SAKT
from src.training import engine
from test_architecture_study import batch
from test_question_difficulty import setup


def test_matched_sakt_zero_scale_preserves_native_rng_dropout_and_parameters():
    cfg = dict(n=100, d=16, num_attn_heads=2, dropout=.1)
    torch.manual_seed(42); baseline = SAKT(7, **cfg); rng = torch.get_rng_state()
    torch.manual_seed(42); model = SAKTQuestion(7, **cfg)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
    data = batch(); difficulty = torch.randn(2, 8)
    for training in [False, True]:
        baseline.train(training); model.train(training)
        torch.manual_seed(19); expected = baseline(*data[:3], data[-1])[0]
        torch.manual_seed(19); actual = model(*data[:3], data[-1], difficulty=difficulty)[0]
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    actual.sum().backward(); assert model.difficulty_scale.grad.abs() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not model.pred._forward_hooks


def test_matched_sakt_synthetic_pipeline_exact_resume(tmp_path):
    roots = [tmp_path / "full", tmp_path / "resumed"]
    configs = []
    for root in roots:
        config, _ = setup(root); config["model"] = "SAKTQuestion"
        config["model_config"] = dict(n=5, d=8, num_attn_heads=2, dropout=.1)
        configs.append(config)
    for config, root, stop in [(configs[0], roots[0], None), (configs[1], roots[1], 1), (configs[1], roots[1], None)]:
        observer = MetricsObserver()
        with record_metrics(engine, observer):
            with feature_matched_control(engine, root, config):
                engine.run(config, root, observer=observer, stop_after_epoch=stop)
    states = [torch.load(root / "checkpoints/toy/latest.pt", weights_only=False) for root in roots]
    for name, value in states[0]["model"].items():
        torch.testing.assert_close(value, states[1]["model"][name], atol=0, rtol=0)
    assert states[0]["history"][-1]["validation"] == states[1]["history"][-1]["validation"]

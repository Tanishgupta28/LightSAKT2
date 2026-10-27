import pytest
import torch

from src.attention.linear import LinearAttention
from src.evaluation.metrics import masked_bce
from src.models.light_sakt import LightSAKT
from src.models.light_sakt_gated import LightSAKTGated
from test_architecture_study import batch


def gated():
    return LightSAKTGated(7, 16, 2, .1, 100)


def test_identity_gate_preserves_baseline_parameters_rng_forward_and_gradients():
    torch.manual_seed(42)
    baseline = LightSAKT(7, 16, 2, .1, 100)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = gated()
    assert torch.equal(rng, torch.get_rng_state())
    assert type(model.attn) is LinearAttention
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == 17
    data = batch()
    for training in (False, True):
        baseline.train(training); model.train(training)
        torch.manual_seed(123); expected, expected_weights = baseline(*data[:3], data[-1], True)
        torch.manual_seed(123); actual, weights = model(*data[:3], data[-1], True)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        torch.testing.assert_close(expected_weights, weights, atol=0, rtol=0)
    baseline.zero_grad(); model.zero_grad()
    masked_bce(expected, data[3], data[-1]).backward()
    masked_bce(actual, data[3], data[-1]).backward()
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(model.named_parameters())[name].grad, atol=0, rtol=0)
    assert model.history_gate_weight.grad.abs().sum() > 0
    assert model.history_gate_bias.grad.abs().sum() > 0


def test_learned_gate_is_causal_padding_and_batch_invariant():
    torch.manual_seed(42); model = gated().eval()
    with torch.no_grad():
        model.history_gate_weight.normal_(0, .2); model.history_gate_bias.fill_(.3)
    data = batch(); data[-1][0, 1] = False; data[-1][0, 6:] = False; data[-1][1] = False
    expected, weights = model(*data[:3], data[-1], True)
    assert torch.isfinite(expected).all() and not expected[~data[-1]].any()
    dirty = [x.clone() for x in data]
    for item in dirty[:3]:
        item[~data[-1]] = 999999
    torch.testing.assert_close(expected, model(*dirty[:3], dirty[-1])[0], atol=0, rtol=0)
    extended = [torch.cat([x, torch.zeros(2, 4, dtype=x.dtype)], 1) for x in data]
    torch.testing.assert_close(expected, model(*extended[:3], extended[-1])[0][:, :8])
    torch.testing.assert_close(expected[:1], model(*[x[:1] for x in data[:3]], data[-1][:1])[0])
    future = batch(); before = model(*future[:3], future[-1])[0]
    for i in range(3):
        future[i][:, 4:] = (future[i][:, 4:] + 1) % (2 if i == 1 else 7)
    torch.testing.assert_close(before[:, :4], model(*future[:3], future[-1])[0][:, :4], atol=1e-7, rtol=1e-6)
    allowed = (torch.ones(8, 8, dtype=torch.bool).tril()[None, None]
               & data[-1][:, None, :, None] & data[-1][:, None, None, :])
    assert not weights.masked_select(~allowed).any()


def test_gate_formula_has_correct_float64_derivatives():
    from torch.nn import functional as F
    torch.manual_seed(42)
    e = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
    w = torch.randn(1, 8, dtype=torch.double, requires_grad=True)
    b = torch.randn(1, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda e, w, b: 2 * torch.sigmoid(F.linear(e, w, b)), (e, w, b))


@pytest.mark.parametrize("bias", [-1000., 1000.])
def test_saturated_gate_is_finite_and_zero_gate_leaves_residual_path(bias):
    torch.manual_seed(42); model = gated().eval()
    with torch.no_grad():
        model.history_gate_bias.fill_(bias)
    data = batch(); p, _ = model(*data[:3], data[-1])
    assert torch.isfinite(p).all()
    masked_bce(p, data[3], data[-1]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if bias < 0:
        residual = LightSAKT(7, 16, 2, .1, 100, variant="no_attention").eval()
        residual.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("history_gate_")})
        torch.testing.assert_close(p, residual(*data[:3], data[-1])[0], atol=0, rtol=0)

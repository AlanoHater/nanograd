"""Tests for layers and loss functions."""

import numpy as np
import pytest

import nanograd as ng
from nanograd import nn
from nanograd.tensor import Tensor

from _gradcheck import check_grad


def test_linear_forward_shape():
    ng.manual_seed(0)
    layer = nn.Linear(4, 7)
    out = layer(Tensor(np.zeros((5, 4))))
    assert out.shape == (5, 7)


def test_linear_parameter_count():
    layer = nn.Linear(4, 7)
    params = layer.parameters()
    assert len(params) == 2  # weight + bias
    assert params[0].shape == (4, 7)
    assert params[1].shape == (7,)

    no_bias = nn.Linear(4, 7, bias=False)
    assert len(no_bias.parameters()) == 1


def test_sequential_forward_and_params():
    ng.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(2, 8), nn.ReLU(),
        nn.Linear(8, 8), nn.Tanh(),
        nn.Linear(8, 3),
    )
    out = model(Tensor(np.zeros((10, 2))))
    assert out.shape == (10, 3)
    # 3 Linear layers -> 3 weights + 3 biases
    assert len(model.parameters()) == 6


def test_linear_weights_receive_gradient():
    ng.manual_seed(0)
    layer = nn.Linear(3, 2)
    x = Tensor(np.ones((4, 3)))
    layer(x).sum().backward()
    assert layer.weight.grad.shape == layer.weight.data.shape
    assert np.any(layer.weight.grad != 0)
    assert np.allclose(layer.bias.grad, 4.0)  # 4 rows summed for each output


def test_mse_loss_value():
    pred = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    target = np.array([[1.0, 2.0], [3.0, 5.0]])
    loss = nn.mse_loss(pred, target)
    assert np.isclose(loss.data, 0.25)  # only one element off by 1, mean over 4


def test_cross_entropy_perfect_vs_uniform():
    # Very confident & correct -> ~0 loss
    confident = Tensor(np.array([[100.0, 0.0, 0.0]]))
    assert nn.cross_entropy(confident, [0]).data < 1e-3

    # Uniform logits -> loss == log(num_classes)
    uniform = Tensor(np.zeros((1, 4)))
    assert np.isclose(nn.cross_entropy(uniform, [0]).data, np.log(4))


def test_cross_entropy_gradient_matches_softmax_minus_onehot():
    # The analytical gradient of softmax-CE w.r.t. logits is (softmax - onehot)/N.
    rng = np.random.default_rng(3)
    logits_np = rng.standard_normal((6, 4))
    labels = rng.integers(0, 4, size=6)

    logits = Tensor(logits_np)
    nn.cross_entropy(logits, labels).backward()

    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    onehot = np.zeros_like(softmax)
    onehot[np.arange(6), labels] = 1.0
    expected = (softmax - onehot) / 6

    assert np.allclose(logits.grad, expected, atol=1e-8)


def test_batchnorm_normalizes_in_training():
    bn = nn.BatchNorm1d(4)  # gamma=1, beta=0
    x = Tensor(np.random.default_rng(0).standard_normal((64, 4)) * 3 + 5)
    out = bn(x).data
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-3)


def test_batchnorm_input_gradient():
    bn = nn.BatchNorm1d(3)
    x = np.random.default_rng(1).standard_normal((8, 3))
    check_grad(lambda t: bn(t), x)


def test_batchnorm_eval_uses_running_stats():
    bn = nn.BatchNorm1d(2)
    # Prime running stats with a few training batches.
    for _ in range(20):
        bn(Tensor(np.random.default_rng(0).standard_normal((16, 2))))
    bn.eval()
    x = Tensor(np.zeros((3, 2)))
    out1 = bn(x).data
    out2 = bn(x).data  # eval is deterministic; running stats not updated
    assert np.allclose(out1, out2)


def test_dropout_eval_is_identity():
    drop = nn.Dropout(0.5).eval()
    x = Tensor(np.ones((4, 4)))
    assert np.allclose(drop(x).data, 1.0)


def test_dropout_train_zeros_and_scales():
    ng.manual_seed(0)
    drop = nn.Dropout(0.5)
    out = drop(Tensor(np.ones((2000, 1)))).data
    frac_zero = np.mean(out == 0)
    assert 0.45 < frac_zero < 0.55          # roughly p of units dropped
    assert np.allclose(out[out != 0], 2.0)  # survivors scaled by 1/(1-p)


def test_train_eval_mode_propagates():
    model = nn.Sequential(
        nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.3)
    )
    model.eval()
    assert all(not m.training for m in model.layers)
    model.train()
    assert all(m.training for m in model.layers)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

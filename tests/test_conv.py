"""Tests for the convolutional layers."""

import numpy as np
import pytest

import nanograd as ng
from nanograd import nn, optim, utils
from nanograd.conv import Conv2d, MaxPool2d
from nanograd.tensor import Tensor

from _gradcheck import check_grad

rng = np.random.default_rng(0)


def _reference_conv(x, w, b, stride, pad):
    """Brute-force convolution used as ground truth in the forward test."""
    batch, c_in, h, w_ = x.shape
    c_out, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w_ + 2 * pad - kw) // stride + 1
    out = np.zeros((batch, c_out, out_h, out_w))
    for bi in range(batch):
        for co in range(c_out):
            for a in range(out_h):
                for d in range(out_w):
                    region = xp[bi, :, a*stride:a*stride+kh, d*stride:d*stride+kw]
                    out[bi, co, a, d] = np.sum(region * w[co]) + b[co]
    return out


def test_conv_forward_matches_reference():
    c_in, c_out, k = 3, 4, 3
    x = rng.standard_normal((2, c_in, 7, 7))
    w = rng.standard_normal((c_out, c_in, k, k))
    b = rng.standard_normal(c_out)

    conv = Conv2d(c_in, c_out, k, stride=2, padding=1)
    conv.weight = Tensor(w.reshape(c_out, -1))
    conv.bias = Tensor(b.reshape(c_out, 1))

    out = conv(Tensor(x)).data
    assert np.allclose(out, _reference_conv(x, w, b, stride=2, pad=1))


def test_conv_output_shape():
    same = Conv2d(1, 4, 3, stride=1, padding=1)
    assert same(Tensor(np.zeros((2, 1, 8, 8)))).shape == (2, 4, 8, 8)
    strided = Conv2d(1, 4, 3, stride=2, padding=1)
    assert strided(Tensor(np.zeros((2, 1, 8, 8)))).shape == (2, 4, 4, 4)


def test_conv_input_gradient():
    conv = Conv2d(2, 3, 3, stride=2, padding=1)
    check_grad(lambda x: conv(x), rng.standard_normal((2, 2, 5, 5)))


def test_conv_weight_gradient():
    conv = Conv2d(2, 3, 3, stride=1, padding=1)
    x = rng.standard_normal((2, 2, 5, 5))

    def build(w):  # w is the Tensor supplied by check_grad
        conv.weight = w
        return conv(Tensor(x))

    check_grad(build, conv.weight.data.copy())


def test_conv_bias_gradient():
    conv = Conv2d(2, 3, 3, stride=1, padding=1)
    x = rng.standard_normal((2, 2, 5, 5))

    def build(b):  # b is the Tensor supplied by check_grad
        conv.bias = b
        return conv(Tensor(x))

    check_grad(build, conv.bias.data.copy())


def test_maxpool_shape_and_gradient():
    mp = MaxPool2d(2)
    assert mp(Tensor(np.zeros((2, 3, 8, 8)))).shape == (2, 3, 4, 4)
    check_grad(lambda x: mp(x), rng.standard_normal((2, 3, 4, 4)))


def test_flatten():
    out = nn.Flatten()(Tensor(np.zeros((5, 3, 4, 4))))
    assert out.shape == (5, 48)


def _make_bars(n_samples):
    """Synthetic task: vertical bar (class 0) vs horizontal bar (class 1)."""
    gen = np.random.default_rng(1)
    x = gen.standard_normal((n_samples, 1, 8, 8)) * 0.1
    y = gen.integers(0, 2, size=n_samples)
    for i in range(n_samples):
        k = gen.integers(0, 8)
        if y[i] == 0:
            x[i, 0, :, k] += 2.0   # vertical bar
        else:
            x[i, 0, k, :] += 2.0   # horizontal bar
    return x, y


def test_cnn_learns_bars():
    ng.manual_seed(0)
    x, y = _make_bars(96)
    model = nn.Sequential(
        Conv2d(1, 8, 3, padding=1), nn.ReLU(), MaxPool2d(2),
        Conv2d(8, 16, 3, padding=1), nn.ReLU(), MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 2 * 2, 2),
    )
    opt = optim.Adam(model.parameters(), lr=3e-3)
    inputs = Tensor(x)
    for _ in range(40):
        opt.zero_grad()
        loss = nn.cross_entropy(model(inputs), y)
        loss.backward()
        opt.step()
    acc = utils.accuracy(model(inputs), y)
    assert acc > 0.95, f"CNN failed to learn bars task: {acc:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

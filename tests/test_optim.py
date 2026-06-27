"""Tests that optimizers actually minimize a simple objective."""

import numpy as np
import pytest

from nanograd import optim
from nanograd.tensor import Tensor


def _minimize(make_opt, steps=500):
    """Minimize f(w) = sum((w - target)^2) and return the final parameter."""
    target = np.array([3.0, -2.0, 0.5])
    w = Tensor(np.zeros(3))
    opt = make_opt([w])
    for _ in range(steps):
        opt.zero_grad()
        diff = w - Tensor(target, requires_grad=False)
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
    return w.data, target, float(loss.data)


def test_sgd_converges():
    w, target, _ = _minimize(lambda p: optim.SGD(p, lr=0.1))
    assert np.allclose(w, target, atol=1e-3)


def test_sgd_momentum_converges():
    w, target, _ = _minimize(lambda p: optim.SGD(p, lr=0.05, momentum=0.9))
    assert np.allclose(w, target, atol=1e-3)


def test_adam_converges():
    w, target, _ = _minimize(lambda p: optim.Adam(p, lr=0.1), steps=800)
    assert np.allclose(w, target, atol=1e-3)


def test_loss_decreases_monotonically_enough():
    target = np.array([1.0, 1.0])
    w = Tensor(np.zeros(2))
    opt = optim.SGD([w], lr=0.1)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        diff = w - Tensor(target, requires_grad=False)
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-4


def test_zero_grad_resets():
    w = Tensor(np.ones(3))
    opt = optim.SGD([w], lr=0.1)
    (w * w).sum().backward()
    assert np.any(w.grad != 0)
    opt.zero_grad()
    assert np.allclose(w.grad, 0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""End-to-end test: the whole stack should learn a non-linear dataset.

This is the strongest integration test — it exercises the autodiff engine, the
layers, the loss and the optimizer together, and only passes if the model
genuinely fits data that a linear classifier could not.
"""

import numpy as np
import pytest

import nanograd as ng
from nanograd import nn, optim, utils
from nanograd.tensor import Tensor


def test_mlp_learns_spiral():
    ng.manual_seed(42)
    x, y = utils.make_spiral(n_points=100, n_classes=3, noise=0.2)

    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 3),
    )
    opt = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    inputs, targets = Tensor(x), y
    for _ in range(300):
        opt.zero_grad()
        loss = nn.cross_entropy(model(inputs), targets)
        loss.backward()
        opt.step()

    acc = utils.accuracy(model(inputs), y)
    assert acc > 0.95, f"expected >0.95 train accuracy, got {acc:.3f}"


def test_linear_model_cannot_solve_spiral_but_mlp_can():
    # Sanity check that the task is actually non-linear: a single linear layer
    # should do clearly worse than the MLP above.
    ng.manual_seed(0)
    x, y = utils.make_spiral(n_points=100, n_classes=3, noise=0.2)

    linear = nn.Linear(2, 3)
    opt = optim.Adam(linear.parameters(), lr=1e-2)
    inputs = Tensor(x)
    for _ in range(300):
        opt.zero_grad()
        loss = nn.cross_entropy(linear(inputs), y)
        loss.backward()
        opt.step()

    acc = utils.accuracy(linear(inputs), y)
    assert acc < 0.7  # linear model is fundamentally limited here


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

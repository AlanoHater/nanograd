"""Fit a noisy sine wave with an MLP regressor (mean squared error).

Run from the repo root:

    python examples/regression_sine.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import nanograd as ng  # noqa: E402
from nanograd import nn, optim  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

EPOCHS = 2000


def main():
    ng.manual_seed(0)
    data_rng = np.random.default_rng(0)

    x = np.linspace(-1.0, 1.0, 120).reshape(-1, 1)
    y = np.sin(2.0 * np.pi * x) + 0.1 * data_rng.standard_normal(x.shape)

    model = nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )
    opt = optim.Adam(model.parameters(), lr=5e-3)

    inputs, targets = Tensor(x), Tensor(y, requires_grad=False)
    losses = []
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = nn.mse_loss(model(inputs), targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if (epoch + 1) % 250 == 0:
            print(f"epoch {epoch + 1:4d}  mse {loss.data:.5f}")

    print(f"final MSE: {losses[-1]:.5f}")

    x_line = np.linspace(-1.0, 1.0, 400).reshape(-1, 1)
    y_line = model(Tensor(x_line)).data

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    _plot.save_regression(
        x.ravel(), y.ravel(), x_line.ravel(), y_line.ravel(),
        os.path.join(assets, "regression_sine.png"),
        f"Sine regression — MLP (final MSE {losses[-1]:.4f})",
    )
    print("saved figures to assets/")


if __name__ == "__main__":
    main()

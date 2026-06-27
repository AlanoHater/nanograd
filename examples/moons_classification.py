"""Train a small MLP on the two-moons dataset (binary classification).

Run from the repo root:

    python examples/moons_classification.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import nanograd as ng  # noqa: E402
from nanograd import nn, optim, utils  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

EPOCHS = 300


def main():
    ng.manual_seed(7)
    x, y = utils.make_moons(n_samples=300, noise=0.15)

    model = nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 2),
    )
    opt = optim.Adam(model.parameters(), lr=1e-2)

    inputs = Tensor(x)
    losses = []
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = nn.cross_entropy(model(inputs), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if (epoch + 1) % 50 == 0:
            acc = utils.accuracy(model(inputs), y)
            print(f"epoch {epoch + 1:4d}  loss {loss.data:.4f}  acc {acc:.3f}")

    acc = utils.accuracy(model(inputs), y)
    print(f"final train accuracy: {acc:.3f}")

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    _plot.save_decision_boundary(
        model, x, y, os.path.join(assets, "moons_decision_boundary.png"),
        f"Two moons — MLP (train acc {acc:.2f})",
    )
    print("saved figures to assets/")


if __name__ == "__main__":
    main()

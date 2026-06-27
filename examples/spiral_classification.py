"""Train an MLP from scratch on the intertwined-spiral dataset.

Run from the repo root:

    python examples/spiral_classification.py

Saves a decision-boundary plot and a loss curve into ``assets/``.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # use the source checkout without installing

import nanograd as ng  # noqa: E402
from nanograd import nn, optim, utils  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

EPOCHS = 400


def main():
    ng.manual_seed(42)
    x, y = utils.make_spiral(n_points=100, n_classes=3, noise=0.2)

    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 3),
    )
    opt = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

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
        model, x, y, os.path.join(assets, "spiral_decision_boundary.png"),
        f"Spiral classification — MLP (train acc {acc:.2f})",
    )
    _plot.save_loss_curve(
        losses, os.path.join(assets, "spiral_loss_curve.png"),
        "Spiral classification — training loss",
    )
    print("saved figures to assets/")


if __name__ == "__main__":
    main()

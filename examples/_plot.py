"""Small Matplotlib helpers shared by the example scripts.

Kept out of the core ``nanograd`` package so the library itself only depends on
NumPy; plotting is an example-time concern.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to files

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nanograd.tensor import Tensor  # noqa: E402


def save_decision_boundary(model, x, y, path, title=""):
    """Render a 2-D classifier's decision regions plus the data points."""
    pad = 0.3
    x_min, x_max = x[:, 0].min() - pad, x[:, 0].max() + pad
    y_min, y_max = x[:, 1].min() - pad, x[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model(Tensor(grid)).data
    zz = np.argmax(scores, axis=1).reshape(xx.shape)

    n_classes = int(zz.max()) + 1
    levels = np.arange(-0.5, n_classes + 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, zz, levels=levels, alpha=0.35, cmap="viridis")
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", s=18,
               edgecolors="k", linewidths=0.3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_loss_curve(losses, path, title=""):
    """Plot a training loss curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, color="#3b6ea5")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_regression(x, y, x_line, y_line, path, title=""):
    """Plot noisy 1-D data points and the model's fitted curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=14, alpha=0.6, label="data", color="#5b8c5a")
    ax.plot(x_line, y_line, color="#c1432e", linewidth=2.2, label="model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

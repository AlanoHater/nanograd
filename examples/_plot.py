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


def save_accuracy_curves(train_acc, val_acc, path, title=""):
    """Plot train vs validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_acc, label="train", color="#3b6ea5")
    ax.plot(val_acc, label="validation", color="#c1432e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, n_classes, path, title=""):
    """Plot a confusion matrix as an annotated heatmap."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_digit_predictions(images, y_true, y_pred, path, title="", rows=4, cols=8):
    """Show a grid of digit images with predicted/true labels (red if wrong)."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= len(images):
            continue
        ax.imshow(images[k], cmap="gray_r")
        correct = int(y_pred[k]) == int(y_true[k])
        ax.set_title(f"{int(y_pred[k])} ({int(y_true[k])})", fontsize=8,
                     color="#2a7a2a" if correct else "#c1432e")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_attention_grid(maps, tokens, path, title=""):
    """Plot attention weights as a grid of heatmaps (rows=layers, cols=heads).

    ``maps`` is a list (one per layer) of arrays shaped ``(n_heads, T, T)`` for a
    single example. Brighter cells mean the query position (row) attends more to
    the key position (column).
    """
    n_layers = len(maps)
    n_heads = maps[0].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(n_heads * 2.4, n_layers * 2.4), squeeze=False)
    labels = [str(t) for t in tokens]
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            ax.imshow(maps[layer][head], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6)
            if layer == 0:
                ax.set_title(f"head {head + 1}", fontsize=9)
            if head == 0:
                ax.set_ylabel(f"layer {layer + 1}\nquery", fontsize=8)
            if layer == n_layers - 1:
                ax.set_xlabel("key", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_feature_maps(image, maps, path, title=""):
    """Show an input image alongside the feature maps a conv layer produced."""
    n = maps.shape[0]
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 1.25, 1.7))
    axes[0].imshow(image, cmap="gray_r")
    axes[0].set_title("input", fontsize=8)
    axes[0].axis("off")
    for i in range(n):
        axes[i + 1].imshow(maps[i], cmap="viridis")
        axes[i + 1].set_title(f"map {i + 1}", fontsize=7)
        axes[i + 1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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

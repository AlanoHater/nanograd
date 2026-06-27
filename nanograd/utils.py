"""Synthetic datasets and small helpers used by the examples and tests."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._random import rng
from .tensor import Tensor


def make_spiral(n_points: int = 100, n_classes: int = 3,
                noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the classic intertwined-spiral classification dataset.

    Returns ``X`` of shape ``(n_points * n_classes, 2)`` and integer labels
    ``y`` of shape ``(n_points * n_classes,)``. The classes are not linearly
    separable, which makes this a good test for a non-linear model.
    """
    generator = rng()
    x = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)
    for c in range(n_classes):
        ix = np.arange(n_points * c, n_points * (c + 1))
        radius = np.linspace(0.0, 1.0, n_points)
        theta = (np.linspace(c * 4.0, (c + 1) * 4.0, n_points)
                 + generator.standard_normal(n_points) * noise)
        x[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[ix] = c
    return x, y


def make_moons(n_samples: int = 200, noise: float = 0.1
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half-moons (a binary classification task)."""
    generator = rng()
    n_out = n_samples // 2
    n_in = n_samples - n_out
    outer = np.c_[np.cos(np.linspace(0, np.pi, n_out)),
                  np.sin(np.linspace(0, np.pi, n_out))]
    inner = np.c_[1 - np.cos(np.linspace(0, np.pi, n_in)),
                  1 - np.sin(np.linspace(0, np.pi, n_in)) - 0.5]
    x = np.vstack([outer, inner])
    y = np.array([0] * n_out + [1] * n_in)
    x += generator.standard_normal(x.shape) * noise
    return x, y


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    out = np.zeros((labels.shape[0], n_classes))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def accuracy(scores, labels) -> float:
    """Fraction of correct predictions given class scores/logits."""
    if isinstance(scores, Tensor):
        scores = scores.data
    preds = np.argmax(scores, axis=1)
    return float(np.mean(preds == np.asarray(labels)))


def train_test_split(x: np.ndarray, y: np.ndarray, test_frac: float = 0.2):
    """Shuffle and split arrays into train/test partitions."""
    n = x.shape[0]
    idx = rng().permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int,
                        shuffle: bool = True):
    """Yield ``(x_batch, y_batch)`` tuples covering the dataset once (an epoch)."""
    n = x.shape[0]
    indices = rng().permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def standardize(x: np.ndarray, mean=None, std=None):
    """Zero-mean, unit-variance scaling. Returns ``(x_scaled, mean, std)``."""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0) + 1e-8
    return (x - mean) / std, mean, std


# --------------------------------------------------------------------------- #
# Toy algorithmic task: sort a sequence of digits (used by the Transformer demo)
# --------------------------------------------------------------------------- #
def make_sort_dataset(n_samples: int, length: int = 6, num_digits: int = 3):
    """Build a next-token-prediction dataset for sorting digit sequences.

    Each example concatenates an unsorted sequence with its sorted version. The
    model is trained to predict the next token; loss on the unsorted "prompt"
    region is masked out with ``-1`` so the model is only graded on producing
    the sorted answer.

    Returns ``(x, y)`` of shape ``(n_samples, 2*length - 1)``.
    """
    inp = rng().integers(0, num_digits, size=(n_samples, length))
    sol = np.sort(inp, axis=1)
    cat = np.concatenate([inp, sol], axis=1)        # (n, 2L)
    x = cat[:, :-1].copy()                          # (n, 2L-1)
    y = cat[:, 1:].copy()                           # (n, 2L-1)
    y[:, :length - 1] = -1                          # ignore loss on the prompt
    return x, y


def make_recall_dataset(n_samples: int, length: int = 12):
    """Sequence-memory task: the label is the *first* bit of a binary sequence.

    The model sees the whole sequence and must reproduce its first bit at the
    final step, so it has to carry information across all ``length`` time steps —
    a clean test of recurrent memory. Returns ``x`` of shape
    ``(n_samples, length, 1)`` and labels ``(n_samples,)``.
    """
    bits = rng().integers(0, 2, size=(n_samples, length))
    y = bits[:, 0].copy()
    x = bits.astype(np.float64)[:, :, None]
    return x, y


def sort_accuracy(model, n_samples: int = 512, length: int = 6, num_digits: int = 3):
    """Autoregressively decode sorted sequences and measure accuracy.

    Returns ``(exact_match, per_token)`` accuracy. ``model(idx)`` must accept an
    integer array of shape ``(n, block_size)`` and return logits ``(n, block, V)``.
    """
    block = 2 * length - 1
    inp = rng().integers(0, num_digits, size=(n_samples, length))
    sol = np.sort(inp, axis=1)

    x = np.zeros((n_samples, block), dtype=int)
    x[:, :length] = inp
    last = None
    for pos in range(length - 1, block):  # positions whose next-token is the answer
        logits = model(x).data            # (n, block, V)
        nxt = np.argmax(logits[:, pos, :], axis=1)
        if pos + 1 < block:
            x[:, pos + 1] = nxt
        else:
            last = nxt
    pred = np.concatenate([x[:, length:block], last[:, None]], axis=1)  # (n, L)

    exact = float(np.mean(np.all(pred == sol, axis=1)))
    per_token = float(np.mean(pred == sol))
    return exact, per_token
